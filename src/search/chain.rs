use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::core::git;
use crate::core::types::*;
use crate::index::{IndexReader, MmapIndexReader};
use super::vector_search;

/// A search result from chain search, with path and commit already resolved.
#[derive(Debug, Clone)]
pub struct ChainSearchResult {
    /// File path relative to the repository root.
    pub path: String,
    /// Byte offset of the chunk within the file.
    pub byte_offset: u32,
    /// Byte length of the chunk.
    pub chunk_len: u16,
    /// Hamming distance from the query embedding.
    pub hamming_distance: u32,
    /// The commit hash (hex) this result came from. Use this to read
    /// the blob content for reranking.
    pub commit_hex: String,
}

/// Search a chain of indexes (newest first) for the top N results,
/// with proper shadowing of files covered by newer delta indexes.
///
/// Indexes are searched in parallel for speed, then results are
/// filtered sequentially to apply shadowing logic.
pub fn chain_search(
    readers: &[MmapIndexReader],
    repo: &gix::Repository,
    query_embedding: &BinaryEmbedding,
    top_n: usize,
) -> Result<Vec<ChainSearchResult>> {
    if readers.is_empty() || top_n == 0 {
        return Ok(Vec::new());
    }

    // Phase 1: Search all indexes in parallel.
    let per_index_results: Vec<Vec<VectorSearchResult>> = readers
        .par_iter()
        .map(|reader| {
            let n = top_n.min(reader.embedding_count());
            vector_search(query_embedding, reader, n).unwrap_or_default()
        })
        .collect();

    // Phase 2: Walk newest → oldest, resolve paths, filter shadowed.
    let mut shadowed_paths: HashSet<String> = HashSet::new();
    let mut all_results: Vec<ChainSearchResult> = Vec::new();

    for (i, (reader, results)) in readers.iter().zip(per_index_results).enumerate() {
        let commit_hex = commit_hash_to_hex(&reader.header().commit_hash);
        let entries = git::entries_at_commit(repo, &commit_hex)
            .with_context(|| format!("failed to get entries at commit {}", &commit_hex))?;

        // Build position → path map for this index's commit.
        let pos_to_path: HashMap<usize, &str> = entries
            .iter()
            .map(|(pos, path)| (*pos, path.as_str()))
            .collect();

        // Filter results: keep only those whose path is not shadowed.
        for result in results {
            let file_pos = result.chunk_ref.file_index as usize;
            if let Some(path) = pos_to_path.get(&file_pos) {
                if !shadowed_paths.contains(*path) {
                    all_results.push(ChainSearchResult {
                        path: path.to_string(),
                        byte_offset: result.chunk_ref.byte_offset,
                        chunk_len: result.chunk_ref.chunk_len,
                        hamming_distance: result.hamming_distance,
                        commit_hex: commit_hex.clone(),
                    });
                }
            }
            // Results with unresolvable positions (e.g. tombstones
            // with synthetic positions) are silently dropped — they
            // have 0 chunks so vector_search never returns them anyway.
        }

        // Update shadowed paths for subsequent (older) indexes.
        // Use changed_files_between to capture both modified AND
        // deleted files, including tombstones whose paths can't be
        // resolved from git_index_positions alone.
        let parent_hex = commit_hash_to_hex(reader.parent_hash());
        if !parent_hex.is_empty() && i < readers.len() - 1 {
            match git::changed_files_between(repo, &parent_hex, &commit_hex) {
                Ok(diff) => {
                    shadowed_paths.extend(diff.changed);
                    shadowed_paths.extend(diff.deleted);
                }
                Err(e) => {
                    eprintln!(
                        "  warn: could not compute diff for shadowing: {}",
                        e
                    );
                    // If we can't compute the diff, conservatively
                    // continue without shadowing. This means some
                    // stale results might appear, but we won't lose
                    // valid results.
                }
            }
        }
    }

    // Sort by hamming distance (closest first) and take top_n.
    all_results.sort_by_key(|r| r.hamming_distance);
    all_results.truncate(top_n);
    Ok(all_results)
}

/// Load the chain of index files starting from the latest, walking
/// the parent hash links. Returns readers ordered newest-first.
///
/// The chain stops when a root index (parent_hash all zeroes) is
/// reached, or when the parent index file cannot be found.
pub fn load_index_chain(kb_dir: &Path) -> Result<Vec<MmapIndexReader>> {
    let latest_path = find_latest_index(kb_dir)?;
    let mut chain = Vec::new();
    let mut current_path = latest_path;

    loop {
        let reader = MmapIndexReader::open(&current_path)?;
        let parent_hex = commit_hash_to_hex(reader.parent_hash());
        chain.push(reader);

        if parent_hex.is_empty() {
            break; // Root index, chain complete.
        }

        match find_index_by_commit(kb_dir, &parent_hex) {
            Some(path) => current_path = path,
            None => break, // Parent not found, chain ends here.
        }
    }

    Ok(chain)
}

/// Extract the commit hash hex string from a GitHash (stored as ASCII
/// hex bytes padded with zeroes).
fn commit_hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
    String::from_utf8_lossy(&hash[..end]).into_owned()
}

/// Find the most recent .kbi file in a directory by modification time.
fn find_latest_index(kb_dir: &Path) -> Result<PathBuf> {
    if !kb_dir.exists() {
        anyhow::bail!(
            "No index directory found at {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    let mut kbi_files: Vec<_> = std::fs::read_dir(kb_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "kbi")
        })
        .collect();

    if kbi_files.is_empty() {
        anyhow::bail!(
            "No index files found in {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    kbi_files.sort_by_key(|e| {
        std::cmp::Reverse(
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    Ok(kbi_files[0].path())
}

/// Find an index file whose commit hash matches the given hex string.
/// Tries filename prefix match first, then falls back to scanning
/// all .kbi file headers.
fn find_index_by_commit(kb_dir: &Path, commit_hex: &str) -> Option<PathBuf> {
    // Index files are named <commit_hash_hex[..16]>.kbi
    let prefix = &commit_hex[..commit_hex.len().min(16)];
    let candidate = kb_dir.join(format!("{}.kbi", prefix));
    if candidate.exists() {
        return Some(candidate);
    }

    // Fallback: scan all .kbi files and check headers.
    for entry in std::fs::read_dir(kb_dir).ok()? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "kbi") {
                if let Ok(reader) = MmapIndexReader::open(&path) {
                    let hash = commit_hash_to_hex(&reader.header().commit_hash);
                    if hash == commit_hex {
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use crate::index::format::write_index;
    use crate::index::{ChunkEntry, FileEntry};

    fn make_embedding(val: u8) -> BinaryEmbedding {
        [val; EMBEDDING_BYTES]
    }

    fn make_git_hash(hex: &str) -> GitHash {
        let mut hash: GitHash = [0; MAX_HASH_LEN];
        let bytes = hex.as_bytes();
        hash[..bytes.len().min(MAX_HASH_LEN)]
            .copy_from_slice(&bytes[..bytes.len().min(MAX_HASH_LEN)]);
        hash
    }

    fn make_header(commit: &str, parent: &str) -> IndexHeader {
        IndexHeader {
            version: 1,
            commit_hash: make_git_hash(commit),
            parent_hash: make_git_hash(parent),
        }
    }

    #[test]
    fn load_chain_single_root_index() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        let header = make_header("aaaa", "");
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 100,
                embedding: make_embedding(0xAA),
            }],
        }];
        write_index(&kb_dir.join("aaaa.kbi"), &header, &entries).unwrap();

        let chain = load_index_chain(kb_dir).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(
            commit_hash_to_hex(&chain[0].header().commit_hash),
            "aaaa"
        );
    }

    #[test]
    fn load_chain_two_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        // Root index at commit "aaaa"
        let root_header = make_header("aaaa", "");
        let root_entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 100,
                embedding: make_embedding(0xAA),
            }],
        }];
        write_index(&kb_dir.join("aaaa.kbi"), &root_header, &root_entries).unwrap();

        // Delta index at commit "bbbb" with parent "aaaa"
        // Sleep briefly to ensure different mtime
        std::thread::sleep(std::time::Duration::from_millis(10));
        let delta_header = make_header("bbbb", "aaaa");
        let delta_entries = vec![FileEntry {
            git_index_position: 1,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 200,
                embedding: make_embedding(0xBB),
            }],
        }];
        write_index(&kb_dir.join("bbbb.kbi"), &delta_header, &delta_entries).unwrap();

        let chain = load_index_chain(kb_dir).unwrap();
        assert_eq!(chain.len(), 2);
        // Newest first
        assert_eq!(
            commit_hash_to_hex(&chain[0].header().commit_hash),
            "bbbb"
        );
        assert_eq!(
            commit_hash_to_hex(&chain[1].header().commit_hash),
            "aaaa"
        );
    }

    #[test]
    fn load_chain_broken_link() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        // Delta index pointing to a nonexistent parent
        let header = make_header("cccc", "missing_parent");
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 50,
                embedding: make_embedding(0xCC),
            }],
        }];
        write_index(&kb_dir.join("cccc.kbi"), &header, &entries).unwrap();

        let chain = load_index_chain(kb_dir).unwrap();
        // Chain should contain just the one index (broken link)
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn commit_hash_to_hex_basic() {
        let hash = make_git_hash("deadbeef");
        assert_eq!(commit_hash_to_hex(&hash), "deadbeef");
    }

    #[test]
    fn commit_hash_to_hex_empty() {
        let hash: GitHash = [0; MAX_HASH_LEN];
        assert_eq!(commit_hash_to_hex(&hash), "");
    }

    #[test]
    fn find_index_by_commit_prefix_match() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        let header = make_header("abcdef1234567890extra", "");
        write_index(
            &kb_dir.join("abcdef1234567890.kbi"),
            &header,
            &[],
        ).unwrap();

        let found = find_index_by_commit(kb_dir, "abcdef1234567890extra");
        assert!(found.is_some());
    }

    #[test]
    fn find_index_by_commit_header_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        // File named differently than expected
        let header = make_header("deadbeef", "");
        write_index(&kb_dir.join("other_name.kbi"), &header, &[]).unwrap();

        let found = find_index_by_commit(kb_dir, "deadbeef");
        assert!(found.is_some());
    }
}

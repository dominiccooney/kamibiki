use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::vector_search;
use crate::core::git;
use crate::core::types::*;
use crate::index::{IndexReader, MmapIndexReader};

/// The result of loading an index chain: the chain of readers plus
/// how many commits the starting point is ahead of the nearest
/// indexed commit.
pub struct IndexChain {
    /// Index readers ordered newest-first.
    pub readers: Vec<MmapIndexReader>,
    /// Number of commits the starting point is ahead of the newest
    /// index in the chain. Zero means the index matches the
    /// starting commit.
    pub commits_behind: usize,
}

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

/// A pair of repository-root-relative directory prefixes that scope a
/// search: results must live under one of `include` (when non-empty)
/// and must not live under any of `exclude`. Exclusion takes
/// precedence over inclusion.
///
/// Prefixes are interpreted as git-style, forward-slash, repo-relative
/// directory paths (e.g. `src/search`). They are matched on whole path
/// segments, so `src/app` matches `src/app/x.rs` but not
/// `src/apple/x.rs`.
#[derive(Debug, Default, Clone)]
pub struct DirFilter {
    /// Restrict to files under these directories. Empty = no restriction.
    pub include: Vec<String>,
    /// Exclude files under these directories. Always applied.
    pub exclude: Vec<String>,
}

impl DirFilter {
    /// Build a filter from raw include/exclude prefixes, normalizing
    /// each by trimming surrounding slashes and dropping empties.
    ///
    /// Prefixes here are assumed to already be repository-root-relative
    /// (git-style). Use [`DirFilter::from_user_input`] to accept the
    /// looser set of paths a human might type.
    pub fn new(include: &[String], exclude: &[String]) -> Self {
        DirFilter {
            include: normalize_dirs(include),
            exclude: normalize_dirs(exclude),
        }
    }

    /// Build a filter from directory arguments as a user might type
    /// them — absolute paths, paths relative to their current working
    /// directory, or already-repo-relative paths — by relativizing
    /// each against the repository root.
    ///
    /// `repo_root` is the worktree root (where repo-relative chunk
    /// paths are anchored) and `cwd` is the directory the user is
    /// operating from (used to resolve relative inputs). Both should be
    /// absolute; callers typically canonicalize them first so symlinks
    /// resolve consistently. The user-supplied paths themselves are
    /// only normalized *lexically* (`.`/`..` collapsed as text), never
    /// canonicalized — so a directory that does not exist on disk
    /// (because an earlier `--commit` is being searched) still
    /// relativizes correctly.
    ///
    /// Inputs that resolve to a location outside the repository are
    /// dropped: they can never match a repo-relative chunk path.
    pub fn from_user_input(
        include: &[String],
        exclude: &[String],
        repo_root: &Path,
        cwd: &Path,
    ) -> Self {
        let relativize = |dirs: &[String]| -> Vec<String> {
            dirs.iter()
                .filter_map(|d| relativize_dir(d, repo_root, cwd))
                .collect()
        };
        // Re-run through `new` so the repo-relative results are
        // normalized (trimmed, `.`/empty dropped) identically.
        DirFilter::new(&relativize(include), &relativize(exclude))
    }

    /// Returns true when no include or exclude prefixes are set, so the
    /// filter accepts every path and matching can be skipped entirely.
    pub fn is_empty(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty()
    }

    /// Decide whether a repo-relative `path` passes the filter.
    ///
    /// A path is accepted when it is *not* under any `exclude` prefix
    /// (exclusion wins) and, if `include` is non-empty, is under at
    /// least one `include` prefix.
    pub fn accepts(&self, path: &str) -> bool {
        if self.exclude.iter().any(|d| path_under_dir(path, d)) {
            return false;
        }
        if self.include.is_empty() {
            return true;
        }
        self.include.iter().any(|d| path_under_dir(path, d))
    }
}

/// Normalize a list of directory prefixes: trim surrounding slashes so
/// a chunk path matches when it equals the directory or is nested
/// beneath it, and drop entries that normalize to empty (e.g. `.` or
/// the repo root itself, which would match everything).
fn normalize_dirs(dirs: &[String]) -> Vec<String> {
    dirs.iter()
        .map(|d| d.trim_matches('/').to_string())
        .filter(|d| !d.is_empty() && d != ".")
        .collect()
}

/// Whole-segment prefix match: true when `path` equals `dir` or sits
/// beneath it. `dir` is assumed already normalized (no surrounding
/// slashes). The trailing-slash comparison prevents `src/app` from
/// matching `src/apple`.
fn path_under_dir(path: &str, dir: &str) -> bool {
    if path == dir {
        return true;
    }
    let mut prefix = String::with_capacity(dir.len() + 1);
    prefix.push_str(dir);
    prefix.push('/');
    path.starts_with(&prefix)
}

/// Collapse `.` and `..` segments in a path purely lexically (without
/// touching the filesystem), returning the cleaned, forward-slash
/// component list. A leading `..` that would escape the path is kept
/// so callers can detect an out-of-bounds result.
fn lexical_components(path: &Path) -> Vec<String> {
    use std::path::Component;
    let mut out: Vec<String> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // Pop a real segment if there is one; otherwise keep the
                // `..` so an escaping path stays detectably out of range.
                if matches!(out.last().map(String::as_str), Some("..") | None) {
                    out.push("..".to_string());
                } else {
                    out.pop();
                }
            }
            Component::Normal(s) => out.push(s.to_string_lossy().into_owned()),
            // Root / prefix components are dropped: we only care about
            // the portion of the path relative to the repo root.
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    out
}

/// Relativize a single user-supplied directory argument to a
/// repository-root-relative, forward-slash prefix.
///
/// * Absolute inputs are made relative to `repo_root`.
/// * Relative inputs are first tried relative to `cwd`; if that lands
///   inside the repo, that interpretation wins. Otherwise the input is
///   treated as already repo-root-relative.
///
/// All path math is lexical (no `canonicalize`), so inputs that do not
/// exist on disk still relativize. Returns `None` when the result
/// would fall outside the repository.
fn relativize_dir(input: &str, repo_root: &Path, cwd: &Path) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let raw = Path::new(trimmed);

    // Resolve to an absolute, lexically-cleaned component list.
    let abs_components = if raw.is_absolute() {
        lexical_components(raw)
    } else {
        // Prefer cwd-relative resolution; fall back to repo-relative.
        let cwd_joined = lexical_components(&cwd.join(raw));
        let root_components = lexical_components(repo_root);
        if starts_with_components(&cwd_joined, &root_components) {
            cwd_joined
        } else {
            // Treat as already repo-relative: anchor at the repo root
            // and re-collapse so any `.`/`..` resolve against the root
            // (and an escaping input stays outside, to be rejected
            // below).
            lexical_components(&repo_root.join(raw))
        }
    };

    let root_components = lexical_components(repo_root);
    strip_prefix_components(&abs_components, &root_components).map(|rel| rel.join("/"))
}

/// True when `components` begins with every element of `prefix`.
fn starts_with_components(components: &[String], prefix: &[String]) -> bool {
    components.len() >= prefix.len() && components[..prefix.len()] == *prefix
}

/// Strip `prefix` from the front of `components`, returning the
/// remainder. Returns `None` when `components` does not start with
/// `prefix` (i.e. the path is outside the repository root).
fn strip_prefix_components(components: &[String], prefix: &[String]) -> Option<Vec<String>> {
    if starts_with_components(components, prefix) {
        Some(components[prefix.len()..].to_vec())
    } else {
        None
    }
}

/// Search a chain of indexes (newest first) for the top N results,
/// with proper shadowing of files covered by newer delta indexes.
///
/// Indexes are searched in parallel for speed, then results are
/// filtered sequentially to apply shadowing logic.
///
/// `dir_filter` optionally restricts results to (or away from)
/// specific directories. When it is empty, the whole repository is
/// searched. See [`DirFilter`] for matching semantics.
pub fn chain_search(
    readers: &[MmapIndexReader],
    repo: &gix::Repository,
    query_embedding: &BinaryEmbedding,
    top_n: usize,
    dir_filter: &DirFilter,
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

        // Filter results: keep only those whose path is not shadowed
        // and that pass the directory filter.
        for result in results {
            let file_pos = result.chunk_ref.file_index as usize;
            if let Some(path) = pos_to_path.get(&file_pos) {
                if !shadowed_paths.contains(*path) && dir_filter.accepts(path) {
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
                    eprintln!("  warn: could not compute diff for shadowing: {}", e);
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

/// Load the chain of index files starting from the most recent
/// ancestor of the given commit (or HEAD if `None`) that has been
/// indexed, then walking the parent hash links. Returns an
/// [`IndexChain`] containing readers ordered newest-first and the
/// number of commits the starting point is ahead of the nearest
/// indexed commit.
///
/// The chain start is found by walking the starting commit's git
/// ancestors (a fast DAG traversal) and checking each commit
/// against the set of indexed commits (read from .kbi file
/// headers, 65 bytes each). The chain then follows the
/// parent_hash links embedded in each index file until a root
/// index or missing parent is reached.
///
/// `commit_hex` is the resolved commit hash to start from. If
/// `None`, HEAD is used.
pub fn load_index_chain(
    kb_dir: &Path,
    repo: &gix::Repository,
    commit_hex: Option<&str>,
) -> Result<IndexChain> {
    let (start_path, commits_behind) = find_chain_start(kb_dir, repo, commit_hex)?;
    let readers = walk_index_chain(kb_dir, start_path)?;
    Ok(IndexChain {
        readers,
        commits_behind,
    })
}

/// Walk a chain of indexes from a starting .kbi file, following
/// parent hash links. Returns readers ordered newest-first.
///
/// The chain stops when a root index (parent_hash all zeroes) is
/// reached, or when the parent index file cannot be found.
fn walk_index_chain(kb_dir: &Path, start_path: PathBuf) -> Result<Vec<MmapIndexReader>> {
    let mut chain = Vec::new();
    let mut current_path = start_path;

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

/// Find the best starting index by walking the given commit's (or
/// HEAD's) git ancestors until we find a commit that has a .kbi
/// file.
///
/// Returns `(path, commits_behind)` where `commits_behind` is the
/// number of commits the starting point is ahead of the indexed
/// commit (0 when the index matches the starting commit).
///
/// `commit_hex` is the resolved commit hash to start from. If
/// `None`, HEAD is used.
///
/// This reads only the first 65 bytes of each .kbi file (version
/// byte + commit hash) and walks the git commit DAG, which reads
/// only small commit objects. Both operations are fast even for
/// large repositories.
fn find_chain_start(
    kb_dir: &Path,
    repo: &gix::Repository,
    commit_hex: Option<&str>,
) -> Result<(PathBuf, usize)> {
    if !kb_dir.exists() {
        anyhow::bail!(
            "No index directory found at {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    // Build commit_hex → path map from all .kbi file headers.
    let indexed = scan_indexed_commits(kb_dir)?;
    if indexed.is_empty() {
        anyhow::bail!(
            "No index files found in {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    // Resolve the starting commit.
    let start_id = match commit_hex {
        Some(hex) => {
            let oid = gix::ObjectId::from_hex(hex.as_bytes()).context("invalid commit hash hex")?;
            oid
        }
        None => repo
            .head_commit()
            .context("failed to resolve HEAD commit")?
            .id()
            .into(),
    };
    let start_hex = start_id.to_hex().to_string();

    // Check the starting commit itself first (fast path — the
    // common case after running `kb index`).
    if let Some(path) = indexed.get(&start_hex) {
        return Ok((path.clone(), 0));
    }

    // Walk ancestors in topological order (newest first).
    // The rev_walk includes the starting commit itself as the first
    // entry, which we already checked above, so we count each step
    // to track how many commits the start is ahead of the indexed
    // commit.
    let walk = repo
        .rev_walk([start_id])
        .all()
        .context("failed to start ancestor walk")?;

    let mut commits_walked: usize = 0;
    for info in walk {
        let info = info.context("error during ancestor walk")?;
        let hex = info.id.to_hex().to_string();
        commits_walked += 1;
        if let Some(path) = indexed.get(&hex) {
            return Ok((path.clone(), commits_walked - 1));
        }
    }

    let ref_label = commit_hex.unwrap_or("HEAD");
    anyhow::bail!(
        "No indexed ancestor of {} found in {}. Run 'kb index' first.",
        ref_label,
        kb_dir.display()
    );
}

/// Scan all .kbi files in a directory and build a map of
/// commit hash (hex) → file path. Reads only the first 65 bytes
/// of each file (version byte + commit hash).
fn scan_indexed_commits(kb_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(kb_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "kbi") {
            if let Ok(hash) = read_commit_hash_from_header(&path) {
                if !hash.is_empty() {
                    map.insert(hash, path);
                }
            }
        }
    }
    Ok(map)
}

/// Read just the commit hash from a .kbi file header without
/// parsing the rest of the file. Reads exactly `1 + MAX_HASH_LEN`
/// (65) bytes.
fn read_commit_hash_from_header(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 1 + MAX_HASH_LEN];
    file.read_exact(&mut buf)?;
    // commit_hash starts at byte 1 (after version byte).
    let hash_bytes = &buf[1..];
    let end = hash_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(MAX_HASH_LEN);
    Ok(String::from_utf8_lossy(&hash_bytes[..end]).into_owned())
}

/// Extract the commit hash hex string from a GitHash (stored as ASCII
/// hex bytes padded with zeroes).
fn commit_hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
    String::from_utf8_lossy(&hash[..end]).into_owned()
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
    fn walk_chain_single_root_index() {
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
        let start = kb_dir.join("aaaa.kbi");
        write_index(&start, &header, &entries).unwrap();

        let chain = walk_index_chain(kb_dir, start).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(commit_hash_to_hex(&chain[0].header().commit_hash), "aaaa");
    }

    #[test]
    fn walk_chain_two_indexes() {
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
        let delta_header = make_header("bbbb", "aaaa");
        let delta_entries = vec![FileEntry {
            git_index_position: 1,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 200,
                embedding: make_embedding(0xBB),
            }],
        }];
        let start = kb_dir.join("bbbb.kbi");
        write_index(&start, &delta_header, &delta_entries).unwrap();

        let chain = walk_index_chain(kb_dir, start).unwrap();
        assert_eq!(chain.len(), 2);
        // Newest first
        assert_eq!(commit_hash_to_hex(&chain[0].header().commit_hash), "bbbb");
        assert_eq!(commit_hash_to_hex(&chain[1].header().commit_hash), "aaaa");
    }

    #[test]
    fn walk_chain_broken_link() {
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
        let start = kb_dir.join("cccc.kbi");
        write_index(&start, &header, &entries).unwrap();

        let chain = walk_index_chain(kb_dir, start).unwrap();
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
        write_index(&kb_dir.join("abcdef1234567890.kbi"), &header, &[]).unwrap();

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

    #[test]
    fn scan_indexed_commits_finds_all() {
        let dir = tempfile::tempdir().unwrap();
        let kb_dir = dir.path();

        write_index(&kb_dir.join("aaaa.kbi"), &make_header("aaaa1111", ""), &[]).unwrap();
        write_index(
            &kb_dir.join("bbbb.kbi"),
            &make_header("bbbb2222", "aaaa1111"),
            &[],
        )
        .unwrap();

        let indexed = scan_indexed_commits(kb_dir).unwrap();
        assert_eq!(indexed.len(), 2);
        assert!(indexed.contains_key("aaaa1111"));
        assert!(indexed.contains_key("bbbb2222"));
    }

    #[test]
    fn dir_filter_empty_accepts_everything() {
        let f = DirFilter::new(&[], &[]);
        assert!(f.is_empty());
        assert!(f.accepts("src/main.rs"));
        assert!(f.accepts("README.md"));
    }

    #[test]
    fn dir_filter_include_scopes_to_subtree() {
        let f = DirFilter::new(&["src/search".to_string()], &[]);
        assert!(!f.is_empty());
        assert!(f.accepts("src/search/chain.rs"));
        assert!(f.accepts("src/search")); // the directory itself
        assert!(!f.accepts("src/server.rs"));
        assert!(!f.accepts("README.md"));
    }

    #[test]
    fn dir_filter_include_matches_whole_segments() {
        // `src/app` must not match `src/apple`.
        let f = DirFilter::new(&["src/app".to_string()], &[]);
        assert!(f.accepts("src/app/main.rs"));
        assert!(!f.accepts("src/apple/main.rs"));
    }

    #[test]
    fn dir_filter_exclude_removes_subtree() {
        let f = DirFilter::new(&[], &["target".to_string()]);
        assert!(f.accepts("src/main.rs"));
        assert!(!f.accepts("target/debug/foo"));
        assert!(!f.accepts("target")); // the directory itself
    }

    #[test]
    fn dir_filter_exclude_wins_over_include() {
        // Precedence: a path under both include and exclude is excluded.
        let f = DirFilter::new(&["src".to_string()], &["src/generated".to_string()]);
        assert!(f.accepts("src/main.rs"));
        assert!(!f.accepts("src/generated/proto.rs"));
        assert!(!f.accepts("docs/guide.md")); // not under include
    }

    #[test]
    fn relativize_absolute_path_inside_repo() {
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj/src");
        assert_eq!(
            relativize_dir("/home/me/proj/src/search", root, cwd),
            Some("src/search".to_string())
        );
    }

    #[test]
    fn relativize_relative_path_against_cwd() {
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj/src");
        // "search" relative to cwd is proj/src/search.
        assert_eq!(
            relativize_dir("search", root, cwd),
            Some("src/search".to_string())
        );
        // ".." walks up from src to the repo root.
        assert_eq!(relativize_dir("../docs", root, cwd), Some("docs".to_string()));
    }

    #[test]
    fn relativize_repo_relative_fallback() {
        // When cwd is the repo root, a bare relative path is already
        // repo-relative.
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj");
        assert_eq!(
            relativize_dir("src/search", root, cwd),
            Some("src/search".to_string())
        );
    }

    #[test]
    fn relativize_nonexistent_path_for_earlier_commit() {
        // The path need not exist on disk: relativization is lexical.
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj");
        assert_eq!(
            relativize_dir("legacy/removed_module", root, cwd),
            Some("legacy/removed_module".to_string())
        );
    }

    #[test]
    fn relativize_outside_repo_is_dropped() {
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj/src");
        // Absolute path outside the repo.
        assert_eq!(relativize_dir("/etc/passwd", root, cwd), None);
        // Relative path that escapes the repo root.
        assert_eq!(relativize_dir("../../other", root, cwd), None);
    }

    #[test]
    fn from_user_input_builds_relativized_filter() {
        let root = Path::new("/home/me/proj");
        let cwd = Path::new("/home/me/proj/src");
        let f = DirFilter::from_user_input(
            &["/home/me/proj/src/search".to_string(), "index".to_string()],
            &["../target".to_string()],
            root,
            cwd,
        );
        assert_eq!(
            f.include,
            vec!["src/search".to_string(), "src/index".to_string()]
        );
        assert_eq!(f.exclude, vec!["target".to_string()]);
        assert!(f.accepts("src/search/vector.rs"));
        assert!(f.accepts("src/index/mod.rs"));
        assert!(!f.accepts("src/server.rs"));
        assert!(!f.accepts("target/debug/x"));
    }

    #[test]
    fn dir_filter_normalizes_surrounding_slashes_and_dot() {
        // Leading/trailing slashes are trimmed; "." and "" are dropped.
        let f = DirFilter::new(
            &["/src/search/".to_string(), ".".to_string(), String::new()],
            &[],
        );
        assert_eq!(f.include, vec!["src/search".to_string()]);
        assert!(f.accepts("src/search/vector.rs"));
        assert!(!f.accepts("src/index/mod.rs"));
    }

    #[test]
    fn read_commit_hash_from_header_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");

        let header = make_header("deadbeef42", "");
        write_index(&path, &header, &[]).unwrap();

        let hash = read_commit_hash_from_header(&path).unwrap();
        assert_eq!(hash, "deadbeef42");
    }
}

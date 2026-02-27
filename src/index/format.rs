use std::path::Path;

use anyhow::{Result, ensure};

use crate::core::types::{BinaryEmbedding, IndexHeader, EMBEDDING_ALIGNMENT, EMBEDDING_BYTES, MAX_HASH_LEN};
use super::{FileEntry, FileChunkInfo};

/// Size of the serialized header: version(1) + commit_hash(64) + parent_hash(64).
pub const HEADER_SIZE: usize = 1 + MAX_HASH_LEN + MAX_HASH_LEN;

/// Write an index file at the given path.
pub fn write_index(path: &Path, header: &IndexHeader, file_entries: &[FileEntry]) -> Result<()> {
    let data = encode_index(header, file_entries)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &data)?;
    Ok(())
}

/// Encode a complete index into a byte vector.
pub fn encode_index(header: &IndexHeader, file_entries: &[FileEntry]) -> Result<Vec<u8>> {
    // Validate: file entries must be sorted by git_index_position, unique.
    for w in file_entries.windows(2) {
        ensure!(
            w[1].git_index_position > w[0].git_index_position,
            "file entries must be sorted by git_index_position and unique"
        );
    }

    let total_chunks: usize = file_entries.iter().map(|e| e.chunks.len()).sum();

    let mut buf = Vec::with_capacity(
        HEADER_SIZE
            + file_entries.len() * 6
            + total_chunks * 2
            + EMBEDDING_ALIGNMENT
            + total_chunks * EMBEDDING_BYTES,
    );

    // --- Header ---
    buf.push(header.version);
    buf.extend_from_slice(&header.commit_hash);
    buf.extend_from_slice(&header.parent_hash);

    // --- Offset table ---
    encode_offset_table(&mut buf, file_entries);

    // --- Chunk count table (u16 LE per included file) ---
    for entry in file_entries {
        let count = entry.chunks.len();
        ensure!(count <= 65534, "file has too many chunks: {}", count);
        buf.extend_from_slice(&(count as u16).to_le_bytes());
    }

    // --- Length table (u16 LE per chunk) ---
    for entry in file_entries {
        for chunk in &entry.chunks {
            buf.extend_from_slice(&chunk.chunk_len.to_le_bytes());
        }
    }

    // --- Padding to EMBEDDING_ALIGNMENT ---
    let padding = (EMBEDDING_ALIGNMENT - (buf.len() % EMBEDDING_ALIGNMENT)) % EMBEDDING_ALIGNMENT;
    buf.resize(buf.len() + padding, 0);

    // --- Embeddings ---
    for entry in file_entries {
        for chunk in &entry.chunks {
            buf.extend_from_slice(&chunk.embedding);
        }
    }

    Ok(buf)
}

/// Encode the offset table as run-length i16 LE values, terminated by 0.
fn encode_offset_table(buf: &mut Vec<u8>, file_entries: &[FileEntry]) {
    if file_entries.is_empty() {
        buf.extend_from_slice(&0i16.to_le_bytes());
        return;
    }

    let mut current_pos: usize = 0;
    let mut include_run: usize = 0;

    for entry in file_entries {
        let pos = entry.git_index_position;

        if pos > current_pos {
            if include_run > 0 {
                write_positive_runs(buf, include_run);
                include_run = 0;
            }
            write_negative_runs(buf, pos - current_pos);
        }

        include_run += 1;
        current_pos = pos + 1;
    }

    if include_run > 0 {
        write_positive_runs(buf, include_run);
    }

    buf.extend_from_slice(&0i16.to_le_bytes());
}

fn write_positive_runs(buf: &mut Vec<u8>, mut count: usize) {
    while count > 0 {
        let n = count.min(i16::MAX as usize) as i16;
        buf.extend_from_slice(&n.to_le_bytes());
        count -= n as usize;
    }
}

fn write_negative_runs(buf: &mut Vec<u8>, mut count: usize) {
    while count > 0 {
        let n = count.min(i16::MAX as usize) as i16;
        buf.extend_from_slice(&(-n).to_le_bytes());
        count -= n as usize;
    }
}

/// Decode an offset table from a byte slice starting at offset 0.
///
/// Returns `(git_index_positions, bytes_consumed)`.
pub fn decode_offset_table(data: &[u8]) -> Result<(Vec<usize>, usize)> {
    let mut positions = Vec::new();
    let mut git_pos: usize = 0;
    let mut offset: usize = 0;

    loop {
        ensure!(offset + 2 <= data.len(), "offset table truncated");
        let run = i16::from_le_bytes([data[offset], data[offset + 1]]);
        offset += 2;

        if run == 0 {
            break;
        } else if run > 0 {
            for _ in 0..run {
                positions.push(git_pos);
                git_pos += 1;
            }
        } else {
            git_pos += (-run) as usize;
        }
    }

    Ok((positions, offset))
}

// ── Incremental indexing support ─────────────────────────────────

/// Parsed layout of an index file, for incremental embedding writes.
pub struct IndexLayout {
    /// Byte offset where embedding data starts in the file.
    pub embeddings_offset: usize,
    /// Prefix sum of chunk counts: `chunk_prefix[i]` is the total
    /// number of chunks in files 0..i. Length = num_files + 1.
    pub chunk_prefix: Vec<usize>,
    /// Number of files in this index.
    pub num_files: usize,
}

/// Write a skeleton index file with zeroed embeddings. This
/// pre-allocates the full file so embeddings can be written
/// incrementally at known offsets.
pub fn write_skeleton(
    path: &Path,
    header: &IndexHeader,
    file_infos: &[FileChunkInfo],
) -> Result<()> {
    let file_entries = file_infos_to_entries(file_infos);
    write_index(path, header, &file_entries)
}

/// Convert lightweight FileChunkInfo to FileEntry with zeroed
/// embeddings, suitable for writing a skeleton index.
fn file_infos_to_entries(file_infos: &[FileChunkInfo]) -> Vec<FileEntry> {
    file_infos
        .iter()
        .map(|info| {
            let mut offset = 0u32;
            FileEntry {
                git_index_position: info.git_index_position,
                chunks: info
                    .chunk_lengths
                    .iter()
                    .map(|&len| {
                        let entry = super::ChunkEntry {
                            byte_offset: offset,
                            chunk_len: len,
                            embedding: [0u8; EMBEDDING_BYTES],
                        };
                        offset += len as u32;
                        entry
                    })
                    .collect(),
            }
        })
        .collect()
}

/// Parse an index file's layout for incremental writes.
/// This reads only the metadata (not embeddings) to determine
/// where each file's embeddings live in the file.
pub fn index_layout(data: &[u8]) -> Result<IndexLayout> {
    ensure!(data.len() >= HEADER_SIZE, "index file too small");

    let (positions, ot_bytes) = decode_offset_table(&data[HEADER_SIZE..])?;
    let num_files = positions.len();
    let mut pos = HEADER_SIZE + ot_bytes;

    // Read chunk counts and build prefix sum.
    ensure!(
        pos + num_files * 2 <= data.len(),
        "index truncated at chunk count table"
    );
    let mut chunk_prefix = Vec::with_capacity(num_files + 1);
    chunk_prefix.push(0usize);
    for i in 0..num_files {
        let off = pos + i * 2;
        let count = u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        chunk_prefix.push(chunk_prefix.last().unwrap() + count);
    }
    pos += num_files * 2;

    let total_chunks = *chunk_prefix.last().unwrap();

    // Skip length table.
    pos += total_chunks * 2;

    // Compute embeddings offset (aligned).
    let embeddings_offset =
        (pos + EMBEDDING_ALIGNMENT - 1) & !(EMBEDDING_ALIGNMENT - 1);

    Ok(IndexLayout {
        embeddings_offset,
        chunk_prefix,
        num_files,
    })
}

/// Detect files with all-zero embeddings (incomplete / not yet
/// embedded). Returns indices into the file_infos array.
pub fn detect_incomplete(data: &[u8], layout: &IndexLayout) -> Vec<usize> {
    let mut incomplete = Vec::new();
    for i in 0..layout.num_files {
        let count = layout.chunk_prefix[i + 1] - layout.chunk_prefix[i];
        if count == 0 {
            continue; // no chunks, nothing to embed
        }
        let off = layout.embeddings_offset + layout.chunk_prefix[i] * EMBEDDING_BYTES;
        if off + EMBEDDING_BYTES <= data.len()
            && data[off..off + EMBEDDING_BYTES].iter().all(|&b| b == 0)
        {
            incomplete.push(i);
        }
    }
    incomplete
}

/// Write embeddings for a specific file into a pre-allocated index
/// file. The file must already have the correct size (from
/// `write_skeleton`).
pub fn write_embeddings_at(
    path: &Path,
    layout: &IndexLayout,
    file_idx: usize,
    embeddings: &[BinaryEmbedding],
) -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let offset = layout.embeddings_offset + layout.chunk_prefix[file_idx] * EMBEDDING_BYTES;
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(offset as u64))?;
    for emb in embeddings {
        file.write_all(emb)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BinaryEmbedding, GitHash};
    use crate::index::ChunkEntry;

    fn make_header() -> IndexHeader {
        let mut commit_hash: GitHash = [0; MAX_HASH_LEN];
        commit_hash[..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        IndexHeader {
            version: 1,
            commit_hash,
            parent_hash: [0; MAX_HASH_LEN],
        }
    }

    fn make_embedding(val: u8) -> BinaryEmbedding {
        [val; EMBEDDING_BYTES]
    }

    #[test]
    fn offset_table_empty() {
        let mut buf = Vec::new();
        encode_offset_table(&mut buf, &[]);
        let (positions, consumed) = decode_offset_table(&buf).unwrap();
        assert!(positions.is_empty());
        assert_eq!(consumed, 2);
    }

    #[test]
    fn offset_table_contiguous() {
        let entries: Vec<FileEntry> = (0..3)
            .map(|i| FileEntry {
                git_index_position: i,
                chunks: vec![],
            })
            .collect();
        let mut buf = Vec::new();
        encode_offset_table(&mut buf, &entries);
        let (positions, _) = decode_offset_table(&buf).unwrap();
        assert_eq!(positions, vec![0, 1, 2]);
    }

    #[test]
    fn offset_table_with_gaps() {
        let positions_in = vec![0, 1, 5, 6, 10];
        let entries: Vec<FileEntry> = positions_in
            .iter()
            .map(|&i| FileEntry {
                git_index_position: i,
                chunks: vec![],
            })
            .collect();
        let mut buf = Vec::new();
        encode_offset_table(&mut buf, &entries);
        let (positions_out, _) = decode_offset_table(&buf).unwrap();
        assert_eq!(positions_out, positions_in);
    }

    #[test]
    fn offset_table_leading_gap() {
        let entries = vec![
            FileEntry { git_index_position: 5, chunks: vec![] },
            FileEntry { git_index_position: 6, chunks: vec![] },
        ];
        let mut buf = Vec::new();
        encode_offset_table(&mut buf, &entries);
        let (positions, _) = decode_offset_table(&buf).unwrap();
        assert_eq!(positions, vec![5, 6]);
    }

    #[test]
    fn offset_table_large_gap() {
        let entries = vec![
            FileEntry { git_index_position: 0, chunks: vec![] },
            FileEntry { git_index_position: 40000, chunks: vec![] },
        ];
        let mut buf = Vec::new();
        encode_offset_table(&mut buf, &entries);
        let (positions, _) = decode_offset_table(&buf).unwrap();
        assert_eq!(positions, vec![0, 40000]);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let header = make_header();
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![
                    ChunkEntry {
                        byte_offset: 0,
                        chunk_len: 100,
                        embedding: make_embedding(0xAA),
                    },
                    ChunkEntry {
                        byte_offset: 100,
                        chunk_len: 50,
                        embedding: make_embedding(0xBB),
                    },
                ],
            },
            FileEntry {
                git_index_position: 3,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 200,
                    embedding: make_embedding(0xCC),
                }],
            },
        ];

        let data = encode_index(&header, &entries).unwrap();

        // Verify header
        assert_eq!(data[0], 1);
        assert_eq!(&data[1..5], &[0xDE, 0xAD, 0xBE, 0xEF]);

        // Verify offset table decodes correctly
        let (positions, consumed) = decode_offset_table(&data[HEADER_SIZE..]).unwrap();
        assert_eq!(positions, vec![0, 3]);

        // Verify chunk counts
        let cc_start = HEADER_SIZE + consumed;
        let cc0 = u16::from_le_bytes([data[cc_start], data[cc_start + 1]]);
        let cc1 = u16::from_le_bytes([data[cc_start + 2], data[cc_start + 3]]);
        assert_eq!(cc0, 2);
        assert_eq!(cc1, 1);

        // Verify lengths
        let lt_start = cc_start + 4;
        let l0 = u16::from_le_bytes([data[lt_start], data[lt_start + 1]]);
        let l1 = u16::from_le_bytes([data[lt_start + 2], data[lt_start + 3]]);
        let l2 = u16::from_le_bytes([data[lt_start + 4], data[lt_start + 5]]);
        assert_eq!(l0, 100);
        assert_eq!(l1, 50);
        assert_eq!(l2, 200);

        // Verify embeddings are aligned
        let emb_start = (lt_start + 6 + EMBEDDING_ALIGNMENT - 1) & !(EMBEDDING_ALIGNMENT - 1);
        assert_eq!(data[emb_start], 0xAA);
        assert_eq!(data[emb_start + EMBEDDING_BYTES], 0xBB);
        assert_eq!(data[emb_start + EMBEDDING_BYTES * 2], 0xCC);

        // Total size
        assert_eq!(data.len(), emb_start + 3 * EMBEDDING_BYTES);
    }

    #[test]
    fn unsorted_entries_rejected() {
        let header = make_header();
        let entries = vec![
            FileEntry { git_index_position: 5, chunks: vec![] },
            FileEntry { git_index_position: 2, chunks: vec![] },
        ];
        assert!(encode_index(&header, &entries).is_err());
    }

    #[test]
    fn duplicate_positions_rejected() {
        let header = make_header();
        let entries = vec![
            FileEntry { git_index_position: 3, chunks: vec![] },
            FileEntry { git_index_position: 3, chunks: vec![] },
        ];
        assert!(encode_index(&header, &entries).is_err());
    }

    #[test]
    fn skeleton_and_incremental_write() {
        let header = make_header();
        let file_infos = vec![
            FileChunkInfo {
                git_index_position: 0,
                path: "a.rs".to_string(),
                chunk_lengths: vec![100, 50],
            },
            FileChunkInfo {
                git_index_position: 3,
                path: "b.rs".to_string(),
                chunk_lengths: vec![200],
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");

        // Write skeleton with zeroed embeddings.
        write_skeleton(&path, &header, &file_infos).unwrap();

        // Parse layout.
        let data = std::fs::read(&path).unwrap();
        let layout = index_layout(&data).unwrap();
        assert_eq!(layout.num_files, 2);
        assert_eq!(layout.chunk_prefix, vec![0, 2, 3]);

        // All files should be incomplete.
        let incomplete = detect_incomplete(&data, &layout);
        assert_eq!(incomplete, vec![0, 1]);
        drop(data);

        // Write embeddings for file 0.
        let embs_0 = vec![make_embedding(0xAA), make_embedding(0xBB)];
        write_embeddings_at(&path, &layout, 0, &embs_0).unwrap();

        // Now only file 1 should be incomplete.
        let data = std::fs::read(&path).unwrap();
        let incomplete = detect_incomplete(&data, &layout);
        assert_eq!(incomplete, vec![1]);
        drop(data);

        // Write embeddings for file 1.
        let embs_1 = vec![make_embedding(0xCC)];
        write_embeddings_at(&path, &layout, 1, &embs_1).unwrap();

        // All complete.
        let data = std::fs::read(&path).unwrap();
        let incomplete = detect_incomplete(&data, &layout);
        assert!(incomplete.is_empty());

        // Verify embeddings match the full-write approach.
        let off = layout.embeddings_offset;
        assert_eq!(data[off], 0xAA);
        assert_eq!(data[off + EMBEDDING_BYTES], 0xBB);
        assert_eq!(data[off + EMBEDDING_BYTES * 2], 0xCC);
    }

    #[test]
    fn index_layout_matches_mmap_reader() {
        let header = make_header();
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![
                    ChunkEntry {
                        byte_offset: 0,
                        chunk_len: 100,
                        embedding: make_embedding(0xAA),
                    },
                ],
            },
            FileEntry {
                git_index_position: 5,
                chunks: vec![
                    ChunkEntry {
                        byte_offset: 0,
                        chunk_len: 50,
                        embedding: make_embedding(0xBB),
                    },
                    ChunkEntry {
                        byte_offset: 50,
                        chunk_len: 75,
                        embedding: make_embedding(0xCC),
                    },
                ],
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        write_index(&path, &header, &entries).unwrap();

        let data = std::fs::read(&path).unwrap();
        let layout = index_layout(&data).unwrap();

        assert_eq!(layout.num_files, 2);
        assert_eq!(layout.chunk_prefix, vec![0, 1, 3]);

        // Verify we can read embeddings at the computed offsets.
        let off = layout.embeddings_offset;
        assert_eq!(data[off], 0xAA);
        assert_eq!(data[off + EMBEDDING_BYTES], 0xBB);
        assert_eq!(data[off + EMBEDDING_BYTES * 2], 0xCC);
    }
}

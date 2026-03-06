use std::path::Path;

use anyhow::{Result, ensure, Context};
use memmap2::Mmap;

use crate::core::types::*;
use super::IndexReader;
use super::format::{HEADER_SIZE, decode_offset_table};

/// An index reader backed by a memory-mapped file. Provides
/// zero-copy access to embeddings and efficient chunk reference
/// resolution via precomputed prefix sums.
pub struct MmapIndexReader {
    mmap: Mmap,
    header: IndexHeader,
    /// Git index positions of included files.
    git_positions: Vec<usize>,
    /// Number of chunks per included file.
    chunk_counts: Vec<u16>,
    /// Prefix sum of chunk counts: `chunk_prefix[i]` is the total
    /// number of chunks in files 0..i. Length = num_files + 1.
    chunk_prefix: Vec<usize>,
    /// Byte length of each chunk (flat array, total_chunks entries).
    chunk_lengths: Vec<u16>,
    /// Byte offset where embedding data starts in the mmap.
    embeddings_offset: usize,
    /// Total number of embeddings (= total chunks across all files).
    total_chunks: usize,
}

impl MmapIndexReader {
    /// Open an index file at `path` and parse its metadata.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open index: {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        ensure!(mmap.len() >= HEADER_SIZE, "index file too small for header");

        // --- Header ---
        let version = mmap[0];
        ensure!(version == 1, "unsupported index version: {}", version);

        let mut commit_hash = [0u8; MAX_HASH_LEN];
        commit_hash.copy_from_slice(&mmap[1..1 + MAX_HASH_LEN]);

        let mut parent_hash = [0u8; MAX_HASH_LEN];
        parent_hash.copy_from_slice(&mmap[1 + MAX_HASH_LEN..HEADER_SIZE]);

        let header = IndexHeader {
            version,
            commit_hash,
            parent_hash,
        };

        // --- Offset table ---
        let (git_positions, ot_bytes) = decode_offset_table(&mmap[HEADER_SIZE..])?;
        let mut pos = HEADER_SIZE + ot_bytes;
        let num_files = git_positions.len();

        // --- Chunk count table ---
        ensure!(
            pos + num_files * 2 <= mmap.len(),
            "index truncated at chunk count table"
        );
        let mut chunk_counts = Vec::with_capacity(num_files);
        for i in 0..num_files {
            let off = pos + i * 2;
            chunk_counts.push(u16::from_le_bytes([mmap[off], mmap[off + 1]]));
        }
        pos += num_files * 2;

        // --- Prefix sum ---
        let mut chunk_prefix = Vec::with_capacity(num_files + 1);
        chunk_prefix.push(0usize);
        for &c in &chunk_counts {
            chunk_prefix.push(chunk_prefix.last().unwrap() + c as usize);
        }
        let total_chunks = *chunk_prefix.last().unwrap();

        // --- Length table ---
        ensure!(
            pos + total_chunks * 2 <= mmap.len(),
            "index truncated at length table"
        );
        let mut chunk_lengths = Vec::with_capacity(total_chunks);
        for i in 0..total_chunks {
            let off = pos + i * 2;
            chunk_lengths.push(u16::from_le_bytes([mmap[off], mmap[off + 1]]));
        }
        pos += total_chunks * 2;

        // --- Embedding alignment ---
        let embeddings_offset =
            (pos + EMBEDDING_ALIGNMENT - 1) & !(EMBEDDING_ALIGNMENT - 1);
        ensure!(
            embeddings_offset + total_chunks * EMBEDDING_BYTES <= mmap.len(),
            "index truncated at embedding data"
        );

        Ok(MmapIndexReader {
            mmap,
            header,
            git_positions,
            chunk_counts,
            chunk_prefix,
            chunk_lengths,
            embeddings_offset,
            total_chunks,
        })
    }

    /// Git index positions of every included file.
    pub fn git_positions(&self) -> &[usize] {
        &self.git_positions
    }

    /// Chunk counts per included file.
    pub fn chunk_counts(&self) -> &[u16] {
        &self.chunk_counts
    }

    /// Byte lengths of all chunks (flat, in order).
    pub fn chunk_lengths(&self) -> &[u16] {
        &self.chunk_lengths
    }

    /// Number of included files.
    pub fn file_count(&self) -> usize {
        self.git_positions.len()
    }

    /// Get the range of flat embedding indices for a specific included
    /// file. Use with `embedding()` to read individual embeddings.
    pub fn file_embedding_range(&self, file_idx: usize) -> std::ops::Range<usize> {
        self.chunk_prefix[file_idx]..self.chunk_prefix[file_idx + 1]
    }

    /// Get the chunk lengths for a specific included file.
    pub fn file_chunk_lengths(&self, file_idx: usize) -> &[u16] {
        let start = self.chunk_prefix[file_idx];
        let end = self.chunk_prefix[file_idx + 1];
        &self.chunk_lengths[start..end]
    }

    /// Find the internal file index for a given git index position.
    /// Returns `None` if the position is not in this index.
    pub fn find_file_by_git_position(&self, git_pos: usize) -> Option<usize> {
        self.git_positions.binary_search(&git_pos).ok()
    }
}

impl IndexReader for MmapIndexReader {
    fn header(&self) -> &IndexHeader {
        &self.header
    }

    fn embedding_count(&self) -> usize {
        self.total_chunks
    }

    fn embedding(&self, index: usize) -> &BinaryEmbedding {
        assert!(index < self.total_chunks, "embedding index out of bounds");
        let off = self.embeddings_offset + index * EMBEDDING_BYTES;
        // BinaryEmbedding = [u8; 256], alignment = 1, so this is safe.
        self.mmap[off..off + EMBEDDING_BYTES]
            .try_into()
            .expect("slice is exactly EMBEDDING_BYTES")
    }

    fn resolve_chunk_ref(&self, embedding_index: usize) -> Result<ChunkRef> {
        ensure!(
            embedding_index < self.total_chunks,
            "embedding index {} out of bounds (total: {})",
            embedding_index,
            self.total_chunks
        );

        // Binary search on prefix sums to find the file index.
        // chunk_prefix[file_idx] <= embedding_index < chunk_prefix[file_idx + 1]
        let file_idx = self.chunk_prefix.partition_point(|&p| p <= embedding_index) - 1;

        let chunk_index = embedding_index - self.chunk_prefix[file_idx];

        // Compute byte offset by summing lengths of preceding chunks
        // within this file.
        let first_chunk = self.chunk_prefix[file_idx];
        let mut byte_offset: u32 = 0;
        for i in first_chunk..(first_chunk + chunk_index) {
            byte_offset += self.chunk_lengths[i] as u32;
        }

        Ok(ChunkRef {
            file_index: self.git_positions[file_idx] as u32,
            chunk_index: chunk_index as u16,
            byte_offset,
            chunk_len: self.chunk_lengths[embedding_index],
        })
    }

    fn parent_hash(&self) -> &GitHash {
        &self.header.parent_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::format::write_index;
    use crate::index::{FileEntry, ChunkEntry};
    use std::io::Write;

    fn make_embedding(val: u8) -> BinaryEmbedding {
        [val; EMBEDDING_BYTES]
    }

    fn make_header(commit: u8, parent: u8) -> IndexHeader {
        let mut commit_hash = [0u8; MAX_HASH_LEN];
        commit_hash[0] = commit;
        let mut parent_hash = [0u8; MAX_HASH_LEN];
        parent_hash[0] = parent;
        IndexHeader {
            version: 1,
            commit_hash,
            parent_hash,
        }
    }

    /// Write an index to a temp file and read it back.
    fn write_and_read(
        header: &IndexHeader,
        entries: &[FileEntry],
    ) -> MmapIndexReader {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        write_index(&path, header, entries).unwrap();
        MmapIndexReader::open(&path).unwrap()
    }

    #[test]
    fn empty_index() {
        let header = make_header(0xAA, 0);
        let reader = write_and_read(&header, &[]);
        assert_eq!(reader.embedding_count(), 0);
        assert_eq!(reader.file_count(), 0);
        assert_eq!(reader.header().version, 1);
        assert_eq!(reader.header().commit_hash[0], 0xAA);
        assert_eq!(reader.parent_hash()[0], 0);
    }

    #[test]
    fn single_file_single_chunk() {
        let header = make_header(0x01, 0);
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 150,
                embedding: make_embedding(0x42),
            }],
        }];
        let reader = write_and_read(&header, &entries);

        assert_eq!(reader.embedding_count(), 1);
        assert_eq!(reader.file_count(), 1);
        assert_eq!(reader.git_positions(), &[0]);
        assert_eq!(reader.chunk_counts(), &[1]);
        assert_eq!(reader.chunk_lengths(), &[150]);

        // Embedding data
        assert_eq!(reader.embedding(0), &make_embedding(0x42));

        // Chunk ref resolution
        let cr = reader.resolve_chunk_ref(0).unwrap();
        assert_eq!(cr.file_index, 0);
        assert_eq!(cr.chunk_index, 0);
        assert_eq!(cr.byte_offset, 0);
        assert_eq!(cr.chunk_len, 150);
    }

    #[test]
    fn multiple_files_multiple_chunks() {
        let header = make_header(0x02, 0);
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![
                    ChunkEntry {
                        byte_offset: 0,
                        chunk_len: 100,
                        embedding: make_embedding(0xA0),
                    },
                    ChunkEntry {
                        byte_offset: 100,
                        chunk_len: 200,
                        embedding: make_embedding(0xA1),
                    },
                ],
            },
            FileEntry {
                git_index_position: 5,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 300,
                    embedding: make_embedding(0xB0),
                }],
            },
            FileEntry {
                git_index_position: 7,
                chunks: vec![
                    ChunkEntry {
                        byte_offset: 0,
                        chunk_len: 50,
                        embedding: make_embedding(0xC0),
                    },
                    ChunkEntry {
                        byte_offset: 50,
                        chunk_len: 75,
                        embedding: make_embedding(0xC1),
                    },
                    ChunkEntry {
                        byte_offset: 125,
                        chunk_len: 25,
                        embedding: make_embedding(0xC2),
                    },
                ],
            },
        ];
        let reader = write_and_read(&header, &entries);

        assert_eq!(reader.embedding_count(), 6);
        assert_eq!(reader.file_count(), 3);
        assert_eq!(reader.git_positions(), &[0, 5, 7]);
        assert_eq!(reader.chunk_counts(), &[2, 1, 3]);

        // Verify all embeddings
        assert_eq!(reader.embedding(0), &make_embedding(0xA0));
        assert_eq!(reader.embedding(1), &make_embedding(0xA1));
        assert_eq!(reader.embedding(2), &make_embedding(0xB0));
        assert_eq!(reader.embedding(3), &make_embedding(0xC0));
        assert_eq!(reader.embedding(4), &make_embedding(0xC1));
        assert_eq!(reader.embedding(5), &make_embedding(0xC2));

        // Resolve chunk refs
        let cr0 = reader.resolve_chunk_ref(0).unwrap();
        assert_eq!(cr0.file_index, 0);
        assert_eq!(cr0.chunk_index, 0);
        assert_eq!(cr0.byte_offset, 0);
        assert_eq!(cr0.chunk_len, 100);

        let cr1 = reader.resolve_chunk_ref(1).unwrap();
        assert_eq!(cr1.file_index, 0);
        assert_eq!(cr1.chunk_index, 1);
        assert_eq!(cr1.byte_offset, 100);
        assert_eq!(cr1.chunk_len, 200);

        let cr2 = reader.resolve_chunk_ref(2).unwrap();
        assert_eq!(cr2.file_index, 5);
        assert_eq!(cr2.chunk_index, 0);
        assert_eq!(cr2.byte_offset, 0);
        assert_eq!(cr2.chunk_len, 300);

        let cr3 = reader.resolve_chunk_ref(3).unwrap();
        assert_eq!(cr3.file_index, 7);
        assert_eq!(cr3.chunk_index, 0);
        assert_eq!(cr3.byte_offset, 0);
        assert_eq!(cr3.chunk_len, 50);

        let cr4 = reader.resolve_chunk_ref(4).unwrap();
        assert_eq!(cr4.file_index, 7);
        assert_eq!(cr4.chunk_index, 1);
        assert_eq!(cr4.byte_offset, 50);
        assert_eq!(cr4.chunk_len, 75);

        let cr5 = reader.resolve_chunk_ref(5).unwrap();
        assert_eq!(cr5.file_index, 7);
        assert_eq!(cr5.chunk_index, 2);
        assert_eq!(cr5.byte_offset, 125);
        assert_eq!(cr5.chunk_len, 25);
    }

    #[test]
    fn parent_hash_preserved() {
        let header = make_header(0x10, 0xFF);
        let reader = write_and_read(&header, &[]);
        assert_eq!(reader.parent_hash()[0], 0xFF);
    }

    #[test]
    fn out_of_bounds_embedding_panics() {
        let header = make_header(0x01, 0);
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 10,
                embedding: make_embedding(0x01),
            }],
        }];
        let reader = write_and_read(&header, &entries);
        let result = std::panic::catch_unwind(|| reader.embedding(1));
        assert!(result.is_err());
    }

    #[test]
    fn out_of_bounds_resolve_errors() {
        let header = make_header(0x01, 0);
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 10,
                embedding: make_embedding(0x01),
            }],
        }];
        let reader = write_and_read(&header, &entries);
        assert!(reader.resolve_chunk_ref(1).is_err());
    }

    #[test]
    fn truncated_file_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.kbi");
        // Write only 10 bytes — too small for a header.
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        drop(f);
        assert!(MmapIndexReader::open(&path).is_err());
    }

    #[test]
    fn file_embedding_range_and_chunk_lengths() {
        let header = make_header(0x01, 0);
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![
                    ChunkEntry { byte_offset: 0, chunk_len: 100, embedding: make_embedding(0xA0) },
                    ChunkEntry { byte_offset: 100, chunk_len: 200, embedding: make_embedding(0xA1) },
                ],
            },
            FileEntry {
                git_index_position: 3,
                chunks: vec![
                    ChunkEntry { byte_offset: 0, chunk_len: 50, embedding: make_embedding(0xB0) },
                ],
            },
        ];
        let reader = write_and_read(&header, &entries);

        // file_embedding_range
        assert_eq!(reader.file_embedding_range(0), 0..2);
        assert_eq!(reader.file_embedding_range(1), 2..3);

        // file_chunk_lengths
        assert_eq!(reader.file_chunk_lengths(0), &[100, 200]);
        assert_eq!(reader.file_chunk_lengths(1), &[50]);

        // find_file_by_git_position
        assert_eq!(reader.find_file_by_git_position(0), Some(0));
        assert_eq!(reader.find_file_by_git_position(3), Some(1));
        assert_eq!(reader.find_file_by_git_position(1), None);
        assert_eq!(reader.find_file_by_git_position(99), None);
    }

    #[test]
    fn file_with_zero_chunks() {
        // A file entry with zero chunks is valid.
        let header = make_header(0x01, 0);
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![],
            },
            FileEntry {
                git_index_position: 2,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 99,
                    embedding: make_embedding(0xDD),
                }],
            },
        ];
        let reader = write_and_read(&header, &entries);
        assert_eq!(reader.file_count(), 2);
        assert_eq!(reader.embedding_count(), 1);
        assert_eq!(reader.chunk_counts(), &[0, 1]);

        let cr = reader.resolve_chunk_ref(0).unwrap();
        assert_eq!(cr.file_index, 2); // git index position, not file_idx
        assert_eq!(cr.chunk_index, 0);
        assert_eq!(cr.chunk_len, 99);
    }
}

use std::collections::BinaryHeap;

use anyhow::Result;

use crate::core::types::{BinaryEmbedding, EMBEDDING_BYTES, VectorSearchResult};
use crate::index::IndexReader;

/// The outcome of a single-index vector search: the top-N results
/// plus whether any embedding slot scanned was all-zero, which marks
/// the index as incompletely embedded (a skeleton whose embeddings
/// were not all filled in — e.g. an interrupted indexing run).
///
/// `saw_incomplete` reflects the index file as a whole and is
/// independent of `top_n` (the loop scans every embedding) and of any
/// later shadowing, so it is stable across queries against the same
/// index file.
#[derive(Debug, Default)]
pub struct VectorSearchOutcome {
    /// Top-N results, closest first.
    pub results: Vec<VectorSearchResult>,
    /// True if at least one all-zero (unfilled) embedding was seen.
    pub saw_incomplete: bool,
}

/// Compute the Hamming distance between two binary embeddings.
///
/// This is a simple byte-by-byte XOR + POPCOUNT loop that
/// autovectorizes well on modern x86 (POPCNT + AVX2) and ARM
/// (NEON + CNT).
#[inline]
pub fn hamming_distance(a: &BinaryEmbedding, b: &BinaryEmbedding) -> u32 {
    let mut dist = 0u32;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Entry in the max-heap used for top-N selection. We use a
/// max-heap keyed on distance so we can cheaply evict the
/// worst (most distant) candidate when the heap is full.
#[derive(Eq, PartialEq)]
struct HeapEntry {
    distance: u32,
    index: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by distance, tie-break by lower index first.
        self.distance
            .cmp(&other.distance)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Search the index for the top N nearest embeddings by Hamming
/// distance. Returns the results (sorted by distance, closest first)
/// together with a flag indicating whether any unfilled (all-zero)
/// embedding slot was encountered — see [`VectorSearchOutcome`].
///
/// Unfilled slots are detected cheaply: an all-zero stored embedding
/// has `hamming_distance(query, zero) == popcount(query)`, a constant
/// for the whole search. We compare each computed distance against
/// that constant and only do the byte-level confirmation on the
/// (astronomically rare) match, so the common all-filled case adds no
/// measurable cost. Detected zero slots are excluded from the result
/// heap so a placeholder never surfaces as a (bogus) nearest result.
pub fn vector_search(
    query_embedding: &BinaryEmbedding,
    index: &dyn IndexReader,
    top_n: usize,
) -> Result<VectorSearchOutcome> {
    let count = index.embedding_count();
    if count == 0 || top_n == 0 {
        return Ok(VectorSearchOutcome::default());
    }

    // popcount(query) == hamming_distance(query, all-zero). A stored
    // all-zero embedding therefore yields exactly this distance.
    let zero: BinaryEmbedding = [0u8; EMBEDDING_BYTES];
    let query_popcount = hamming_distance(query_embedding, &zero);
    let mut saw_incomplete = false;

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_n + 1);

    for i in 0..count {
        let emb = index.embedding(i);
        let dist = hamming_distance(query_embedding, emb);

        // Cheap gate, then confirm: skip unfilled placeholder slots so
        // they neither pollute results nor go unnoticed.
        if dist == query_popcount && emb.iter().all(|&b| b == 0) {
            saw_incomplete = true;
            continue;
        }

        // Only insert if better than the worst in a full heap.
        if heap.len() < top_n {
            heap.push(HeapEntry {
                distance: dist,
                index: i,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.distance {
                heap.pop();
                heap.push(HeapEntry {
                    distance: dist,
                    index: i,
                });
            }
        }
    }

    // Extract results, resolve chunk refs, sort by distance.
    let mut results: Vec<VectorSearchResult> = heap
        .into_iter()
        .map(|entry| {
            let chunk_ref = index.resolve_chunk_ref(entry.index)?;
            Ok(VectorSearchResult {
                chunk_ref,
                hamming_distance: entry.distance,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    results.sort_by_key(|r| r.hamming_distance);
    Ok(VectorSearchOutcome {
        results,
        saw_incomplete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use crate::index::format::write_index;
    use crate::index::mmap::MmapIndexReader;
    use crate::index::{ChunkEntry, FileEntry};

    fn make_embedding(val: u8) -> BinaryEmbedding {
        [val; EMBEDDING_BYTES]
    }

    fn make_header() -> IndexHeader {
        IndexHeader {
            version: CURRENT_INDEX_VERSION,
            commit_hash: [0; MAX_HASH_LEN],
            parent_hash: [0; MAX_HASH_LEN],
        }
    }

    #[test]
    fn hamming_distance_identical() {
        let a = make_embedding(0xAA);
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_distance_opposite() {
        let a = make_embedding(0x00);
        let b = make_embedding(0xFF);
        // 256 bytes × 8 bits = 2048
        assert_eq!(hamming_distance(&a, &b), 2048);
    }

    #[test]
    fn hamming_distance_one_bit() {
        let a = make_embedding(0x00);
        let mut b = make_embedding(0x00);
        b[0] = 0x01; // 1 bit different
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn hamming_distance_symmetric() {
        let a = make_embedding(0x55);
        let b = make_embedding(0xAA);
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn hamming_distance_known_value() {
        let a = make_embedding(0x55); // 01010101 per byte
        let b = make_embedding(0xAA); // 10101010 per byte
        // Each byte differs in all 8 bits
        assert_eq!(hamming_distance(&a, &b), 256 * 8);
    }

    #[test]
    fn vector_search_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.kbi");
        let header = make_header();
        write_index(&path, &header, &[]).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        let query = make_embedding(0xFF);
        let outcome = vector_search(&query, &reader, 10).unwrap();
        assert!(outcome.results.is_empty());
        assert!(!outcome.saw_incomplete);
    }

    #[test]
    fn vector_search_returns_closest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        let header = make_header();

        // Create entries with distinct embeddings.
        let entries = vec![
            FileEntry {
                git_index_position: 0,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 100,
                    embedding: make_embedding(0xFF), // distance 0 from query 0xFF
                }],
            },
            FileEntry {
                git_index_position: 1,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 200,
                    embedding: make_embedding(0x00), // distance 2048 from 0xFF
                }],
            },
            FileEntry {
                git_index_position: 2,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 150,
                    embedding: make_embedding(0xFE), // distance 256 from 0xFF (1 bit per byte)
                }],
            },
        ];
        write_index(&path, &header, &entries).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        let query = make_embedding(0xFF);
        let VectorSearchOutcome {
            results,
            saw_incomplete,
        } = vector_search(&query, &reader, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be the closest (distance 0).
        assert_eq!(results[0].hamming_distance, 0);
        assert_eq!(results[0].chunk_ref.file_index, 0);
        // Second should be 0xFE (distance 256).
        assert_eq!(results[1].hamming_distance, 256);
        assert_eq!(results[1].chunk_ref.file_index, 2);
        // The 0x00 embedding looks like an unfilled slot to this query
        // (all-zero), so it is treated as incomplete and skipped rather
        // than returned as a (bogus) distance-2048 result.
        assert!(saw_incomplete);
    }

    #[test]
    fn vector_search_top_n_limits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        let header = make_header();

        // Embeddings 0x10..0x1a — all non-zero so none is mistaken for
        // an unfilled (all-zero) placeholder slot.
        let entries: Vec<FileEntry> = (0..10)
            .map(|i| FileEntry {
                git_index_position: i,
                chunks: vec![ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 100,
                    embedding: make_embedding(0x10 + i as u8),
                }],
            })
            .collect();
        write_index(&path, &header, &entries).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        let query = make_embedding(0x10);
        let results = vector_search(&query, &reader, 3).unwrap().results;
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn vector_search_top_n_larger_than_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        let header = make_header();

        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 100,
                embedding: make_embedding(0x42),
            }],
        }];
        write_index(&path, &header, &entries).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        let query = make_embedding(0x42);
        let results = vector_search(&query, &reader, 100).unwrap().results;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].hamming_distance, 0);
    }

    #[test]
    fn vector_search_flags_unfilled_slot() {
        // One real embedding plus one all-zero (skeleton placeholder)
        // slot. The search should report `saw_incomplete`, skip the
        // placeholder, and return only the real result.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        let header = make_header();

        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![
                ChunkEntry {
                    byte_offset: 0,
                    chunk_len: 100,
                    embedding: make_embedding(0x42),
                },
                ChunkEntry {
                    byte_offset: 100,
                    chunk_len: 100,
                    embedding: [0u8; EMBEDDING_BYTES], // unfilled placeholder
                },
            ],
        }];
        write_index(&path, &header, &entries).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        // Non-zero query so only the genuinely all-zero slot trips the
        // detector.
        let query = make_embedding(0x42);
        let VectorSearchOutcome {
            results,
            saw_incomplete,
        } = vector_search(&query, &reader, 10).unwrap();

        assert!(saw_incomplete);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_ref.chunk_index, 0);
        assert_eq!(results[0].hamming_distance, 0);
    }

    #[test]
    fn vector_search_complete_index_not_flagged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.kbi");
        let header = make_header();

        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 100,
                embedding: make_embedding(0x42),
            }],
        }];
        write_index(&path, &header, &entries).unwrap();
        let reader = MmapIndexReader::open(&path).unwrap();

        let query = make_embedding(0x01);
        let outcome = vector_search(&query, &reader, 10).unwrap();
        assert!(!outcome.saw_incomplete);
        assert_eq!(outcome.results.len(), 1);
    }
}

use crate::core::types::{BinaryEmbedding, VectorSearchResult};
use crate::index::IndexReader;
use anyhow::Result;

/// Compute the Hamming distance between two binary embeddings.
pub fn hamming_distance(a: &BinaryEmbedding, b: &BinaryEmbedding) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Search the index for the top N nearest embeddings by Hamming distance.
pub fn vector_search(
    _query_embedding: &BinaryEmbedding,
    _index: &dyn IndexReader,
    _top_n: usize,
) -> Result<Vec<VectorSearchResult>> {
    todo!()
}

pub mod format;
pub mod mmap;
pub mod compact;

use anyhow::Result;
use crate::core::types::{BinaryEmbedding, ChunkRef, GitHash, IndexHeader};

/// Trait for reading index files (object-safe for dynamic dispatch).
pub trait IndexReader {
    /// Read the index header.
    fn header(&self) -> &IndexHeader;

    /// Return the total number of embeddings in this index.
    fn embedding_count(&self) -> usize;

    /// Get the embedding at the given position (zero-copy for mmap).
    fn embedding(&self, index: usize) -> &BinaryEmbedding;

    /// Resolve an embedding index to a chunk reference.
    fn resolve_chunk_ref(&self, embedding_index: usize) -> Result<ChunkRef>;

    /// Get the parent commit hash (all-zeroes if root index).
    fn parent_hash(&self) -> &GitHash;
}

/// A file entry for index writing, containing chunk data.
pub struct FileEntry {
    /// Position of this file in the git index.
    pub git_index_position: usize,
    /// The chunks and their embeddings.
    pub chunks: Vec<ChunkEntry>,
}

/// A single chunk entry with its length and embedding.
pub struct ChunkEntry {
    /// Byte offset of this chunk within the file.
    pub byte_offset: u32,
    /// Length of this chunk in bytes.
    pub chunk_len: u16,
    /// Binary quantized embedding.
    pub embedding: BinaryEmbedding,
}

// Re-exports for convenience.
pub use format::write_index;
pub use mmap::MmapIndexReader;

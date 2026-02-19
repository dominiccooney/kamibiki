use anyhow::Result;
use crate::core::types::{BinaryEmbedding, ChunkRef, GitHash, IndexHeader};

/// Trait for writing index files.
pub trait IndexWriter {
    /// Write an index file for the given commit.
    fn write(
        &self,
        header: &IndexHeader,
        file_entries: &[FileEntry],
    ) -> Result<()>;
}

/// Trait for reading index files.
pub trait IndexReader {
    /// Read the index header.
    fn header(&self) -> &IndexHeader;

    /// Return the total number of embeddings in this index.
    fn embedding_count(&self) -> usize;

    /// Get the embedding at the given position.
    fn embedding(&self, index: usize) -> &BinaryEmbedding;

    /// Resolve an embedding index to a chunk reference.
    fn resolve_chunk_ref(&self, embedding_index: usize) -> Result<ChunkRef>;

    /// Get the parent commit hash, if this is an overlay index.
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

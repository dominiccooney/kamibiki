use anyhow::Result;

/// A chunk produced by a chunker. Chunks are non-overlapping and
/// cover the file from byte 0 contiguously.
pub struct Chunk {
    /// Byte offset of the start of this chunk in the source file.
    pub byte_offset: usize,
    /// The chunk content as bytes (valid UTF-8 for text files).
    pub content: Vec<u8>,
}

/// Trait for chunking file content into embeddable pieces.
pub trait Chunker: Send + Sync {
    /// Chunk the given file content. `path` is used to select the
    /// language parser. Returns chunks in order.
    fn chunk(&self, path: &str, content: &[u8]) -> Result<Vec<Chunk>>;
}

pub mod lines;
pub mod treesitter;

use anyhow::Result;
use std::sync::Arc;

/// A chunk produced by a chunker. Chunks are non-overlapping byte
/// ranges that cover the file from byte 0 contiguously.
///
/// This is a zero-copy type — it references a range in the source
/// content rather than owning a copy.
pub struct Chunk {
    /// Byte offset of the start of this chunk in the source file.
    pub byte_offset: usize,
    /// Length of this chunk in bytes.
    pub len: usize,
}

impl Chunk {
    /// Get the content of this chunk from the source buffer.
    #[inline]
    pub fn content<'a>(&self, source: &'a [u8]) -> &'a [u8] {
        &source[self.byte_offset..self.byte_offset + self.len]
    }
}

/// Trait for chunking file content into embeddable pieces.
pub trait Chunker: Send + Sync {
    /// Chunk the given file content. `path` is used to select the
    /// language parser. `max_tokens` is the target maximum number of
    /// tokens per chunk (depends on the embedding model's
    /// capabilities). Returns chunks in order.
    fn chunk(&self, path: &str, content: &[u8], max_tokens: usize) -> Result<Vec<Chunk>>;
}

/// Counts tokens using the embedding model's actual tokenizer,
/// loaded from HuggingFace Hub and cached locally.
#[derive(Clone)]
pub struct TokenCounter {
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl TokenCounter {
    /// Load the tokenizer for the Voyage embedding model from
    /// HuggingFace Hub (cached locally after first download).
    pub fn for_voyage() -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_pretrained("voyageai/voyage-context-3", None)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        Ok(TokenCounter {
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Count the number of tokens in a byte slice. Returns 0 for
    /// empty or non-UTF-8 content.
    pub fn count(&self, content: &[u8]) -> usize {
        let text = match std::str::from_utf8(content) {
            Ok(t) if !t.is_empty() => t,
            _ => return 0,
        };
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.len())
            .unwrap_or(0)
    }
}

/// The tsv1 chunker: tree-sitter based for supported languages,
/// line-based fallback for everything else. This is the standard
/// chunker for IndexVersion::V1.
pub fn tsv1_chunker() -> Result<impl Chunker> {
    let tc = TokenCounter::for_voyage()?;
    Ok(treesitter::TreeSitterChunker::new(tc))
}

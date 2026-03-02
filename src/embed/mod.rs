pub mod voyage;

use anyhow::Result;
use crate::core::types::BinaryEmbedding;

pub use voyage::VoyageEmbedder;
pub use voyage::{MAX_INPUTS_PER_REQUEST, MAX_REQUEST_TOKENS};

/// Trait for embedding text into binary quantized vectors.
pub trait Embedder: Send + Sync {
    /// Embed a batch of document chunks. Each inner Vec<String> is the
    /// set of chunks for one document (file). Chunks are embedded
    /// individually via the standard text embedding endpoint.
    fn embed_documents(
        &self,
        documents: &[Vec<String>],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<BinaryEmbedding>>>> + Send;

    /// Embed a query string.
    fn embed_query(
        &self,
        query: &str,
    ) -> impl std::future::Future<Output = Result<BinaryEmbedding>> + Send;
}

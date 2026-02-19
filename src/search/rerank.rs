use anyhow::Result;

/// A single reranked item with its original index and relevance score.
pub struct RerankItem {
    pub index: usize,
    pub relevance_score: f32,
}

/// Trait for reranking search results.
pub trait Reranker: Send + Sync {
    /// Rerank the given documents against the query, returning the
    /// top_k results sorted by relevance.
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> impl std::future::Future<Output = Result<Vec<RerankItem>>> + Send;
}

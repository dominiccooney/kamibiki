use anyhow::Result;
use reqwest::Client;

/// A single reranked item with its original index and relevance score.
pub struct RerankItem {
    pub index: usize,
    pub relevance_score: f32,
}

/// Trait for reranking search results.
pub trait Reranker: Send + Sync {
    /// Rerank the given documents against the query, returning the
    /// top_k results sorted by relevance (highest first).
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> impl std::future::Future<Output = Result<Vec<RerankItem>>> + Send;
}

/// Voyage AI reranker using rerank-2.5-lite.
pub struct VoyageReranker {
    api_key: String,
    client: Client,
}

// --- Request / Response types ---

#[derive(serde::Serialize)]
struct VoyageRerankRequest<'a> {
    query: &'a str,
    documents: &'a [&'a str],
    model: &'a str,
    top_k: usize,
}

#[derive(serde::Deserialize)]
struct VoyageRerankResponse {
    data: Vec<VoyageRerankData>,
    usage: VoyageRerankUsage,
}

#[derive(serde::Deserialize)]
struct VoyageRerankData {
    index: usize,
    relevance_score: f32,
}

#[derive(serde::Deserialize)]
struct VoyageRerankUsage {
    total_tokens: usize,
}

impl VoyageReranker {
    pub fn new(api_key: String) -> Self {
        VoyageReranker {
            api_key,
            client: Client::new(),
        }
    }
}

impl Reranker for VoyageReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<RerankItem>> {
        if documents.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let body = VoyageRerankRequest {
            query,
            documents,
            model: "rerank-2.5-lite",
            top_k,
        };

        let response = self
            .client
            .post("https://api.voyageai.com/v1/rerank")
            .header("authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Voyage rerank failed: HTTP {}\n{}", status, text);
        }

        let resp: VoyageRerankResponse = response.json().await?;
        eprintln!(
            "info: reranking used {} tokens",
            resp.usage.total_tokens
        );

        // Response is already sorted by relevance_score descending.
        let items = resp
            .data
            .into_iter()
            .map(|d| RerankItem {
                index: d.index,
                relevance_score: d.relevance_score,
            })
            .collect();

        Ok(items)
    }
}

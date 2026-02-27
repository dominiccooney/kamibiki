use anyhow::{Result, ensure};
use reqwest::Client;

use crate::core::types::{BinaryEmbedding, EMBEDDING_BYTES};
use super::Embedder;

/// Voyage AI embedding client using voyage-context-3 with binary
/// quantization.
pub struct VoyageEmbedder {
    api_key: String,
    client: Client,
}

// --- Request types ---

#[derive(serde::Serialize)]
struct ContextualEmbeddingRequest<'a> {
    inputs: &'a [Vec<String>],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
}

#[derive(serde::Serialize)]
struct TextEmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
}

// --- Response types ---

#[derive(serde::Deserialize)]
struct ContextualEmbeddingResponse {
    data: Vec<ContextualDocumentData>,
    usage: Usage,
}

#[derive(serde::Deserialize)]
struct ContextualDocumentData {
    data: Vec<EmbeddingData>,
    index: usize,
}

#[derive(serde::Deserialize)]
struct TextEmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(serde::Deserialize)]
struct Usage {
    total_tokens: usize,
}

/// The output dimension we request from Voyage. 2048 floats →
/// binary quantize → 256 bytes.
const OUTPUT_DIMENSION: usize = 2048;

/// Maximum tokens per API request (Voyage's limit is 120K, we use
/// 110K for safety margin).
const MAX_REQUEST_TOKENS: usize = 110_000;

/// Maximum number of documents per contextual embedding request.
/// Voyage's limit.
const MAX_DOCUMENTS_PER_REQUEST: usize = 16;

impl VoyageEmbedder {
    pub fn new(api_key: String) -> Self {
        VoyageEmbedder {
            api_key,
            client: Client::new(),
        }
    }

    /// Make a contextual embedding API call for a single batch.
    async fn call_contextual(
        &self,
        input_type: &str,
        inputs: &[Vec<String>],
    ) -> Result<ContextualEmbeddingResponse> {
        let body = ContextualEmbeddingRequest {
            inputs,
            model: "voyage-context-3",
            input_type,
            output_dimension: OUTPUT_DIMENSION,
        };
        let response = self
            .client
            .post("https://api.voyageai.com/v1/contextualizedembeddings")
            .header("authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Voyage contextual embedding failed: HTTP {}\n{}",
                status,
                text
            );
        }

        let resp: ContextualEmbeddingResponse = response.json().await?;
        eprintln!(
            "info: contextual embedding used {} tokens",
            resp.usage.total_tokens
        );
        Ok(resp)
    }

    /// Make a text embedding API call.
    async fn call_text_embedding(
        &self,
        input_type: &str,
        input: &[String],
    ) -> Result<TextEmbeddingResponse> {
        let body = TextEmbeddingRequest {
            input,
            model: "voyage-context-3",
            input_type,
            output_dimension: OUTPUT_DIMENSION,
        };
        let response = self
            .client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Voyage text embedding failed: HTTP {}\n{}",
                status,
                text
            );
        }

        let resp: TextEmbeddingResponse = response.json().await?;
        eprintln!(
            "info: text embedding used {} tokens",
            resp.usage.total_tokens
        );
        Ok(resp)
    }
}

/// Binary quantize a float embedding: positive → 1, else → 0,
/// packed MSB-first into bytes.
pub fn binary_quantize(floats: &[f32]) -> Result<BinaryEmbedding> {
    ensure!(
        floats.len() == OUTPUT_DIMENSION,
        "expected {} floats, got {}",
        OUTPUT_DIMENSION,
        floats.len()
    );

    let mut result = [0u8; EMBEDDING_BYTES];
    for (i, &val) in floats.iter().enumerate() {
        if val > 0.0 {
            result[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    Ok(result)
}

impl Embedder for VoyageEmbedder {
    async fn embed_documents(
        &self,
        documents: &[Vec<String>],
    ) -> Result<Vec<Vec<BinaryEmbedding>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Batch documents respecting API limits. We batch by
        // document count (max 16 per request) and do approximate
        // token counting by character length / 4.
        let mut all_results: Vec<Vec<BinaryEmbedding>> = vec![Vec::new(); documents.len()];

        let mut batch_start = 0;
        while batch_start < documents.len() {
            let mut batch_end = batch_start;
            let mut approx_tokens = 0usize;

            while batch_end < documents.len()
                && batch_end - batch_start < MAX_DOCUMENTS_PER_REQUEST
            {
                let doc_chars: usize = documents[batch_end]
                    .iter()
                    .map(|s| s.len())
                    .sum();
                let doc_approx_tokens = doc_chars / 4;
                if approx_tokens + doc_approx_tokens > MAX_REQUEST_TOKENS
                    && batch_end > batch_start
                {
                    break;
                }
                approx_tokens += doc_approx_tokens;
                batch_end += 1;
            }

            // Ensure progress even if a single document is huge.
            if batch_end == batch_start {
                batch_end = batch_start + 1;
            }

            let batch = &documents[batch_start..batch_end];
            let resp = self.call_contextual("document", batch).await?;

            // Parse response: data is ordered by document index,
            // each containing chunk embeddings.
            for doc_data in resp.data {
                let doc_idx = batch_start + doc_data.index;
                let mut chunk_embeddings: Vec<(usize, BinaryEmbedding)> = doc_data
                    .data
                    .iter()
                    .map(|ed| Ok((ed.index, binary_quantize(&ed.embedding)?)))
                    .collect::<Result<Vec<_>>>()?;
                chunk_embeddings.sort_by_key(|(idx, _)| *idx);
                all_results[doc_idx] =
                    chunk_embeddings.into_iter().map(|(_, e)| e).collect();
            }

            batch_start = batch_end;
        }

        Ok(all_results)
    }

    async fn embed_query(&self, query: &str) -> Result<BinaryEmbedding> {
        let input = vec![query.to_string()];
        let resp = self.call_text_embedding("query", &input).await?;

        ensure!(!resp.data.is_empty(), "empty embedding response");
        binary_quantize(&resp.data[0].embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_quantize_basic() {
        let mut floats = vec![0.0f32; OUTPUT_DIMENSION];
        // Set first 8 values: positive, negative, positive, 0, pos, neg, pos, neg
        floats[0] = 1.0;
        floats[1] = -0.5;
        floats[2] = 0.1;
        floats[3] = 0.0;
        floats[4] = 0.01;
        floats[5] = -1.0;
        floats[6] = 0.5;
        floats[7] = -0.1;

        let result = binary_quantize(&floats).unwrap();
        // bits: 1 0 1 0 1 0 1 0 = 0xAA
        assert_eq!(result[0], 0b10101010);
        // Remaining bytes should be 0 (all non-positive).
        for &b in &result[1..] {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn binary_quantize_all_positive() {
        let floats = vec![1.0f32; OUTPUT_DIMENSION];
        let result = binary_quantize(&floats).unwrap();
        for &b in &result {
            assert_eq!(b, 0xFF);
        }
    }

    #[test]
    fn binary_quantize_all_negative() {
        let floats = vec![-1.0f32; OUTPUT_DIMENSION];
        let result = binary_quantize(&floats).unwrap();
        for &b in &result {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn binary_quantize_wrong_size_rejected() {
        let floats = vec![1.0f32; 100];
        assert!(binary_quantize(&floats).is_err());
    }

    #[test]
    fn binary_quantize_exact_dimension() {
        // 2048 floats → 256 bytes
        assert_eq!(OUTPUT_DIMENSION / 8, EMBEDDING_BYTES);
    }
}

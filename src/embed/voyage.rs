use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, ensure};
use reqwest::Client;
use tokio::time::{Duration, Instant};

use crate::core::types::{BinaryEmbedding, EMBEDDING_BYTES};
use super::Embedder;

// ── Token rate limiter ───────────────────────────────────────────

/// Maximum tokens per minute we'll send to Voyage.
/// The actual API limit is 3,000,000; we use 2,800,000 for safety.
const TOKEN_RATE_LIMIT: usize = 2_800_000;

/// The sliding window duration for rate limiting.
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

/// Tracks token consumption in a sliding window and sleeps when
/// the budget would be exceeded.
struct TokenRateLimiter {
    inner: Mutex<RateLimiterInner>,
}

struct RateLimiterInner {
    window: VecDeque<(Instant, usize)>,
    limit: usize,
}

impl RateLimiterInner {
    /// Remove entries older than the rate-limit window.
    fn prune(&mut self, now: Instant) {
        while let Some(&(ts, _)) = self.window.front() {
            if now.duration_since(ts) >= RATE_LIMIT_WINDOW {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }
}

impl TokenRateLimiter {
    fn new(tokens_per_minute: usize) -> Self {
        Self {
            inner: Mutex::new(RateLimiterInner {
                window: VecDeque::new(),
                limit: tokens_per_minute,
            }),
        }
    }

    /// Wait until `tokens` can be consumed without exceeding the
    /// per-minute limit, then record the consumption.
    async fn acquire(&self, tokens: usize) {
        loop {
            let sleep_duration = {
                let mut inner = self.inner.lock().unwrap();
                let now = Instant::now();
                inner.prune(now);
                let current: usize =
                    inner.window.iter().map(|(_, t)| *t).sum();

                if current + tokens <= inner.limit {
                    inner.window.push_back((now, tokens));
                    return;
                }

                // Figure out how long to sleep: walk entries oldest-first
                // until freeing enough budget.
                let mut to_free = (current + tokens).saturating_sub(inner.limit);
                let mut wait_until = now;
                for &(ts, count) in &inner.window {
                    wait_until = ts + RATE_LIMIT_WINDOW;
                    if count >= to_free {
                        break;
                    }
                    to_free -= count;
                }

                wait_until.duration_since(now)
            };

            tokio::time::sleep(sleep_duration).await;
        }
    }
}

// ── Voyage embedder ──────────────────────────────────────────────

/// Voyage AI embedding client using voyage-code-3 with binary
/// quantization.
pub struct VoyageEmbedder {
    api_key: String,
    client: Client,
    rate_limiter: TokenRateLimiter,
}

// --- Request / Response types ---

#[derive(serde::Serialize)]
struct TextEmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
}

#[derive(serde::Deserialize)]
struct TextEmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    usage: Usage,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct Usage {
    total_tokens: usize,
}

/// The output dimension we request from Voyage. 2048 floats →
/// binary quantize → 256 bytes.
const OUTPUT_DIMENSION: usize = 2048;

/// Maximum tokens per API request (Voyage's limit is 120K for
/// voyage-code-3, we use 110K for safety margin).
pub const MAX_REQUEST_TOKENS: usize = 110_000;

/// Maximum number of text inputs per embedding request. Voyage's
/// limit is 1000; we use a smaller batch to keep requests
/// manageable.
pub const MAX_INPUTS_PER_REQUEST: usize = 128;

impl VoyageEmbedder {
    pub fn new(api_key: String) -> Self {
        VoyageEmbedder {
            api_key,
            client: Client::new(),
            rate_limiter: TokenRateLimiter::new(TOKEN_RATE_LIMIT),
        }
    }

    /// Make a text embedding API call. Automatically waits for
    /// rate-limit budget before sending.
    async fn call_text_embedding(
        &self,
        input_type: &str,
        input: &[String],
    ) -> Result<TextEmbeddingResponse> {
        // Estimate tokens (chars / 4) and acquire rate-limit budget.
        let approx_tokens: usize = input.iter().map(|s| s.len() / 4).sum::<usize>().max(1);
        self.rate_limiter.acquire(approx_tokens).await;

        let body = TextEmbeddingRequest {
            input,
            model: "voyage-code-3",
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
        Ok(resp)
    }

    /// Embed a flat batch of text chunks in a single API call.
    /// The caller must ensure the batch fits within API limits
    /// (`MAX_INPUTS_PER_REQUEST` items, ~`MAX_REQUEST_TOKENS` tokens).
    pub async fn embed_batch(&self, chunks: &[String]) -> Result<Vec<BinaryEmbedding>> {
        ensure!(!chunks.is_empty(), "empty batch");
        let resp = self.call_text_embedding("document", chunks).await?;
        ensure!(
            resp.data.len() == chunks.len(),
            "expected {} embeddings, got {}",
            chunks.len(),
            resp.data.len()
        );
        let mut embeddings = vec![[0u8; EMBEDDING_BYTES]; chunks.len()];
        for ed in &resp.data {
            ensure!(
                ed.index < chunks.len(),
                "embedding index {} out of bounds",
                ed.index
            );
            embeddings[ed.index] = binary_quantize(&ed.embedding)?;
        }
        Ok(embeddings)
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

        // Flatten all chunks from all documents into a single list,
        // tracking which document each chunk belongs to.
        let mut all_chunks: Vec<String> = Vec::new();
        let mut chunk_doc_map: Vec<usize> = Vec::new(); // chunk index → document index
        let mut doc_chunk_counts: Vec<usize> = Vec::new();

        for (doc_idx, doc_chunks) in documents.iter().enumerate() {
            doc_chunk_counts.push(doc_chunks.len());
            for chunk in doc_chunks {
                chunk_doc_map.push(doc_idx);
                all_chunks.push(chunk.clone());
            }
        }

        // Allocate result storage.
        let mut all_embeddings: Vec<BinaryEmbedding> = vec![[0u8; EMBEDDING_BYTES]; all_chunks.len()];

        // Batch chunks respecting API limits (max inputs per request
        // and approximate token budget).
        let mut batch_start = 0;
        while batch_start < all_chunks.len() {
            let mut batch_end = batch_start;
            let mut approx_tokens = 0usize;

            while batch_end < all_chunks.len()
                && batch_end - batch_start < MAX_INPUTS_PER_REQUEST
            {
                let chunk_approx_tokens = all_chunks[batch_end].len() / 4;
                if approx_tokens + chunk_approx_tokens > MAX_REQUEST_TOKENS
                    && batch_end > batch_start
                {
                    break;
                }
                approx_tokens += chunk_approx_tokens;
                batch_end += 1;
            }

            // Ensure progress even if a single chunk is huge.
            if batch_end == batch_start {
                batch_end = batch_start + 1;
            }

            let batch = &all_chunks[batch_start..batch_end];
            let resp = self.call_text_embedding("document", batch).await?;

            // Place embeddings at the correct flat indices.
            for ed in &resp.data {
                let flat_idx = batch_start + ed.index;
                all_embeddings[flat_idx] = binary_quantize(&ed.embedding)?;
            }

            batch_start = batch_end;
        }

        // Reassemble flat embeddings into per-document vectors.
        let mut results: Vec<Vec<BinaryEmbedding>> = Vec::with_capacity(documents.len());
        let mut offset = 0;
        for &count in &doc_chunk_counts {
            results.push(all_embeddings[offset..offset + count].to_vec());
            offset += count;
        }

        Ok(results)
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

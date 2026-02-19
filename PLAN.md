# Kamibiki Implementation Plan

## Overview

Kamibiki is a local, high-performance contextual search engine for git
repositories. It indexes code using tree-sitter chunking and Voyage AI
embeddings (binary quantized), stores indexes as flat mmap-able files,
and serves queries via CLI and MCP server.

This plan is designed for parallel implementation using git worktrees.
Each workstream can be developed independently once the shared
types/interfaces in the `core` module are in place on `main`.

## Architecture

```
kb (binary)
├── main.rs          — CLI entry point (clap)
├── server.rs        — Server + MCP (future)
├── core/
│   ├── mod.rs       — re-exports
│   ├── config.rs    — ~/.kb.conf, repo registry
│   ├── types.rs     — shared types (IndexHeader, ChunkRef, etc.)
│   └── git.rs       — gitoxide helpers (repo discovery, index iteration, blob reading)
├── chunk/
│   ├── mod.rs       — Chunker trait + registry
│   ├── treesitter.rs — tsv1: tree-sitter recursive chunker
│   └── lines.rs     — fallback line-based chunker
├── index/
│   ├── mod.rs       — IndexWriter, IndexReader
│   ├── format.rs    — binary format serialization/deserialization
│   ├── mmap.rs      — mmap-based read-only index access
│   └── compact.rs   — compaction (merge overlay chain into single file)
├── embed/
│   ├── mod.rs       — Embedder trait
│   └── voyage.rs    — Voyage AI client (embed + binary quantize)
├── search/
│   ├── mod.rs       — top-level search orchestration
│   ├── vector.rs    — XOR+POPCOUNT binary vector search
│   └── rerank.rs    — Voyage reranker client
└── lib.rs           — library root
```

## Phase 0: Shared Foundation (on `main`, before branching)

This must be merged to `main` first. All workstreams branch from this
commit. It establishes the Cargo project, shared types, traits, and
module stubs.

### Cargo.toml dependencies

```toml
[package]
name = "kb"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "kb"
path = "src/main.rs"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
gix = "0.75"
memmap2 = "0.9"
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
tree-sitter = "0.25"
tree-sitter-rust = "0.24"
tree-sitter-python = "0.23"
tree-sitter-javascript = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-go = "0.23"
tree-sitter-c = "0.23"
tree-sitter-cpp = "0.23"
tree-sitter-java = "0.23"
tree-sitter-ruby = "0.23"
ordered-float = "5"
dirs = "6"
toml = "0.8"
```

Note: tree-sitter grammar crate versions should be verified against
crates.io at implementation time. The `tree-sitter` crate had a major
API change at 0.25; grammars may need the `tree-sitter-language` compat
crate. Resolve this during Phase 0.

### Shared types (`src/core/types.rs`)

```rust
use std::path::PathBuf;

/// Identifies a chunker + embedder combination.
/// Currently only one is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexVersion {
    /// tsv1 chunker + voyage-context-3@2048 binary quantized
    V1 = 1,
}

/// The maximum git hash size we support (SHA-256).
pub const MAX_HASH_LEN: usize = 64;

/// A git hash, padded to MAX_HASH_LEN with zeroes.
pub type GitHash = [u8; MAX_HASH_LEN];

/// On-disk index file header. All multi-byte integers are
/// little-endian.
#[derive(Debug, Clone)]
#[repr(C, packed)]
pub struct IndexHeader {
    /// Index version / format. Currently always 1.
    pub version: u8,
    /// The git commit hash this index was created at.
    pub commit_hash: GitHash,
    /// Parent index commit hash, or all-zeroes if this is a root index.
    pub parent_hash: GitHash,
}

/// Binary quantized embedding for voyage-context-3@2048.
/// 2048 bits = 256 bytes.
pub const EMBEDDING_BYTES: usize = 256;
pub const EMBEDDING_ALIGNMENT: usize = 32; // AVX2-friendly

pub type BinaryEmbedding = [u8; EMBEDDING_BYTES];

/// A reference to a chunk within an index, used during search.
#[derive(Debug, Clone)]
pub struct ChunkRef {
    /// Index of the file in the git index order.
    pub file_index: u32,
    /// Index of the chunk within the file.
    pub chunk_index: u16,
    /// Byte offset of the chunk start within the file.
    pub byte_offset: u32,
    /// Length of the chunk in bytes.
    pub chunk_len: u16,
}

/// A search result before reranking.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub chunk_ref: ChunkRef,
    pub hamming_distance: u32,
}

/// A search result after reranking.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The file path relative to the repository root.
    pub path: String,
    /// The chunk content.
    pub content: String,
    /// Byte offset within the file.
    pub byte_offset: u32,
    /// Byte length of the chunk.
    pub byte_len: u16,
    /// Relevance score from the reranker (higher = more relevant).
    pub relevance_score: f32,
}

/// Configuration for a single indexed repository.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepoConfig {
    pub name: String,
    pub path: PathBuf,
}

/// An alias mapping a name to multiple repository names.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AliasConfig {
    pub name: String,
    pub repos: Vec<String>,
}

/// Top-level configuration stored in ~/.kb.conf.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct KbConfig {
    pub voyage_api_key: Option<String>,
    #[serde(default)]
    pub repos: Vec<RepoConfig>,
    #[serde(default)]
    pub aliases: Vec<AliasConfig>,
}
```

### Chunker trait (`src/chunk/mod.rs`)

```rust
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
```

### Embedder trait (`src/embed/mod.rs`)

```rust
use anyhow::Result;
use crate::core::types::BinaryEmbedding;

/// Trait for embedding text into binary quantized vectors.
pub trait Embedder: Send + Sync {
    /// Embed a batch of document chunks. Each inner Vec<String> is the
    /// set of chunks for one document (file), enabling contextual
    /// embeddings.
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
```

### Reranker trait (`src/search/rerank.rs`)

```rust
use anyhow::Result;

pub struct RerankItem {
    pub index: usize,
    pub relevance_score: f32,
}

pub trait Reranker: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> impl std::future::Future<Output = Result<Vec<RerankItem>>> + Send;
}
```

### Git helpers stub (`src/core/git.rs`)

```rust
use anyhow::Result;
use gix::bstr::BStr;
use std::path::Path;

/// Open a git repository at the given path.
pub fn open_repo(path: &Path) -> Result<gix::Repository> {
    Ok(gix::discover(path)?)
}

/// Iterate over files in the git index at HEAD, yielding
/// (index_position, path) pairs.
pub fn index_entries(repo: &gix::Repository)
    -> Result<impl Iterator<Item = (usize, &BStr)>>
{
    todo!()
}

/// Read a blob from the repository at a given revision and path.
pub fn read_blob(repo: &gix::Repository, commit_hash: &[u8], path: &str)
    -> Result<Vec<u8>>
{
    todo!()
}
```

### CLI stub (`src/main.rs`)

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kb", about = "Kamibiki - contextual search for git repos")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Set up Voyage AI key
    Init,
    /// Add a repository to the index
    Add {
        name: String,
        path: String,
    },
    /// Update the index
    Index {
        names: Vec<String>,
        #[arg(long)]
        compact: bool,
    },
    /// Show indexing status
    Status {
        name: Option<String>,
    },
    /// Start the server
    Start,
    /// Stop the server
    Stop,
    /// Search a repository
    Search {
        name: String,
        query: String,
    },
    /// Create a repository alias
    Alias {
        name: String,
        repos: Vec<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    // dispatch will be filled in as workstreams merge
    todo!()
}
```

## Parallel Workstreams

After Phase 0 is merged, create these worktrees:

```sh
git worktree add ../kamibiki-chunk   chunk
git worktree add ../kamibiki-index   index
git worktree add ../kamibiki-embed   embed
git worktree add ../kamibiki-search  search
git worktree add ../kamibiki-cli     cli
```

### Workstream A: Chunking (`chunk` branch)

**Goal:** Implement the `tsv1` chunker.

**Files:** `src/chunk/mod.rs`, `src/chunk/treesitter.rs`, `src/chunk/lines.rs`

**Tasks:**
1. Implement `LinesChunker` — the fallback chunker that splits on
   newlines and merges adjacent lines until reaching a target chunk
   size (measured in bytes; target ~1500 bytes for voyage-context-3
   which has a 4K token context but we want smaller chunks).
2. Implement `TreeSitterChunker` — uses tree-sitter to parse files in
   supported languages, recursively walks the parse tree, and produces
   chunks by collecting sibling nodes until reaching the target size.
   Falls back to `LinesChunker` for unsupported languages.
3. Language detection from file extension. Supported languages (at
   minimum): Rust, Python, JavaScript, TypeScript, Go, C, C++, Java,
   Ruby. Add more as needed.
4. The chunker must produce non-overlapping chunks starting at byte 0
   that contiguously cover the entire file content.
5. Unit tests with sample code files.

**Key design decisions:**
- Target chunk size: ~1500 bytes (configurable via const).
- Tree-sitter splitting strategy: Start at root, if a node exceeds the
  target size, recurse into children. Collect leaf/small-enough nodes
  into chunks greedily.
- Chunks must break on valid UTF-8 character boundaries for text files.

### Workstream B: Index File Format (`index` branch)

**Goal:** Implement reading and writing of the `.kb` index file format.

**Files:** `src/index/mod.rs`, `src/index/format.rs`, `src/index/mmap.rs`, `src/index/compact.rs`

**Tasks:**
1. Implement `IndexWriter` — given a git commit hash, a parent hash
   (optional), a list of file entries with chunk counts, chunk lengths,
   and embeddings, produce a `.kb/<hash>.kbi` file in the format
   described in the README.
2. Implement `IndexReader` via mmap — open an index file, parse the
   header, and provide zero-copy access to the offset table, chunk
   count table, length table, and embedding data.
3. Implement the offset table encoding: positive/negative i16 run-length
   encoding of which git index entries are included/skipped.
4. Implement `compact()` — walk the parent chain of overlay indexes,
   merge them into a single index at the current revision, and delete
   the old files.
5. Implement `resolve_chunk_ref()` — given an embedding index (position
   in the flat embedding array), walk the offset table, chunk count
   table, and length table to determine the file index, chunk index,
   and byte offset/length.
6. Unit tests with synthetic index data.

**Key design decisions:**
- All integers are little-endian on disk.
- Embedding data is aligned to `EMBEDDING_ALIGNMENT` (32 bytes).
- Index files are named `<commit_hash_hex>.kbi` in `.kb/` directory.
- The offset table uses i16 for compactness. Repositories with >32767
  consecutive included or skipped files would need multiple table
  entries (this is fine and should be handled).

### Workstream C: Embedding (`embed` branch)

**Goal:** Implement the Voyage AI embedding client with binary
quantization.

**Files:** `src/embed/mod.rs`, `src/embed/voyage.rs`

**Tasks:**
1. Implement `VoyageEmbedder` that calls the Voyage AI contextual
   embeddings API (`voyage-context-3` model, 2048 output dimensions).
2. Implement binary quantization: threshold at 0 (positive → 1,
   negative/zero → 0), pack into `[u8; 256]`.
3. Implement query embedding (input_type = "query").
4. Implement document embedding with contextual chunks
   (input_type = "document").
5. Handle API rate limits, batching (respect Voyage's 120K token
   request limit), and error handling.
6. Integration tests (require `VOYAGE_API_KEY` env var, skip if absent).

**Key design decisions:**
- Binary quantization reduces 2048 f32 → 256 bytes (32x compression).
- We use the contextual embeddings API which takes document chunks with
  context. Each chunk is embedded in the context of its containing file.
- The API key is stored in `~/.kb.conf`.

### Workstream D: Search + Rerank (`search` branch)

**Goal:** Implement vector search (XOR+POPCOUNT) and reranking.

**Files:** `src/search/mod.rs`, `src/search/vector.rs`, `src/search/rerank.rs`

**Tasks:**
1. Implement `hamming_distance(a: &BinaryEmbedding, b: &BinaryEmbedding) -> u32`
   using XOR + POPCOUNT. Use explicit SIMD or rely on autovectorization
   with a simple byte-by-byte loop (benchmark both).
2. Implement `vector_search(query_embedding, index_reader, top_n) -> Vec<VectorSearchResult>`
   — scan all embeddings in the index, compute Hamming distance, maintain
   a min-heap (max-heap by distance) of top N results.
3. Implement `resolve_results()` — take the top N embedding offsets,
   sort by offset, scan the index tables in parallel to find file paths
   and byte ranges, then read the actual chunk content from git.
4. Implement `VoyageReranker` — submit query + chunk contents to
   `rerank-2.5-lite`, return reranked results.
5. Implement the parent-index overlay walk: when an index has a parent,
   search both, prefer newer results, suppress stale results.
6. Unit tests for Hamming distance, integration tests for the full
   search pipeline.

**Key design decisions:**
- N (number of vector search results before reranking) should be ~200,
  limited by the reranker's context window.
- Hamming distance on 256 bytes can be very fast with POPCNT
  instructions. A simple loop `(a[i] ^ b[i]).count_ones()` should
  autovectorize well.
- Results from newer overlay indexes shadow results from older indexes
  for the same file regions.

### Workstream E: CLI + Config (`cli` branch)

**Goal:** Wire up the CLI commands and configuration management.

**Files:** `src/main.rs`, `src/core/config.rs`, `src/core/git.rs`, `src/server.rs`

**Tasks:**
1. Implement `kb init` — prompt for Voyage API key, store in
   `~/.kb.conf`.
2. Implement `kb add <name> <path>` — validate path is a git repo,
   add to `~/.kb.conf`, create `.kb/` directory.
3. Implement `kb status [name]` — list repos, show latest indexed
   commit, index file count/size.
4. Implement `kb index <name>` — orchestrate: diff HEAD vs last
   indexed commit, chunk changed files, embed chunks, write overlay
   index. With `--compact`, compact afterward.
5. Implement `kb search <name> <query>` — load index, embed query,
   vector search, rerank, display results.
6. Implement `kb alias` — manage aliases in config.
7. Implement config loading/saving (`~/.kb.conf` as TOML).
8. The `kb start`/`kb stop`/MCP server is deferred to a later phase.

**Key design decisions:**
- Config file is TOML format at `~/.kb.conf`.
- The `.` shorthand for the current repository resolves by discovering
  the git repo from the current directory and finding it in the config.
- `kb index` without `--compact` produces an overlay index. With
  `--compact`, it produces a single flat index.

## Merge Order

Workstreams A-D have no dependencies on each other and can be
developed fully in parallel. Workstream E depends on all others for
final integration, but config management and CLI parsing can proceed
in parallel.

Suggested merge order:
1. A (chunking) — no dependencies
2. B (index format) — no dependencies
3. C (embedding) — no dependencies
4. D (search) — depends on B's `IndexReader` interface, but can stub it
5. E (CLI) — final integration, merges last

When merging, resolve conflicts in `lib.rs` / `main.rs` module
declarations. Each workstream should only touch its own module files
to minimize conflicts.

## Phase 0 Deliverables Checklist

Before creating worktrees, the following must be on `main`:

- [ ] `Cargo.toml` with all dependencies
- [ ] `rust-toolchain.toml` (nightly for portable_simd if needed, or
      stable if we avoid nightly features)
- [ ] `src/lib.rs` with module declarations
- [ ] `src/main.rs` with CLI stub
- [ ] `src/core/mod.rs`, `types.rs`, `config.rs`, `git.rs` (stubs)
- [ ] `src/chunk/mod.rs` with `Chunker` trait
- [ ] `src/embed/mod.rs` with `Embedder` trait
- [ ] `src/index/mod.rs` with `IndexReader`/`IndexWriter` trait stubs
- [ ] `src/search/mod.rs`, `vector.rs`, `rerank.rs` with trait stubs
- [ ] `.gitignore` (target/, .kb/)
- [ ] Project compiles (`cargo check` passes)

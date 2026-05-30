use std::path::PathBuf;

/// Identifies a chunker + embedder combination.
/// Currently only one is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexVersion {
    /// tsv1 chunker + voyage-code-3@2048 binary quantized.
    ///
    /// Historical: v1 recorded `parent_hash` as whatever `.kbi` was
    /// most recently modified at index time, which is *not* a git
    /// ancestor in general. Chain walks following those links could
    /// wander into commits unrelated to the search's HEAD. Such files
    /// are no longer readable and are treated as if absent.
    V1 = 1,
    /// Same chunker/embedder as v1, but `parent_hash` is guaranteed to
    /// be the nearest *git ancestor* among indexed commits (or the
    /// all-zero root). This makes `parent_hash` links safe to follow.
    V2 = 2,
}

/// The index format version written by this build. Readers accept
/// only this version; older files are ignored (treated as absent) so
/// a format change never silently mixes incompatible semantics.
pub const CURRENT_INDEX_VERSION: u8 = IndexVersion::V2 as u8;

/// The maximum git hash size we support (SHA-256).
pub const MAX_HASH_LEN: usize = 64;

/// A git hash, padded to MAX_HASH_LEN with zeroes.
pub type GitHash = [u8; MAX_HASH_LEN];

/// On-disk index file header. All multi-byte integers are
/// little-endian.
#[derive(Debug, Clone)]
#[repr(C, packed)]
pub struct IndexHeader {
    /// Index version / format. See [`IndexVersion`];
    /// [`CURRENT_INDEX_VERSION`] is what this build writes.
    pub version: u8,
    /// The git commit hash this index was created at.
    pub commit_hash: GitHash,
    /// Parent index commit hash, or all-zeroes if this is a root index.
    pub parent_hash: GitHash,
}

/// Binary quantized embedding for voyage-code-3@2048.
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

use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};

use kb::chunk::{self, Chunker};
use kb::core::config;
use kb::core::git;
use kb::core::types::*;
use kb::embed::{Embedder, VoyageEmbedder};
use kb::snippet;
use kb::index::{MmapIndexReader, IndexReader};
use kb::ops::{self, IndexProgress};
use kb::search::{Reranker, VoyageReranker, chain_search, load_index_chain, IndexChain};

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
    /// Update the index for one or more repositories
    Index {
        names: Vec<String>,
        #[arg(long)]
        compact: bool,
        /// Git revision to index at (commit hash, branch name, tag, HEAD~1, etc.).
        /// Defaults to HEAD when not specified.
        #[arg(short, long)]
        commit: Option<String>,
    },
    /// Show indexing status
    Status {
        name: Option<String>,
    },
    /// Start the MCP server on stdio
    Start,
    /// Stop the server (not yet implemented)
    Stop,
    /// Search a repository
    Search {
        name: String,
        query: String,
        /// Number of results to display
        #[arg(short = 'n', long, default_value_t = 10)]
        top: usize,
        /// Git revision to search from (commit hash, branch name, tag, HEAD~1, etc.).
        /// Defaults to HEAD when not specified.
        #[arg(short, long)]
        commit: Option<String>,
    },
    /// Create a repository alias
    Alias {
        name: String,
        repos: Vec<String>,
    },
    /// Drop (delete) all index files for a repository
    Drop {
        name: String,
    },
    /// Debug / diagnostic subcommands
    Debug {
        #[command(subcommand)]
        command: DebugCommands,
    },
}

/// Which chunking strategy to use.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum ChunkerKind {
    /// Tree-sitter AST chunker with line-based fallback (default)
    #[default]
    Treesitter,
    /// Indent-heuristic line-based chunker (no tree-sitter)
    Lines,
}

#[derive(Subcommand)]
enum DebugCommands {
    /// Count files and bytes at HEAD that pass the binary filter
    Files {
        /// Repository name (or '.' for current directory)
        #[arg(default_value = ".")]
        name: String,
    },
    /// Run the chunker on files and display the resulting chunks
    Chunk {
        /// Files to chunk. If none are given, reads from stdin (requires --stdin-filename).
        files: Vec<String>,

        /// Maximum tokens per chunk (default: 430)
        #[arg(long, default_value_t = 430)]
        max_tokens: usize,

        /// Filename to use for language detection when reading from stdin
        #[arg(long)]
        stdin_filename: Option<String>,

        /// Chunking strategy to use
        #[arg(long, value_enum, default_value_t = ChunkerKind::Treesitter)]
        chunker: ChunkerKind,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Add { name, path } => cmd_add(&name, &path),
        Commands::Index { names, compact, commit } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_index(&names, compact, commit.as_deref()))
        }
        Commands::Status { name } => cmd_status(name.as_deref()),
        Commands::Search { name, query, top, commit } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_search(&name, &query, top, commit.as_deref()))
        }
        Commands::Alias { name, repos } => cmd_alias(&name, &repos),
        Commands::Drop { name } => cmd_drop(&name),
        Commands::Start => kb::server::run_mcp_server(),
        Commands::Stop => {
            eprintln!("Server mode not yet implemented.");
            Ok(())
        }
        Commands::Debug { command } => match command {
            DebugCommands::Files { name } => cmd_debug_files(&name),
            DebugCommands::Chunk {
                files,
                max_tokens,
                stdin_filename,
                chunker,
            } => cmd_debug_chunk(files, max_tokens, stdin_filename, chunker),
        },
    }
}

// ── kb init ──────────────────────────────────────────────────────

fn cmd_init() -> Result<()> {
    let mut cfg = config::load_config()?;

    eprint!("Voyage AI API key: ");
    io::stderr().flush()?;
    let mut key = String::new();
    io::stdin().read_line(&mut key)?;
    let key = key.trim().to_string();

    if key.is_empty() {
        anyhow::bail!("API key cannot be empty");
    }

    cfg.voyage_api_key = Some(key);
    config::save_config(&cfg)?;
    eprintln!("Saved to {}", config::config_path()?.display());
    Ok(())
}

// ── kb add ───────────────────────────────────────────────────────

fn cmd_add(name: &str, path: &str) -> Result<()> {
    let abs_path = std::fs::canonicalize(path)
        .with_context(|| format!("path does not exist: {}", path))?;

    git::open_repo(&abs_path)
        .with_context(|| format!("{} is not a git repository", abs_path.display()))?;

    let mut cfg = config::load_config()?;

    if cfg.repos.iter().any(|r| r.name == name) {
        anyhow::bail!("repository '{}' already exists in config", name);
    }

    let kb_dir = abs_path.join(".kb");
    std::fs::create_dir_all(&kb_dir)?;

    cfg.repos.push(RepoConfig {
        name: name.to_string(),
        path: abs_path.clone(),
    });
    config::save_config(&cfg)?;

    eprintln!("Added repository '{}' at {}", name, abs_path.display());
    Ok(())
}

// ── kb status ────────────────────────────────────────────────────

fn cmd_status(name: Option<&str>) -> Result<()> {
    let cfg = config::load_config()?;

    let repos: Vec<&RepoConfig> = match name {
        Some(n) => {
            let repo = resolve_repo(&cfg, n)?;
            vec![repo]
        }
        None => cfg.repos.iter().collect(),
    };

    if repos.is_empty() {
        eprintln!("No repositories configured. Use 'kb add' to add one.");
        return Ok(());
    }

    for repo in repos {
        println!("Repository: {} ({})", repo.name, repo.path.display());

        let kb_dir = repo.path.join(".kb");
        if !kb_dir.exists() {
            println!("  Status: not indexed");
            continue;
        }

        let mut kbi_files: Vec<_> = std::fs::read_dir(&kb_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "kbi")
            })
            .collect();

        if kbi_files.is_empty() {
            println!("  Status: not indexed");
        } else {
            kbi_files.sort_by_key(|e| {
                std::cmp::Reverse(
                    e.metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                )
            });

            let total_size: u64 = kbi_files
                .iter()
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum();

            println!("  Index files: {}", kbi_files.len());
            println!("  Total index size: {}", format_bytes(total_size));

            if let Ok(reader) = MmapIndexReader::open(&kbi_files[0].path()) {
                let commit_hex = commit_hash_to_hex(&reader.header().commit_hash);
                let short = &commit_hex[..commit_hex.len().min(12)];
                println!("  Latest indexed commit: {}", short);
                println!("  Files in latest index: {}", reader.file_count());
                println!("  Embeddings in latest index: {}", reader.embedding_count());
            }
        }
        println!();
    }

    Ok(())
}

// ── kb index ─────────────────────────────────────────────────────
//
// Pipelined indexing with restartable progress:
//
// 1. Determine layout: If a skeleton .kbi already exists for this
//    commit, reconstruct file_infos from its metadata (instant).
//    Otherwise, chunk all tracked files (CPU-parallel) and write
//    a skeleton with zeroed embedding slots.
//
// 2. Detect incomplete files by scanning for all-zero embedding
//    slots in the skeleton.
//
// 3. Embed (pipelined, restartable): A producer thread re-reads
//    and re-chunks only incomplete files, packing chunks across
//    file boundaries into API-sized batches. The async consumer
//    dispatches up to 8 concurrent Voyage API requests via a
//    JoinSet + semaphore, writing results to the pre-allocated
//    index file as each completes.

// ── kb index ─────────────────────────────────────────────────────
//
// Delegates to the shared ops::index_repo pipeline with a CLI-friendly
// progress reporter that uses indicatif progress bars.

/// CLI progress reporter using indicatif progress bars.
struct CliProgress {
    chunk_pb: std::sync::Mutex<Option<ProgressBar>>,
    embed_pb: std::sync::Mutex<Option<ProgressBar>>,
}

impl CliProgress {
    fn new() -> Self {
        CliProgress {
            chunk_pb: std::sync::Mutex::new(None),
            embed_pb: std::sync::Mutex::new(None),
        }
    }
}

impl IndexProgress for CliProgress {
    fn on_start(&self, repo_name: &str) {
        eprintln!("Indexing '{}'...", repo_name);
    }

    fn on_message(&self, msg: &str) {
        // Suspend any active progress bar before printing.
        if let Ok(guard) = self.embed_pb.lock() {
            if let Some(ref pb) = *guard {
                pb.suspend(|| eprintln!("  {}", msg));
                return;
            }
        }
        if let Ok(guard) = self.chunk_pb.lock() {
            if let Some(ref pb) = *guard {
                pb.suspend(|| eprintln!("  {}", msg));
                return;
            }
        }
        eprintln!("  {}", msg);
    }

    fn on_warning(&self, msg: &str) {
        if let Ok(guard) = self.embed_pb.lock() {
            if let Some(ref pb) = *guard {
                pb.suspend(|| eprintln!("  warn: {}", msg));
                return;
            }
        }
        eprintln!("  warn: {}", msg);
    }

    fn on_chunk_start(&self, total: usize) {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "  Chunking  {bar:40.cyan/dim} {pos}/{len} files  {elapsed_precise} [{eta_precise} remaining]"
            )
            .unwrap()
            .progress_chars("━╸─"),
        );
        *self.chunk_pb.lock().unwrap() = Some(pb);
    }

    fn on_chunk_progress(&self, completed: usize) {
        if let Ok(guard) = self.chunk_pb.lock() {
            if let Some(ref pb) = *guard {
                pb.set_position(completed as u64);
            }
        }
    }

    fn on_chunk_done(&self) {
        if let Ok(mut guard) = self.chunk_pb.lock() {
            if let Some(pb) = guard.take() {
                pb.finish_and_clear();
            }
        }
    }

    fn on_embed_start(&self, total: usize, already_done: usize) {
        let pb = ProgressBar::new(total as u64);
        pb.set_position(already_done as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "  Embedding {bar:40.green/dim} {pos}/{len} chunks  {elapsed_precise} [{eta_precise} remaining]"
            )
            .unwrap()
            .progress_chars("━╸─"),
        );
        *self.embed_pb.lock().unwrap() = Some(pb);
    }

    fn on_embed_progress(&self, completed: usize) {
        if let Ok(guard) = self.embed_pb.lock() {
            if let Some(ref pb) = *guard {
                pb.set_position(completed as u64);
            }
        }
    }

    fn on_embed_done(&self) {
        if let Ok(mut guard) = self.embed_pb.lock() {
            if let Some(pb) = guard.take() {
                pb.finish_and_clear();
            }
        }
    }
}

async fn cmd_index(names: &[String], _compact: bool, commit: Option<&str>) -> Result<()> {
    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    let repo_names = resolve_names(&cfg, names)?;
    let progress = CliProgress::new();

    for repo_name in &repo_names {
        let repo_cfg = resolve_repo(&cfg, repo_name)?;

        ops::index_repo(repo_cfg, &api_key, &progress, commit).await
            .with_context(|| format!("failed to index '{}'", repo_cfg.name))?;
    }

    Ok(())
}

// ── kb search ────────────────────────────────────────────────────

async fn cmd_search(name: &str, query: &str, top: usize, commit: Option<&str>) -> Result<()> {
    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    let repo_cfg = resolve_repo(&cfg, name)?;
    let repo = git::open_repo(&repo_cfg.path)?;

    // Resolve the commit ref (branch name, tag, hash, HEAD~1, etc.)
    // to a full hex commit hash. Defaults to HEAD when not specified.
    let resolved_commit = git::resolve_commit_hex(&repo, commit)?;
    let ref_label = commit.unwrap_or("HEAD");

    let kb_dir = repo_cfg.path.join(".kb");
    let IndexChain { readers: chain, commits_behind } =
        load_index_chain(&kb_dir, &repo, Some(&resolved_commit))?;

    let total_embeddings: usize = chain.iter().map(|r| r.embedding_count()).sum();
    eprintln!(
        "Searching {} at {} ({} index file{}, {} total embeddings)...",
        repo_cfg.name,
        ref_label,
        chain.len(),
        if chain.len() == 1 { "" } else { "s" },
        total_embeddings,
    );
    if commits_behind > 0 {
        eprintln!(
            "Note: index is {} commit{} behind {}. Run 'kb index' to update.",
            commits_behind,
            if commits_behind == 1 { "" } else { "s" },
            ref_label,
        );
    }

    let embedder = VoyageEmbedder::new(api_key.clone());
    let query_embedding = embedder.embed_query(query).await?;

    let vector_top_n = 200;
    let vector_results = chain_search(
        &chain,
        &repo,
        &query_embedding,
        vector_top_n,
    )?;

    if vector_results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    // Read chunk content from the commit each result came from.
    let mut chunk_contents: Vec<String> = Vec::new();
    let mut chunk_paths: Vec<String> = Vec::new();
    let mut chunk_offsets: Vec<(u32, u16)> = Vec::new();
    let mut chunk_start_lines: Vec<usize> = Vec::new();

    for result in &vector_results {
        let content = git::read_blob(&repo, &result.commit_hex, &result.path)
            .unwrap_or_default();

        let offset = result.byte_offset as usize;
        let len = result.chunk_len as usize;
        let start_line = snippet::start_line_for_offset(&content, offset);
        let chunk_text = if offset + len <= content.len() {
            String::from_utf8_lossy(&content[offset..offset + len]).into_owned()
        } else if offset < content.len() {
            String::from_utf8_lossy(&content[offset..]).into_owned()
        } else {
            String::new()
        };

        chunk_contents.push(chunk_text);
        chunk_paths.push(result.path.clone());
        chunk_offsets.push((result.byte_offset, result.chunk_len));
        chunk_start_lines.push(start_line);
    }

    let reranker = VoyageReranker::new(api_key);
    let doc_refs: Vec<&str> = chunk_contents.iter().map(|s| s.as_str()).collect();
    let reranked = reranker.rerank(query, &doc_refs, top).await?;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    for (rank, item) in reranked.iter().enumerate() {
        let path = &chunk_paths[item.index];
        let (byte_offset, byte_len) = chunk_offsets[item.index];
        let content = &chunk_contents[item.index];
        let start_line = chunk_start_lines[item.index];

        writeln!(
            out,
            "━━━ Result {} ━━━ {} (offset {}, len {}) [score: {:.4}]",
            rank + 1,
            path,
            byte_offset,
            byte_len,
            item.relevance_score
        )?;
        let staleness = git::check_snippet_staleness(
            &repo_cfg.path,
            path,
            byte_offset as usize,
            byte_len as usize,
            content.as_bytes(),
        );
        if let Some(note) = staleness.note() {
            writeln!(out, "{}", note)?;
        }
        let numbered = snippet::format_with_line_numbers(content, start_line);
        writeln!(out, "{}", numbered)?;
    }

    Ok(())
}

// ── kb alias ─────────────────────────────────────────────────────

fn cmd_alias(name: &str, repos: &[String]) -> Result<()> {
    let mut cfg = config::load_config()?;

    if repos.is_empty() {
        if let Some(alias) = cfg.aliases.iter().find(|a| a.name == name) {
            println!("{}: {}", alias.name, alias.repos.join(", "));
        } else {
            eprintln!("No alias '{}' found.", name);
        }
        return Ok(());
    }

    for repo_name in repos {
        if !cfg.repos.iter().any(|r| r.name == *repo_name) {
            anyhow::bail!("unknown repository: '{}'", repo_name);
        }
    }

    if let Some(alias) = cfg.aliases.iter_mut().find(|a| a.name == name) {
        alias.repos = repos.to_vec();
    } else {
        cfg.aliases.push(AliasConfig {
            name: name.to_string(),
            repos: repos.to_vec(),
        });
    }

    config::save_config(&cfg)?;
    eprintln!("Alias '{}' → {}", name, repos.join(", "));
    Ok(())
}

// ── kb drop ──────────────────────────────────────────────────────

fn cmd_drop(name: &str) -> Result<()> {
    let cfg = config::load_config()?;
    let repo_cfg = resolve_repo(&cfg, name)?;

    let kb_dir = repo_cfg.path.join(".kb");
    if !kb_dir.exists() {
        eprintln!("No index found for '{}'.", repo_cfg.name);
        return Ok(());
    }

    let kbi_files: Vec<_> = std::fs::read_dir(&kb_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "kbi")
        })
        .collect();

    if kbi_files.is_empty() {
        eprintln!("No index files found for '{}'.", repo_cfg.name);
        return Ok(());
    }

    let total_size: u64 = kbi_files
        .iter()
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum();

    let count = kbi_files.len();
    for entry in &kbi_files {
        std::fs::remove_file(entry.path())
            .with_context(|| format!("failed to remove {}", entry.path().display()))?;
    }

    eprintln!(
        "Dropped {} index file{} ({}) for '{}'.",
        count,
        if count == 1 { "" } else { "s" },
        format_bytes(total_size),
        repo_cfg.name,
    );

    Ok(())
}

// ── kb debug files ───────────────────────────────────────────────

fn cmd_debug_files(path: &str) -> Result<()> {
    let repo_path = std::fs::canonicalize(path)
        .with_context(|| format!("path does not exist: {}", path))?;
    let repo = git::open_repo(&repo_path)?;
    let commit_hex = git::head_commit_hex(&repo)?;
    let entries = git::index_entries(&repo)?;

    let mut file_count: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut skipped: u64 = 0;
    let mut errors: u64 = 0;

    for (_git_idx, path) in &entries {
        let content = match git::read_blob_at_head(&repo, path) {
            Ok(c) => c,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        // Simple filter: skip files with null bytes in first 8192 bytes.
        if content.iter().take(8192).any(|&b| b == 0) {
            skipped += 1;
            continue;
        }

        file_count += 1;
        total_bytes += content.len() as u64;
    }

    println!("HEAD: {}", &commit_hex[..commit_hex.len().min(12)]);
    println!("Tracked files: {}", entries.len());
    println!("Files passing filter: {}", file_count);
    println!("Total bytes: {} ({})", total_bytes, format_bytes(total_bytes));
    if skipped > 0 {
        println!("Skipped (binary): {}", skipped);
    }
    if errors > 0 {
        println!("Errors: {}", errors);
    }

    Ok(())
}

// ── kb debug chunk ───────────────────────────────────────────────

fn make_chunker(kind: ChunkerKind) -> Result<Box<dyn Chunker>> {
    match kind {
        ChunkerKind::Treesitter => {
            let c = chunk::tsv1_chunker()?;
            Ok(Box::new(c))
        }
        ChunkerKind::Lines => {
            let c = chunk::lines_chunker()?;
            Ok(Box::new(c))
        }
    }
}

fn cmd_debug_chunk(
    files: Vec<String>,
    max_tokens: usize,
    stdin_filename: Option<String>,
    chunker_kind: ChunkerKind,
) -> Result<()> {
    let chunker = make_chunker(chunker_kind)?;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if files.is_empty() {
        let filename = stdin_filename.as_deref().unwrap_or("stdin.txt");
        let mut content = Vec::new();
        io::stdin().read_to_end(&mut content)?;
        print_chunks(&mut out, filename, &content, &*chunker, max_tokens)?;
    } else {
        for (i, path) in files.iter().enumerate() {
            if i > 0 {
                writeln!(out)?;
            }
            let content = std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path, e))?;
            print_chunks(&mut out, path, &content, &*chunker, max_tokens)?;
        }
    }

    Ok(())
}

fn print_chunks(
    out: &mut impl Write,
    path: &str,
    content: &[u8],
    chunker: &(impl Chunker + ?Sized),
    max_tokens: usize,
) -> Result<()> {
    let chunks = chunker.chunk(path, content, max_tokens)?;

    writeln!(
        out,
        "=== {} ({} chunk{}) ===",
        path,
        chunks.len(),
        if chunks.len() == 1 { "" } else { "s" }
    )?;

    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 {
            writeln!(out, "---")?;
        }
        let text = String::from_utf8_lossy(chunk.content(content));
        let start_line = snippet::start_line_for_offset(content, chunk.byte_offset);
        let numbered = snippet::format_with_line_numbers(&text, start_line);
        write!(out, "{}", numbered)?;
    }

    if chunks
        .last()
        .is_some_and(|c| {
            let end = c.byte_offset + c.len;
            end > 0 && content[end - 1] != b'\n'
        })
    {
        writeln!(out)?;
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────

fn resolve_repo<'a>(cfg: &'a KbConfig, name: &str) -> Result<&'a RepoConfig> {
    if name == "." {
        let cwd = std::env::current_dir()?;
        let repo = git::open_repo(&cwd)?;
        let repo_root = repo
            .workdir()
            .ok_or_else(|| anyhow::anyhow!("bare repository not supported"))?;
        let canon = std::fs::canonicalize(repo_root)?;
        cfg.repos
            .iter()
            .find(|r| {
                std::fs::canonicalize(&r.path)
                    .map(|p| p == canon)
                    .unwrap_or(false)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "current directory's repository is not registered. Use 'kb add' first."
                )
            })
    } else {
        cfg.repos
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| anyhow::anyhow!("unknown repository: '{}'", name))
    }
}

fn resolve_names(cfg: &KbConfig, names: &[String]) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for name in names {
        if let Some(alias) = cfg.aliases.iter().find(|a| a.name == *name) {
            result.extend(alias.repos.clone());
        } else if cfg.repos.iter().any(|r| r.name == *name) || name == "." {
            result.push(name.clone());
        } else {
            anyhow::bail!("unknown repository or alias: '{}'", name);
        }
    }
    Ok(result)
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// ── Delta indexing helpers ────────────────────────────────────────

/// Extract the commit hash hex string from a GitHash (stored as ASCII
/// hex bytes padded with zeroes).
fn commit_hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
    String::from_utf8_lossy(&hash[..end]).into_owned()
}


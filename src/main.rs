use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use kb::chunk::{self, Chunker, TokenCounter};
use kb::core::config;
use kb::core::git;
use kb::core::types::*;
use kb::embed::{Embedder, VoyageEmbedder};
use kb::index::{self, FileChunkInfo, MmapIndexReader, IndexReader};
use kb::search::{self, Reranker, VoyageReranker};

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
        Commands::Index { names, compact } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_index(&names, compact))
        }
        Commands::Status { name } => cmd_status(name.as_deref()),
        Commands::Search { name, query, top } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_search(&name, &query, top))
        }
        Commands::Alias { name, repos } => cmd_alias(&name, &repos),
        Commands::Drop { name } => cmd_drop(&name),
        Commands::Start => kb::server::run_mcp_server(),
        Commands::Stop => {
            eprintln!("Server mode not yet implemented.");
            Ok(())
        }
        Commands::Debug { command } => match command {
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
                let commit_hex = hex::encode(
                    &reader.header().commit_hash,
                );
                let short = &commit_hex[..commit_hex.find(|c| c == '0').unwrap_or(commit_hex.len()).max(8).min(commit_hex.len())];
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

/// Number of concurrent Voyage API embedding requests.
const CONCURRENT_EMBED_REQUESTS: usize = 8;

/// Target max tokens per chunk for voyage-code-3.
const MAX_TOKENS: usize = 430;

/// A batch of chunks ready for a single API embedding call,
/// potentially spanning multiple files.
struct EmbedBatch {
    /// The text of each chunk.
    chunks: Vec<String>,
    /// For each chunk: (file_index_in_file_infos, chunk_index_within_file).
    locations: Vec<(usize, usize)>,
}

async fn cmd_index(names: &[String], _compact: bool) -> Result<()> {
    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    let repo_names = resolve_names(&cfg, names)?;

    for repo_name in &repo_names {
        let repo_cfg = resolve_repo(&cfg, repo_name)?;
        eprintln!("Indexing '{}'...", repo_cfg.name);

        index_repo(repo_cfg, &api_key).await
            .with_context(|| format!("failed to index '{}'", repo_cfg.name))?;
    }

    Ok(())
}

async fn index_repo(repo_cfg: &RepoConfig, api_key: &str) -> Result<()> {
    let repo = git::open_repo(&repo_cfg.path)?;
    let commit_hex = git::head_commit_hex(&repo)?;
    eprintln!("  HEAD commit: {}", &commit_hex[..commit_hex.len().min(12)]);

    let entries = git::index_entries(&repo)?;
    eprintln!("  Tracked files: {}", entries.len());

    let kb_dir = repo_cfg.path.join(".kb");
    std::fs::create_dir_all(&kb_dir)?;
    let index_path = kb_dir.join(format!(
        "{}.kbi",
        &commit_hex[..commit_hex.len().min(16)]
    ));

    // ── Determine file layout ────────────────────────────────
    //
    // If the skeleton .kbi already exists, reconstruct file_infos
    // from its metadata (fast — no chunking needed). Otherwise,
    // chunk all tracked files to determine the layout and write
    // the skeleton.

    let file_infos: Vec<FileChunkInfo> = if index_path.exists() {
        let data = std::fs::read(&index_path)?;
        let infos = index::read_file_infos(&data, &entries)?;
        let total = infos.iter().map(|f| f.chunk_count()).sum::<usize>();
        eprintln!(
            "  Loaded layout from existing index ({} files, {} chunks)",
            infos.len(),
            total,
        );
        infos
    } else {
        let infos = chunk_all_files(&repo_cfg.path, &entries)?;

        let mut commit_hash: GitHash = [0; MAX_HASH_LEN];
        let hex_bytes = commit_hex.as_bytes();
        commit_hash[..hex_bytes.len().min(MAX_HASH_LEN)]
            .copy_from_slice(&hex_bytes[..hex_bytes.len().min(MAX_HASH_LEN)]);

        let header = IndexHeader {
            version: 1,
            commit_hash,
            parent_hash: [0; MAX_HASH_LEN],
        };

        index::write_skeleton(&index_path, &header, &infos)?;
        eprintln!("  Pre-allocated index: {}", index_path.display());
        infos
    };

    let total_chunks: usize = file_infos.iter().map(|f| f.chunk_count()).sum();
    if total_chunks == 0 {
        eprintln!("  No chunks to index.");
        return Ok(());
    }

    // ── Detect incomplete files ──────────────────────────────

    let data = std::fs::read(&index_path)?;
    let layout = Arc::new(index::index_layout(&data)?);
    let needs_work = index::detect_incomplete(&data, &layout);
    drop(data);

    if needs_work.is_empty() {
        eprintln!(
            "  Index complete ({} files, {} embeddings).",
            file_infos.len(),
            total_chunks
        );
        return Ok(());
    }

    let incomplete_chunks: usize = needs_work
        .iter()
        .map(|&i| file_infos[i].chunk_count())
        .sum();
    let already_done_chunks = total_chunks - incomplete_chunks;

    // ── Pipeline: produce cross-file batches, embed concurrently ──
    //
    // Producer thread: re-reads and re-chunks only incomplete files,
    // packing chunks across file boundaries into API-sized batches
    // (up to 128 chunks / ~110K tokens each).
    //
    // Consumer: dispatches up to CONCURRENT_EMBED_REQUESTS API calls
    // in parallel via a JoinSet + semaphore, writing results to the
    // pre-allocated index file as each completes.

    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<EmbedBatch>(CONCURRENT_EMBED_REQUESTS * 2);

    let file_infos = Arc::new(file_infos);
    let file_infos_for_producer = Arc::clone(&file_infos);
    let repo_path = repo_cfg.path.to_path_buf();

    let producer = std::thread::spawn(move || -> Result<()> {
        let repo = git::open_repo(&repo_path)?;
        let chunker = chunk::tsv1_chunker()?;
        let tc = TokenCounter::for_voyage()?;

        let mut batch_chunks: Vec<String> = Vec::new();
        let mut batch_locations: Vec<(usize, usize)> = Vec::new();
        let mut batch_tokens: usize = 0;

        for &file_idx in &needs_work {
            let info = &file_infos_for_producer[file_idx];

            let content = match git::read_blob_at_head(&repo, &info.path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  warn: read failed for {}: {}", info.path, e);
                    continue;
                }
            };

            let chunks = match chunker.chunk(&info.path, &content, MAX_TOKENS) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  warn: re-chunk failed for {}: {}", info.path, e);
                    continue;
                }
            };

            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                let text =
                    String::from_utf8_lossy(chunk.content(&content)).into_owned();
                let chunk_tokens = tc.count(text.as_bytes());

                // Flush if adding this chunk would exceed API limits.
                if !batch_chunks.is_empty()
                    && (batch_chunks.len()
                        >= kb::embed::MAX_INPUTS_PER_REQUEST
                        || batch_tokens + chunk_tokens
                            > kb::embed::MAX_REQUEST_TOKENS)
                {
                    let batch = EmbedBatch {
                        chunks: std::mem::take(&mut batch_chunks),
                        locations: std::mem::take(&mut batch_locations),
                    };
                    batch_tokens = 0;
                    if tx.blocking_send(batch).is_err() {
                        return Ok(()); // consumer dropped
                    }
                }

                batch_chunks.push(text);
                batch_locations.push((file_idx, chunk_idx));
                batch_tokens += chunk_tokens;
            }
        }

        // Flush final partial batch.
        if !batch_chunks.is_empty() {
            let _ = tx.blocking_send(EmbedBatch {
                chunks: batch_chunks,
                locations: batch_locations,
            });
        }

        Ok(())
    });

    // Consumer: embed batches with bounded concurrency.
    let embedder = Arc::new(VoyageEmbedder::new(api_key.to_string()));
    let semaphore = Arc::new(tokio::sync::Semaphore::new(CONCURRENT_EMBED_REQUESTS));
    let mut join_set = tokio::task::JoinSet::new();

    let embed_pb = ProgressBar::new(total_chunks as u64);
    embed_pb.set_position(already_done_chunks as u64);
    embed_pb.set_style(
        ProgressStyle::with_template(
            "  Embedding {bar:40.green/dim} {pos}/{len} chunks  {elapsed_precise} [{eta_precise} remaining]"
        )
        .unwrap()
        .progress_chars("━╸─"),
    );

    let mut had_error = false;

    loop {
        tokio::select! {
            batch = rx.recv() => {
                match batch {
                    Some(batch) => {
                        let permit = Arc::clone(&semaphore)
                            .acquire_owned()
                            .await
                            .map_err(|e| anyhow::anyhow!("semaphore closed: {}", e))?;
                        let embedder = Arc::clone(&embedder);
                        let chunk_count = batch.chunks.len();
                        join_set.spawn(async move {
                            let result = embedder.embed_batch(&batch.chunks).await;
                            drop(permit);
                            result.map(|embs| (batch.locations, embs, chunk_count))
                        });
                    }
                    None => break, // Producer done, channel closed.
                }
            }
            Some(result) = join_set.join_next(), if !join_set.is_empty() => {
                match result {
                    Ok(Ok((locations, embeddings, count))) => {
                        index::write_embeddings_scattered(
                            &index_path, &layout, &locations, &embeddings,
                        )?;
                        embed_pb.inc(count as u64);
                    }
                    Ok(Err(e)) => {
                        embed_pb.suspend(|| {
                            eprintln!(
                                "  error: embedding failed: {}. Progress saved, re-run to resume.",
                                e
                            );
                        });
                        had_error = true;
                        break;
                    }
                    Err(e) => {
                        embed_pb.suspend(|| {
                            eprintln!(
                                "  error: task panicked: {}. Progress saved, re-run to resume.",
                                e
                            );
                        });
                        had_error = true;
                        break;
                    }
                }
            }
        }
    }

    // Drain remaining in-flight tasks.
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok((locations, embeddings, count))) => {
                index::write_embeddings_scattered(
                    &index_path, &layout, &locations, &embeddings,
                )?;
                embed_pb.inc(count as u64);
            }
            Ok(Err(e)) => {
                if !had_error {
                    embed_pb.suspend(|| {
                        eprintln!(
                            "  error: embedding failed: {}. Progress saved, re-run to resume.",
                            e
                        );
                    });
                    had_error = true;
                }
            }
            Err(e) => {
                if !had_error {
                    embed_pb.suspend(|| {
                        eprintln!(
                            "  error: task panicked: {}. Progress saved, re-run to resume.",
                            e
                        );
                    });
                    had_error = true;
                }
            }
        }
    }

    embed_pb.finish_and_clear();

    // Wait for producer thread.
    match producer.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("  warn: producer error: {}", e),
        Err(_) => eprintln!("  warn: producer thread panicked"),
    }

    if !had_error {
        eprintln!(
            "  Wrote index: {} ({} files, {} embeddings)",
            index_path.display(),
            file_infos.len(),
            total_chunks,
        );
    }

    Ok(())
}

/// Chunk all tracked files to determine index layout. Uses half the
/// available CPU cores. Returns FileChunkInfo for each non-empty,
/// non-binary file.
fn chunk_all_files(
    repo_path: &Path,
    entries: &[(usize, String)],
) -> Result<Vec<FileChunkInfo>> {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        / 2;
    let num_threads = num_threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .context("failed to create thread pool")?;

    let chunker = chunk::tsv1_chunker()?;
    let repo_path = repo_path.to_path_buf();

    let chunk_pb = ProgressBar::new(entries.len() as u64);
    chunk_pb.set_style(
        ProgressStyle::with_template(
            "  Chunking  {bar:40.cyan/dim} {pos}/{len} files  {elapsed_precise} [{eta_precise} remaining]"
        )
        .unwrap()
        .progress_chars("━╸─"),
    );

    let chunk_results: Vec<Option<FileChunkInfo>> = pool.install(|| {
        use std::cell::RefCell;
        thread_local! {
            static THREAD_REPO: RefCell<Option<gix::Repository>> = const { RefCell::new(None) };
        }

        entries
            .par_iter()
            .map(|(git_idx, path)| {
                chunk_pb.inc(1);

                let content = THREAD_REPO.with(|cell| {
                    let mut opt = cell.borrow_mut();
                    let repo = opt.get_or_insert_with(|| {
                        git::open_repo(&repo_path)
                            .expect("failed to open repo in worker thread")
                    });
                    git::read_blob_at_head(repo, path).ok()
                });

                let content = match content {
                    Some(c) => c,
                    None => return None,
                };

                if content.iter().take(8192).any(|&b| b == 0) {
                    return None;
                }

                match chunker.chunk(path, &content, MAX_TOKENS) {
                    Ok(chunks) if !chunks.is_empty() => Some(FileChunkInfo {
                        git_index_position: *git_idx,
                        path: path.clone(),
                        chunk_lengths: chunks.iter().map(|c| c.len as u16).collect(),
                    }),
                    Ok(_) => None,
                    Err(e) => {
                        chunk_pb.suspend(|| {
                            eprintln!("  warn: chunking failed for {}: {}", path, e);
                        });
                        None
                    }
                }
            })
            .collect()
    });

    chunk_pb.finish_and_clear();

    let file_infos: Vec<FileChunkInfo> = chunk_results.into_iter().flatten().collect();
    let total_chunks: usize = file_infos.iter().map(|f| f.chunk_count()).sum();
    let skipped = entries.len() - file_infos.len();

    eprintln!(
        "  Chunked: {} files → {} chunks (using {} threads)",
        file_infos.len(),
        total_chunks,
        num_threads,
    );
    if skipped > 0 {
        eprintln!("  Skipped: {}", skipped);
    }

    Ok(file_infos)
}

// ── kb search ────────────────────────────────────────────────────

async fn cmd_search(name: &str, query: &str, top: usize) -> Result<()> {
    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    let repo_cfg = resolve_repo(&cfg, name)?;
    let repo = git::open_repo(&repo_cfg.path)?;

    let kb_dir = repo_cfg.path.join(".kb");
    let index_path = find_latest_index(&kb_dir)?;
    let reader = MmapIndexReader::open(&index_path)?;

    eprintln!(
        "Searching {} ({} embeddings)...",
        repo_cfg.name,
        reader.embedding_count()
    );

    let embedder = VoyageEmbedder::new(api_key.clone());
    let query_embedding = embedder.embed_query(query).await?;

    let vector_top_n = 200.min(reader.embedding_count());
    let vector_results = search::vector_search(
        &query_embedding,
        &reader,
        vector_top_n,
    )?;

    if vector_results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    let entries = git::index_entries(&repo)?;

    let mut chunk_contents: Vec<String> = Vec::new();
    let mut chunk_paths: Vec<String> = Vec::new();
    let mut chunk_offsets: Vec<(u32, u16)> = Vec::new();

    for result in &vector_results {
        let file_idx = result.chunk_ref.file_index as usize;
        let path = entries
            .iter()
            .find(|(i, _)| *i == file_idx)
            .map(|(_, p)| p.as_str())
            .unwrap_or("<unknown>");

        let content = git::read_blob_at_head(&repo, path).unwrap_or_default();

        let offset = result.chunk_ref.byte_offset as usize;
        let len = result.chunk_ref.chunk_len as usize;
        let chunk_text = if offset + len <= content.len() {
            String::from_utf8_lossy(&content[offset..offset + len]).into_owned()
        } else if offset < content.len() {
            String::from_utf8_lossy(&content[offset..]).into_owned()
        } else {
            String::new()
        };

        chunk_contents.push(chunk_text);
        chunk_paths.push(path.to_string());
        chunk_offsets.push((result.chunk_ref.byte_offset, result.chunk_ref.chunk_len));
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

        writeln!(
            out,
            "━━━ Result {} ━━━ {} (offset {}, len {}) [score: {:.4}]",
            rank + 1,
            path,
            byte_offset,
            byte_len,
            item.relevance_score
        )?;
        writeln!(out, "{}", content)?;
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
        write!(out, "{}", text)?;
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

fn find_latest_index(kb_dir: &Path) -> Result<PathBuf> {
    if !kb_dir.exists() {
        anyhow::bail!(
            "No index directory found at {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    let mut kbi_files: Vec<_> = std::fs::read_dir(kb_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "kbi")
        })
        .collect();

    if kbi_files.is_empty() {
        anyhow::bail!(
            "No index files found in {}. Run 'kb index' first.",
            kb_dir.display()
        );
    }

    kbi_files.sort_by_key(|e| {
        std::cmp::Reverse(
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    Ok(kbi_files[0].path())
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

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

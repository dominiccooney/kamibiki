use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use kb::chunk::{self, Chunker};
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
    /// Start the server (not yet implemented)
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
    /// Debug / diagnostic subcommands
    Debug {
        #[command(subcommand)]
        command: DebugCommands,
    },
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
        Commands::Start => {
            eprintln!("Server mode not yet implemented.");
            Ok(())
        }
        Commands::Stop => {
            eprintln!("Server mode not yet implemented.");
            Ok(())
        }
        Commands::Debug { command } => match command {
            DebugCommands::Chunk {
                files,
                max_tokens,
                stdin_filename,
            } => cmd_debug_chunk(files, max_tokens, stdin_filename),
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
// Two-pass pipelined indexing:
//
// Pass 1 (fast, CPU only): Read every tracked file from git, chunk
// it, record chunk counts and lengths. This determines the index
// file layout.
//
// Write skeleton: Pre-allocate the full .kbi file with zeroed
// embedding slots.
//
// Pass 2 (pipelined, restartable): A producer thread re-reads files
// and re-chunks them (cheap) to produce document strings. It sends
// batches through a channel. The async consumer embeds each batch
// via the Voyage API and writes embeddings directly into the
// pre-allocated index file at the correct offsets.
//
// On restart: detect files with all-zero embedding slots and
// re-process only those.

/// Maximum files per embedding API request (Voyage limit = 16).
const EMBED_BATCH_SIZE: usize = 16;

/// Target max tokens per chunk for voyage-context-3.
const MAX_TOKENS: usize = 430;

/// A batch of files ready for embedding.
struct EmbedWork {
    /// Indices into the file_infos array.
    file_indices: Vec<usize>,
    /// One Vec<String> per file: the chunk texts for that file.
    documents: Vec<Vec<String>>,
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

    let chunker = chunk::tsv1_chunker()?;

    // ── Pass 1: Chunk all files to determine layout ──────────

    eprintln!("  Chunking files...");
    let mut file_infos: Vec<FileChunkInfo> = Vec::new();
    let mut total_chunks = 0usize;
    let mut skipped = 0usize;

    for (git_idx, path) in &entries {
        let content = match git::read_blob_at_head(&repo, path) {
            Ok(c) => c,
            Err(_) => { skipped += 1; continue; }
        };

        // Skip binary files.
        if content.iter().take(8192).any(|&b| b == 0) {
            continue;
        }

        match chunker.chunk(path, &content, MAX_TOKENS) {
            Ok(chunks) if !chunks.is_empty() => {
                total_chunks += chunks.len();
                file_infos.push(FileChunkInfo {
                    git_index_position: *git_idx,
                    path: path.clone(),
                    chunk_lengths: chunks.iter().map(|c| c.len as u16).collect(),
                });
            }
            Ok(_) => {} // empty
            Err(e) => {
                eprintln!("  warn: chunking failed for {}: {}", path, e);
                skipped += 1;
            }
        }
    }

    eprintln!("  Files to index: {}, total chunks: {}", file_infos.len(), total_chunks);
    if skipped > 0 {
        eprintln!("  Skipped: {}", skipped);
    }
    if total_chunks == 0 {
        eprintln!("  No chunks to index.");
        return Ok(());
    }

    // ── Write or locate skeleton index ───────────────────────

    let kb_dir = repo_cfg.path.join(".kb");
    std::fs::create_dir_all(&kb_dir)?;
    let index_path = kb_dir.join(format!(
        "{}.kbi",
        &commit_hex[..commit_hex.len().min(16)]
    ));

    if !index_path.exists() {
        let mut commit_hash: GitHash = [0; MAX_HASH_LEN];
        let hex_bytes = commit_hex.as_bytes();
        commit_hash[..hex_bytes.len().min(MAX_HASH_LEN)]
            .copy_from_slice(&hex_bytes[..hex_bytes.len().min(MAX_HASH_LEN)]);

        let header = IndexHeader {
            version: 1,
            commit_hash,
            parent_hash: [0; MAX_HASH_LEN],
        };

        index::write_skeleton(&index_path, &header, &file_infos)?;
        eprintln!("  Pre-allocated index: {}", index_path.display());
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

    let already_done = file_infos.len() - needs_work.len();
    eprintln!(
        "  Embedding {} files ({} already done)...",
        needs_work.len(),
        already_done
    );

    // ── Pass 2: Pipeline — producer chunks, consumer embeds ──

    let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedWork>(2);

    // Share file_infos with the producer thread.
    let file_infos = Arc::new(file_infos);
    let file_infos_for_producer = Arc::clone(&file_infos);
    let repo_path = repo_cfg.path.to_path_buf();

    let producer = std::thread::spawn(move || -> Result<()> {
        let repo = git::open_repo(&repo_path)?;
        let chunker = chunk::tsv1_chunker()?;

        let mut batch_indices: Vec<usize> = Vec::new();
        let mut batch_docs: Vec<Vec<String>> = Vec::new();

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

            let doc: Vec<String> = chunks
                .iter()
                .map(|c| String::from_utf8_lossy(c.content(&content)).into_owned())
                .collect();

            batch_indices.push(file_idx);
            batch_docs.push(doc);

            if batch_docs.len() >= EMBED_BATCH_SIZE {
                let work = EmbedWork {
                    file_indices: std::mem::take(&mut batch_indices),
                    documents: std::mem::take(&mut batch_docs),
                };
                if tx.blocking_send(work).is_err() {
                    break; // receiver dropped, consumer hit an error
                }
            }
        }

        // Flush final partial batch.
        if !batch_docs.is_empty() {
            let _ = tx.blocking_send(EmbedWork {
                file_indices: batch_indices,
                documents: batch_docs,
            });
        }

        Ok(())
    });

    // Consumer: embed each batch and write to the index file.
    let embedder = VoyageEmbedder::new(api_key.to_string());
    let mut embedded_count = already_done;
    let total_files = file_infos.len();

    while let Some(work) = rx.recv().await {
        match embedder.embed_documents(&work.documents).await {
            Ok(all_embeddings) => {
                for (i, &file_idx) in work.file_indices.iter().enumerate() {
                    let embeddings = &all_embeddings[i];
                    index::write_embeddings_at(
                        &index_path,
                        &layout,
                        file_idx,
                        embeddings,
                    )?;
                }
                embedded_count += work.file_indices.len();
                eprintln!(
                    "  Progress: {}/{} files",
                    embedded_count,
                    total_files
                );
            }
            Err(e) => {
                eprintln!("  error: embedding failed: {}. Progress saved, re-run to resume.", e);
                break;
            }
        }
    }

    // Wait for producer thread.
    match producer.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("  warn: producer error: {}", e),
        Err(_) => eprintln!("  warn: producer thread panicked"),
    }

    eprintln!(
        "  Wrote index: {} ({} files, {} embeddings)",
        index_path.display(),
        total_files,
        total_chunks
    );

    Ok(())
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

// ── kb debug chunk ───────────────────────────────────────────────

fn cmd_debug_chunk(
    files: Vec<String>,
    max_tokens: usize,
    stdin_filename: Option<String>,
) -> Result<()> {
    let chunker = chunk::tsv1_chunker()?;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if files.is_empty() {
        let filename = stdin_filename.as_deref().unwrap_or("stdin.txt");
        let mut content = Vec::new();
        io::stdin().read_to_end(&mut content)?;
        print_chunks(&mut out, filename, &content, &chunker, max_tokens)?;
    } else {
        for (i, path) in files.iter().enumerate() {
            if i > 0 {
                writeln!(out)?;
            }
            let content = std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path, e))?;
            print_chunks(&mut out, path, &content, &chunker, max_tokens)?;
        }
    }

    Ok(())
}

fn print_chunks(
    out: &mut impl Write,
    path: &str,
    content: &[u8],
    chunker: &impl Chunker,
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

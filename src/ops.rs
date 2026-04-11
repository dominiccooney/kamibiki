//! Shared indexing operations used by both the CLI and the MCP server.
//!
//! The core `index_repo` function implements the full indexing pipeline:
//! chunking, skeleton writing, delta detection, embedding, and progress
//! reporting via callbacks.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::chunk::{self, Chunker, TokenCounter};
use crate::core::git;
use crate::core::types::*;
use crate::embed::VoyageEmbedder;
use crate::index::{self, FileChunkInfo, IndexReader, MmapIndexReader};

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

/// Progress reporting callback for indexing operations.
pub trait IndexProgress: Send + Sync {
    /// Called when indexing starts for a repository.
    fn on_start(&self, repo_name: &str);
    /// Called to report a status message.
    fn on_message(&self, msg: &str);
    /// Called to report a warning.
    fn on_warning(&self, msg: &str);
    /// Called when chunking starts. `total` is the number of files.
    fn on_chunk_start(&self, total: usize);
    /// Called after chunking a file.
    fn on_chunk_progress(&self, completed: usize);
    /// Called when chunking finishes.
    fn on_chunk_done(&self);
    /// Called when embedding starts. `total` is the number of chunks.
    fn on_embed_start(&self, total: usize, already_done: usize);
    /// Called after embedding a batch.
    fn on_embed_progress(&self, completed: usize);
    /// Called when embedding finishes.
    fn on_embed_done(&self);
}

/// A simple progress reporter that writes to stderr (suitable for
/// non-interactive / MCP use).
pub struct StderrProgress;

impl IndexProgress for StderrProgress {
    fn on_start(&self, repo_name: &str) {
        eprintln!("Indexing '{}'...", repo_name);
    }
    fn on_message(&self, msg: &str) {
        eprintln!("  {}", msg);
    }
    fn on_warning(&self, msg: &str) {
        eprintln!("  warn: {}", msg);
    }
    fn on_chunk_start(&self, total: usize) {
        eprintln!("  Chunking {} files...", total);
    }
    fn on_chunk_progress(&self, _completed: usize) {
        // no-op for simple stderr
    }
    fn on_chunk_done(&self) {}
    fn on_embed_start(&self, total: usize, already_done: usize) {
        eprintln!("  Embedding {} chunks ({} already done)...", total, already_done);
    }
    fn on_embed_progress(&self, _completed: usize) {
        // no-op for simple stderr
    }
    fn on_embed_done(&self) {}
}

/// Result summary returned after indexing a repository.
#[derive(Debug, Clone)]
pub struct IndexResult {
    /// The commit that was indexed.
    pub commit_hex: String,
    /// Number of files in the index.
    pub file_count: usize,
    /// Total number of embedding chunks.
    pub total_chunks: usize,
    /// Whether this was a delta index (vs full).
    pub is_delta: bool,
    /// Whether an error occurred during embedding (partial progress saved).
    pub had_error: bool,
    /// Whether the index was already up to date (no work needed).
    pub already_up_to_date: bool,
}

/// Index a single repository. This is the core indexing pipeline,
/// shared between the CLI and MCP server.
///
/// `commit` is an optional git revision spec (commit hash, branch
/// name, tag, HEAD~1, etc.). When `None`, indexes at HEAD.
pub async fn index_repo(
    repo_cfg: &RepoConfig,
    api_key: &str,
    progress: &dyn IndexProgress,
    commit: Option<&str>,
) -> Result<IndexResult> {
    progress.on_start(&repo_cfg.name);

    let repo = git::open_repo(&repo_cfg.path)?;
    let commit_hex = git::resolve_commit_hex(&repo, commit)?;
    let ref_label = commit.unwrap_or("HEAD");
    progress.on_message(&format!(
        "{} commit: {}",
        ref_label,
        &commit_hex[..commit_hex.len().min(12)]
    ));

    let entries = git::entries_at_commit(&repo, &commit_hex)?;
    progress.on_message(&format!("Tracked files: {}", entries.len()));

    let kb_dir = repo_cfg.path.join(".kb");
    std::fs::create_dir_all(&kb_dir)?;
    let index_path = kb_dir.join(format!(
        "{}.kbi",
        &commit_hex[..commit_hex.len().min(16)]
    ));

    // ── Determine file layout ────────────────────────────────

    let mut is_delta = false;

    let file_infos: Vec<FileChunkInfo> = if index_path.exists() {
        let data = std::fs::read(&index_path)?;
        let infos = index::read_file_infos(&data, &entries)?;
        let total = infos.iter().map(|f| f.chunk_count()).sum::<usize>();
        progress.on_message(&format!(
            "Loaded layout from existing index ({} files, {} chunks)",
            infos.len(),
            total,
        ));
        infos
    } else {
        // Try delta mode: find an existing parent index to build on.
        let parent_info = find_parent_index_info(&kb_dir, &index_path);

        let (infos, parent_hash) = if let Some((_, ref parent_commit_hex)) = parent_info {
            is_delta = true;
            // Delta mode: only chunk files that changed since the parent.
            let diff = git::changed_files_between(&repo, parent_commit_hex, &commit_hex)?;
            let total_changes = diff.changed.len() + diff.deleted.len();
            progress.on_message(&format!(
                "Delta from {}: {} files added/modified, {} deleted",
                &parent_commit_hex[..parent_commit_hex.len().min(12)],
                diff.changed.len(),
                diff.deleted.len(),
            ));

            if total_changes == 0 {
                progress.on_message("No files changed. Index is up to date.");
                return Ok(IndexResult {
                    commit_hex,
                    file_count: 0,
                    total_chunks: 0,
                    is_delta: true,
                    had_error: false,
                    already_up_to_date: true,
                });
            }

            // Chunk added/modified files (those present at HEAD).
            let changed_set: HashSet<&str> = diff.changed.iter().map(|s| s.as_str()).collect();
            let changed_entries: Vec<(usize, String)> = entries
                .iter()
                .filter(|(_, path)| changed_set.contains(path.as_str()))
                .cloned()
                .collect();

            let mut infos = chunk_all_files(&repo_cfg.path, &changed_entries, &commit_hex, progress)?;

            // Any changed file that didn't produce chunks (binary,
            // empty, unreadable) also needs a tombstone to suppress
            // stale parent results.
            let chunked_paths: HashSet<&str> =
                infos.iter().map(|info| info.path.as_str()).collect();
            let mut unchunked_changed: Vec<&str> = changed_entries
                .iter()
                .map(|(_, p)| p.as_str())
                .filter(|p| !chunked_paths.contains(p))
                .collect();

            // Combine with explicitly deleted files.
            let mut tombstone_paths: Vec<String> = diff.deleted;
            tombstone_paths.extend(unchunked_changed.drain(..).map(|s| s.to_string()));

            // Add tombstone entries (0 chunks).
            let max_real_pos = entries.iter().map(|(i, _)| *i).max().unwrap_or(0);
            for (i, path) in tombstone_paths.iter().enumerate() {
                infos.push(FileChunkInfo {
                    git_index_position: max_real_pos + 1 + i,
                    path: path.clone(),
                    chunk_lengths: vec![], // tombstone: 0 chunks
                });
            }

            // Re-sort by git_index_position for the offset table.
            infos.sort_by_key(|info| info.git_index_position);

            let mut ph: GitHash = [0; MAX_HASH_LEN];
            let ph_bytes = parent_commit_hex.as_bytes();
            ph[..ph_bytes.len().min(MAX_HASH_LEN)]
                .copy_from_slice(&ph_bytes[..ph_bytes.len().min(MAX_HASH_LEN)]);

            (infos, ph)
        } else {
            // Full mode: no parent index exists.
            let infos = chunk_all_files(&repo_cfg.path, &entries, &commit_hex, progress)?;
            (infos, [0u8; MAX_HASH_LEN])
        };

        let mut commit_hash: GitHash = [0; MAX_HASH_LEN];
        let hex_bytes = commit_hex.as_bytes();
        commit_hash[..hex_bytes.len().min(MAX_HASH_LEN)]
            .copy_from_slice(&hex_bytes[..hex_bytes.len().min(MAX_HASH_LEN)]);

        let header = IndexHeader {
            version: 1,
            commit_hash,
            parent_hash,
        };

        index::write_skeleton(&index_path, &header, &infos)?;

        // Pre-fill reusable embeddings from the parent index.
        if let Some((ref parent_path, ref parent_commit_hex)) = parent_info {
            let reused = prefill_reusable_embeddings(
                &repo,
                &index_path,
                &infos,
                &commit_hex,
                parent_path,
                parent_commit_hex,
                progress,
            )?;
            if reused > 0 {
                progress.on_message(&format!("Reused {} embeddings from parent index", reused));
            }
        }

        progress.on_message(&format!("Pre-allocated index: {}", index_path.display()));
        infos
    };

    let total_chunks: usize = file_infos.iter().map(|f| f.chunk_count()).sum();
    if total_chunks == 0 {
        let tombstones = file_infos
            .iter()
            .filter(|f| f.chunk_lengths.is_empty())
            .count();
        if tombstones > 0 {
            progress.on_message(&format!(
                "Delta index written ({} tombstones, no embeddings needed).",
                tombstones,
            ));
        } else {
            progress.on_message("No chunks to index.");
        }
        return Ok(IndexResult {
            commit_hex,
            file_count: file_infos.len(),
            total_chunks: 0,
            is_delta,
            had_error: false,
            already_up_to_date: false,
        });
    }

    // ── Detect incomplete files ──────────────────────────────

    let data = std::fs::read(&index_path)?;
    let layout = Arc::new(index::index_layout(&data)?);
    let needs_work = index::detect_incomplete(&data, &layout);

    let skip_chunks: HashSet<(usize, usize)> = {
        let mut skips = HashSet::new();
        for &fi in &needs_work {
            let start = layout.chunk_prefix[fi];
            let end = layout.chunk_prefix[fi + 1];
            for ci in 0..(end - start) {
                let off = layout.embeddings_offset + (start + ci) * EMBEDDING_BYTES;
                if off + EMBEDDING_BYTES <= data.len()
                    && !data[off..off + EMBEDDING_BYTES].iter().all(|&b| b == 0)
                {
                    skips.insert((fi, ci));
                }
            }
        }
        skips
    };
    drop(data);

    if needs_work.is_empty() {
        progress.on_message(&format!(
            "Index complete ({} files, {} embeddings).",
            file_infos.len(),
            total_chunks
        ));
        return Ok(IndexResult {
            commit_hex,
            file_count: file_infos.len(),
            total_chunks,
            is_delta,
            had_error: false,
            already_up_to_date: true,
        });
    }

    let incomplete_chunks: usize = needs_work
        .iter()
        .map(|&i| file_infos[i].chunk_count())
        .sum();
    let already_done_chunks = total_chunks - incomplete_chunks + skip_chunks.len();

    // ── Pipeline: produce cross-file batches, embed concurrently ──

    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<EmbedBatch>(CONCURRENT_EMBED_REQUESTS * 2);

    let file_infos = Arc::new(file_infos);
    let file_infos_for_producer = Arc::clone(&file_infos);
    let repo_path = repo_cfg.path.to_path_buf();
    let skip_chunks = Arc::new(skip_chunks);
    let skip_chunks_for_producer = Arc::clone(&skip_chunks);
    let producer_commit_hex = commit_hex.clone();

    let producer = std::thread::spawn(move || -> Result<()> {
        let repo = git::open_repo(&repo_path)?;
        let chunker = chunk::tsv1_chunker()?;
        let tc = TokenCounter::for_voyage()?;

        let mut batch_chunks: Vec<String> = Vec::new();
        let mut batch_locations: Vec<(usize, usize)> = Vec::new();
        let mut batch_tokens: usize = 0;

        for &file_idx in &needs_work {
            let info = &file_infos_for_producer[file_idx];

            let content = match git::read_blob(&repo, &producer_commit_hex, &info.path) {
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
                if skip_chunks_for_producer.contains(&(file_idx, chunk_idx)) {
                    continue;
                }

                let text =
                    String::from_utf8_lossy(chunk.content(&content)).into_owned();
                let chunk_tokens = tc.count(text.as_bytes());

                // Flush if adding this chunk would exceed API limits.
                if !batch_chunks.is_empty()
                    && (batch_chunks.len() >= crate::embed::MAX_INPUTS_PER_REQUEST
                        || batch_tokens + chunk_tokens > crate::embed::MAX_REQUEST_TOKENS)
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

    progress.on_embed_start(total_chunks, already_done_chunks);
    let mut completed_chunks = already_done_chunks;
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
                        completed_chunks += count;
                        progress.on_embed_progress(completed_chunks);
                    }
                    Ok(Err(e)) => {
                        progress.on_warning(&format!(
                            "embedding failed: {}. Progress saved, re-run to resume.",
                            e
                        ));
                        had_error = true;
                        break;
                    }
                    Err(e) => {
                        progress.on_warning(&format!(
                            "task panicked: {}. Progress saved, re-run to resume.",
                            e
                        ));
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
                completed_chunks += count;
                progress.on_embed_progress(completed_chunks);
            }
            Ok(Err(e)) => {
                if !had_error {
                    progress.on_warning(&format!(
                        "embedding failed: {}. Progress saved, re-run to resume.",
                        e
                    ));
                    had_error = true;
                }
            }
            Err(e) => {
                if !had_error {
                    progress.on_warning(&format!(
                        "task panicked: {}. Progress saved, re-run to resume.",
                        e
                    ));
                    had_error = true;
                }
            }
        }
    }

    progress.on_embed_done();

    // Wait for producer thread.
    match producer.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => progress.on_warning(&format!("producer error: {}", e)),
        Err(_) => progress.on_warning("producer thread panicked"),
    }

    let file_count = file_infos.len();
    if !had_error {
        progress.on_message(&format!(
            "Wrote index: {} ({} files, {} embeddings)",
            index_path.display(),
            file_count,
            total_chunks,
        ));
    }

    Ok(IndexResult {
        commit_hex,
        file_count,
        total_chunks,
        is_delta,
        had_error,
        already_up_to_date: false,
    })
}

/// Chunk all tracked files to determine index layout. Uses half the
/// available CPU cores. Returns FileChunkInfo for each non-empty,
/// non-binary file.
///
/// `commit_hex` is the commit hash to read blobs from.
pub fn chunk_all_files(
    repo_path: &Path,
    entries: &[(usize, String)],
    commit_hex: &str,
    progress: &dyn IndexProgress,
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

    progress.on_chunk_start(entries.len());

    let chunk_results: Vec<Option<FileChunkInfo>> = pool.install(|| {
        use std::cell::RefCell;
        use std::sync::atomic::{AtomicUsize, Ordering};

        thread_local! {
            static THREAD_REPO: RefCell<Option<gix::Repository>> = const { RefCell::new(None) };
        }

        let completed = AtomicUsize::new(0);

        entries
            .par_iter()
            .map(|(git_idx, path)| {
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress.on_chunk_progress(done);

                let content = THREAD_REPO.with(|cell| {
                    let mut opt = cell.borrow_mut();
                    let repo = opt.get_or_insert_with(|| {
                        git::open_repo(&repo_path)
                            .expect("failed to open repo in worker thread")
                    });
                    git::read_blob(repo, commit_hex, path).ok()
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
                        eprintln!("  warn: chunking failed for {}: {}", path, e);
                        None
                    }
                }
            })
            .collect()
    });

    progress.on_chunk_done();

    let file_infos: Vec<FileChunkInfo> = chunk_results.into_iter().flatten().collect();
    let total_chunks: usize = file_infos.iter().map(|f| f.chunk_count()).sum();
    let skipped = entries.len() - file_infos.len();

    progress.on_message(&format!(
        "Chunked: {} files → {} chunks (using {} threads)",
        file_infos.len(),
        total_chunks,
        num_threads,
    ));
    if skipped > 0 {
        progress.on_message(&format!("Skipped: {}", skipped));
    }

    Ok(file_infos)
}

/// Find the most recent existing index file to use as a parent for
/// delta indexing. Returns `(path, commit_hex)` of the parent index,
/// or `None` if no suitable parent exists.
fn find_parent_index_info(kb_dir: &Path, exclude_path: &Path) -> Option<(PathBuf, String)> {
    let mut kbi_files: Vec<_> = std::fs::read_dir(kb_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().is_some_and(|ext| ext == "kbi") && p != exclude_path
        })
        .collect();

    if kbi_files.is_empty() {
        return None;
    }

    // Most recent first.
    kbi_files.sort_by_key(|e| {
        std::cmp::Reverse(
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    let path = kbi_files[0].path();
    let reader = MmapIndexReader::open(&path).ok()?;
    let commit_hex = commit_hash_to_hex(&reader.header().commit_hash);

    if commit_hex.is_empty() {
        return None;
    }

    Some((path, commit_hex))
}

/// Extract the commit hash hex string from a GitHash.
fn commit_hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
    String::from_utf8_lossy(&hash[..end]).into_owned()
}

/// Pre-fill reusable embeddings from a parent index into a newly
/// written skeleton. For each file in the new delta index, this
/// finds the same file in the parent index and compares chunk texts.
/// Chunks with identical text reuse the parent's embedding.
///
/// Returns the number of embeddings reused.
fn prefill_reusable_embeddings(
    repo: &gix::Repository,
    new_index_path: &Path,
    new_file_infos: &[FileChunkInfo],
    new_commit_hex: &str,
    parent_path: &Path,
    parent_commit_hex: &str,
    progress: &dyn IndexProgress,
) -> Result<usize> {
    let old_reader = MmapIndexReader::open(parent_path)?;

    let old_entries = git::entries_at_commit(repo, parent_commit_hex)?;
    let old_data = std::fs::read(parent_path)?;
    let old_file_infos = match index::read_file_infos(&old_data, &old_entries) {
        Ok(infos) => infos,
        Err(e) => {
            progress.on_warning(&format!("cannot read parent index metadata: {}", e));
            return Ok(0);
        }
    };

    // Build path → old file info index.
    let old_path_to_idx: HashMap<&str, usize> = old_file_infos
        .iter()
        .enumerate()
        .map(|(i, info)| (info.path.as_str(), i))
        .collect();

    // Parse the new index layout for writing pre-filled embeddings.
    let new_data = std::fs::read(new_index_path)?;
    let new_layout = index::index_layout(&new_data)?;
    drop(new_data);

    let mut locations: Vec<(usize, usize)> = Vec::new();
    let mut embeddings: Vec<BinaryEmbedding> = Vec::new();

    for (fi, info) in new_file_infos.iter().enumerate() {
        let old_fi = match old_path_to_idx.get(info.path.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };

        let old_emb_range = old_reader.file_embedding_range(old_fi);
        let old_chunk_lengths = old_reader.file_chunk_lengths(old_fi);

        let old_content = match git::read_blob(repo, parent_commit_hex, &info.path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let new_content = match git::read_blob(repo, new_commit_hex, &info.path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut old_text_to_emb: HashMap<&[u8], BinaryEmbedding> = HashMap::new();
        let mut old_offset = 0usize;
        for (ci, &len) in old_chunk_lengths.iter().enumerate() {
            let end = old_offset + len as usize;
            if end <= old_content.len() {
                let text = &old_content[old_offset..end];
                let emb_idx = old_emb_range.start + ci;
                let emb = old_reader.embedding(emb_idx);
                if !emb.iter().all(|&b| b == 0) {
                    old_text_to_emb.insert(text, *emb);
                }
            }
            old_offset = end;
        }

        let mut new_offset = 0usize;
        for (ci, &len) in info.chunk_lengths.iter().enumerate() {
            let end = new_offset + len as usize;
            if end <= new_content.len() {
                let text = &new_content[new_offset..end];
                if let Some(emb) = old_text_to_emb.get(text) {
                    locations.push((fi, ci));
                    embeddings.push(*emb);
                }
            }
            new_offset = end;
        }
    }

    let count = locations.len();
    if !locations.is_empty() {
        index::write_embeddings_scattered(
            new_index_path,
            &new_layout,
            &locations,
            &embeddings,
        )?;
    }

    Ok(count)
}

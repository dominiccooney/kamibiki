//! MCP (Model Context Protocol) server for Kamibiki.
//!
//! Implements JSON-RPC 2.0 over stdio, exposing `kb_status`, `kb_search`,
//! and `kb_index` as MCP tools.

use std::io::{self, BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Value, json};

/// Parse an optional `cwd` argument from the JSON-RPC arguments.
/// Used by tools that accept `.` as a repo name — the MCP client
/// should pass its own working directory here since the server's
/// process cwd may differ.
fn parse_cwd(args: &Value) -> Option<std::path::PathBuf> {
    args.get("cwd")
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
}

use crate::core::config;
use crate::core::git;
use crate::core::types::*;
use crate::embed::{Embedder, VoyageEmbedder};
use crate::index::{IndexReader, MmapIndexReader};
use crate::ops::{self, StderrProgress};
use crate::search::{IndexChain, Reranker, VoyageReranker, chain_search, load_index_chain};

// ── MCP stdio server ─────────────────────────────────────────────

/// Run the MCP server, reading JSON-RPC from stdin and writing
/// responses to stdout. Diagnostic messages go to stderr.
pub fn run_mcp_server() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_mcp_loop())
}

async fn run_mcp_loop() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    eprintln!("kamibiki: MCP server starting on stdio");

    let mut buf = String::new();
    loop {
        buf.clear();

        // Read one line of JSON-RPC (newline-delimited).
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            // EOF — client closed stdin.
            eprintln!("kamibiki: stdin closed, shutting down");
            break;
        }

        let line = buf.trim();
        if line.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                write_message(&mut writer, &err_resp)?;
                continue;
            }
        };

        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        // Notifications (no id) — just acknowledge silently.
        if id.is_none() {
            match method {
                "notifications/initialized" => {
                    eprintln!("kamibiki: client initialized");
                }
                "notifications/cancelled" => {
                    // Cancellation — we don't support long-running
                    // requests yet, so just ignore.
                }
                _ => {
                    eprintln!("kamibiki: ignoring notification: {}", method);
                }
            }
            continue;
        }

        let id = id.unwrap();

        let response = match method {
            "initialize" => handle_initialize(&id),
            "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
            "tools/list" => handle_tools_list(&id),
            "tools/call" => handle_tools_call(&id, &msg).await,
            _ => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {}", method)
                }
            }),
        };

        write_message(&mut writer, &response)?;
    }

    Ok(())
}

fn write_message(writer: &mut impl Write, msg: &Value) -> Result<()> {
    let s = serde_json::to_string(msg)?;
    writeln!(writer, "{}", s)?;
    writer.flush()?;
    Ok(())
}

// ── initialize ───────────────────────────────────────────────────

fn handle_initialize(id: &Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "kamibiki",
                "version": env!("CARGO_PKG_VERSION")
            },
            "instructions": "Kamibiki provides semantic code search over indexed git repositories. Prefer kb_search over grep or regex search when you need to find code by meaning rather than by exact text — for example, when looking for how a feature is implemented, finding usage patterns, understanding architecture, or locating code related to a concept when you don't know the precise identifiers or strings involved. kb_search accepts natural language queries and returns the most relevant code chunks ranked by semantic similarity. Use kb_status to check which repositories are indexed and their current state. Use kb_index to update the index after code changes."
        }
    })
}

// ── tools/list ───────────────────────────────────────────────────

fn handle_tools_list(id: &Value) -> Value {
    let repo_listing = build_repo_listing();

    let search_desc = format!(
        "Search an indexed git repository for code chunks relevant to a query. Returns ranked results with file paths, byte offsets, and code content. The repository must have been previously indexed with `kb add` and `kb index`. Use this tool instead of grep or file search when you don't know the exact string to search for — it understands natural language queries like \"error handling in the API layer\" or \"where are database connections configured\" and finds semantically relevant code even when no exact keyword match exists.{}",
        repo_listing
    );

    let index_desc = format!(
        "Update the search index for one or more repositories. This chunks files, computes embeddings via the Voyage AI API, and writes the index. Supports delta indexing (only changed files are re-embedded). This operation can take seconds to minutes depending on repository size and number of changes. The operation is restartable: if interrupted, re-running will resume from where it left off.{}",
        repo_listing
    );
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "kb_search",
                    "description": search_desc,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Repository name as registered with `kb add`, or '.' for the current repository"
                            },
                            "query": {
                                "type": "string",
                                "description": "Natural language or code search query"
                            },
                            "top": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10
                            },
                            "commit": {
                                "type": "string",
                                "description": "Git revision to search from (commit hash, branch name, tag, HEAD~1, etc.). Defaults to HEAD when not specified."
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Absolute path used to resolve '.' as a repository name. Required when name='.' — typically the client's current working directory. If the path is inside a git worktree, the appropriate registered repository is selected and the worktree's HEAD is used for revision resolution. Ignored when name is not '.'."
                            }
                        },
                        "required": ["name", "query"]
                    }
                },
                {
                    "name": "kb_status",
                    "description": "Show the indexing status of registered repositories, including the number of index files, total index size, latest indexed commit, and embedding count. For repositories with linked git worktrees, each worktree is listed under the repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Repository name to check status for. If omitted, shows all repositories. Pass '.' to query the repository containing `cwd`."
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Absolute path used to resolve '.' as a repository name. Only consulted when name='.'."
                            }
                        }
                    }
                },
                {
                    "name": "kb_index",
                    "description": index_desc,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Repository names to index. If omitted or empty, indexes all registered repositories. Use '.' to index the repository containing `cwd`."
                            },
                            "commit": {
                                "type": "string",
                                "description": "Git revision to index at (commit hash, branch name, tag, HEAD~1, etc.). Defaults to HEAD when not specified. For linked worktrees this is resolved against the worktree's refs."
                            },
                            "compact": {
                                "type": "boolean",
                                "description": "When true, write a self-contained root index (parent_hash = 0) by re-chunking and re-embedding every tracked file, discarding any existing delta at this commit. Chain walks terminate at a compact index. Defaults to false.",
                                "default": false
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Absolute path used to resolve any '.' entries in `names`. Typically the client's current working directory."
                            }
                        }
                    }
                }

            ]

        }
    })
}

// ── Dynamic repo listing for tool descriptions ──────────────────

/// Build a human-readable summary of indexed repositories to append
/// to tool descriptions. This gives the agent immediate visibility
/// into which repos are ready to search, and the file count puts
/// subtle pressure on using the tool for larger repositories.
///
/// Repos are sorted by name. Linked worktrees are listed under their repo.
///
/// When no repositories are configured, returns an empty string so
/// the tool description remains clean.
fn build_repo_listing() -> String {
    let cfg = match config::load_config() {
        Ok(c) => c,
        Err(_) => return String::new(),
    };

    if cfg.repos.is_empty() {
        return String::new();
    }

    let mut repos: Vec<&RepoConfig> = cfg.repos.iter().collect();
    repos.sort_by(|a, b| a.name.cmp(&b.name));

    let mut lines = vec!["\n\nIndexed repositories:".to_string()];

    for repo in &repos {
        let file_count = read_file_count(&repo.path);

        match file_count {
            Some(n) => lines.push(format!(
                "• {} ({}) — {} file{} indexed",
                repo.name,
                repo.path.display(),
                n,
                if n == 1 { "" } else { "s" }
            )),
            None => lines.push(format!(
                "• {} ({}) — not yet indexed",
                repo.name,
                repo.path.display()
            )),
        }

        if let Ok(opened) = git::open_repo(&repo.path) {
            if let Ok(worktrees) = git::list_worktrees(&opened) {
                let linked: Vec<_> = worktrees.iter().filter(|w| !w.is_main).collect();
                if !linked.is_empty() {
                    lines.push("  Linked worktrees:".to_string());
                    for wt in linked {
                        lines.push(format!("    {}", wt.path.display()));
                    }
                }
            }
        }
    }

    lines.join("\n")
}

// ── Helpers for build_repo_listing ─────────────────────────────

/// Read the file count from the newest `.kbi` index for a repo.
/// Returns `None` when the repo hasn't been indexed yet.
fn read_file_count(repo_path: &Path) -> Option<usize> {
    let kb_dir = repo_path.join(".kb");
    if !kb_dir.exists() {
        return None;
    }

    let mut kbi_files: Vec<_> = match std::fs::read_dir(&kb_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "kbi"))
            .collect(),
        Err(_) => return None,
    };

    if kbi_files.is_empty() {
        return None;
    }

    kbi_files.sort_by_key(|e| {
        std::cmp::Reverse(
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    MmapIndexReader::open(&kbi_files[0].path())
        .ok()
        .map(|r| r.file_count())
}

// ── tools/call ───────────────────────────────────────────────────

async fn handle_tools_call(id: &Value, msg: &Value) -> Value {
    let params = msg.get("params").cloned().unwrap_or(json!({}));
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match tool_name {
        "kb_search" => match tool_search(&arguments).await {
            Ok(text) => tool_result(id, &text, false),
            Err(e) => tool_result(id, &format!("Error: {:#}", e), true),
        },
        "kb_status" => match tool_status(&arguments) {
            Ok(text) => tool_result(id, &text, false),
            Err(e) => tool_result(id, &format!("Error: {:#}", e), true),
        },
        "kb_index" => match tool_index(&arguments).await {
            Ok(text) => tool_result(id, &text, false),
            Err(e) => tool_result(id, &format!("Error: {:#}", e), true),
        },
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32602,
                "message": format!("Unknown tool: {}", tool_name)
            }
        }),
    }
}

fn tool_result(id: &Value, text: &str, is_error: bool) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ],
            "isError": is_error
        }
    })
}

// ── kb_search tool ───────────────────────────────────────────────

async fn tool_search(args: &Value) -> Result<String> {
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required parameter: name"))?;
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required parameter: query"))?;
    let top = args.get("top").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let commit = args.get("commit").and_then(|v| v.as_str());
    let cwd = parse_cwd(args);

    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    let (repo_cfg, worktree_path) = resolve_repo_and_worktree(&cfg, name, cwd.as_deref())?;

    // Open git from the current worktree so HEAD resolves correctly
    // when the user is operating from a linked worktree.
    let repo = git::open_repo(&worktree_path)?;

    // Resolve the commit ref (branch name, tag, hash, HEAD~1, etc.)
    // to a full hex commit hash. Defaults to HEAD when not specified.
    let resolved_commit = git::resolve_commit_hex(&repo, commit)?;
    let ref_label = commit.unwrap_or("HEAD");

    // The shared `.kb/` always lives at the main registered path.
    let kb_dir = repo_cfg.path.join(".kb");

    let IndexChain {
        readers: chain,
        commits_behind,
    } = load_index_chain(&kb_dir, &repo, Some(&resolved_commit))?;

    let embedder = VoyageEmbedder::new(api_key.clone());
    let query_embedding = embedder.embed_query(query).await?;

    let vector_top_n = 200;
    let vector_results = chain_search(&chain, &repo, &query_embedding, vector_top_n)?;

    if vector_results.is_empty() {
        return Ok("No results found.".to_string());
    }

    // Read chunk content from the commit each result came from.
    let mut chunk_contents: Vec<String> = Vec::new();
    let mut chunk_paths: Vec<String> = Vec::new();
    let mut chunk_offsets: Vec<(u32, u16)> = Vec::new();
    let mut chunk_start_lines: Vec<usize> = Vec::new();

    for result in &vector_results {
        let content = git::read_blob(&repo, &result.commit_hex, &result.path).unwrap_or_default();

        let offset = result.byte_offset as usize;
        let len = result.chunk_len as usize;
        let start_line = crate::snippet::start_line_for_offset(&content, offset);
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

    let mut output = String::new();

    if commits_behind > 0 {
        output.push_str(&format!(
            "Note: index is {} commit{} behind {}. Run 'kb index' to update.\n\n",
            commits_behind,
            if commits_behind == 1 { "" } else { "s" },
            ref_label,
        ));
    }

    for (rank, item) in reranked.iter().enumerate() {
        let path = &chunk_paths[item.index];
        let (byte_offset, byte_len) = chunk_offsets[item.index];
        let content = &chunk_contents[item.index];
        let start_line = chunk_start_lines[item.index];

        if rank > 0 {
            output.push('\n');
        }
        let numbered = crate::snippet::format_with_line_numbers(content, start_line);
        output.push_str(&format!(
            "━━━ Result {} ━━━ {} (offset {}, len {}) [score: {:.4}]\n",
            rank + 1,
            path,
            byte_offset,
            byte_len,
            item.relevance_score,
        ));
        // Staleness compares against the file on disk in the
        // worktree the user is searching from.
        let staleness = git::check_snippet_staleness(
            &worktree_path,
            path,
            byte_offset as usize,
            byte_len as usize,
            content.as_bytes(),
        );

        if let Some(note) = staleness.note() {
            output.push_str(note);
            output.push('\n');
        }
        output.push_str(&numbered);
    }

    Ok(output)
}

// ── kb_status tool ───────────────────────────────────────────────

fn tool_status(args: &Value) -> Result<String> {
    let name = args.get("name").and_then(|v| v.as_str());
    let cwd = parse_cwd(args);
    let cfg = config::load_config()?;

    let repos: Vec<&RepoConfig> = match name {
        Some(n) => {
            let repo = resolve_repo(&cfg, n, cwd.as_deref())?;
            vec![repo]
        }
        None => cfg.repos.iter().collect(),
    };

    if repos.is_empty() {
        return Ok("No repositories configured. Use 'kb add' to add one.".to_string());
    }

    let mut output = String::new();

    for (i, repo) in repos.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format!(
            "Repository: {} ({})\n",
            repo.name,
            repo.path.display()
        ));

        // List worktrees (main + linked). All worktrees share the
        // same `.kb/` since git objects are shared.
        if let Ok(opened) = git::open_repo(&repo.path) {
            if let Ok(worktrees) = git::list_worktrees(&opened) {
                let linked_count = worktrees.iter().filter(|w| !w.is_main).count();
                if linked_count > 0 {
                    output.push_str("  Worktrees:\n");
                    for wt in &worktrees {
                        output.push_str(&format!("    {}: {}\n", wt.label(), wt.path.display()));
                    }
                }
            }
        }

        let kb_dir = repo.path.join(".kb");
        if !kb_dir.exists() {
            output.push_str("  Status: not indexed\n");
            continue;
        }

        let mut kbi_files: Vec<_> = std::fs::read_dir(&kb_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "kbi"))
            .collect();

        if kbi_files.is_empty() {
            output.push_str("  Status: not indexed\n");
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

            output.push_str(&format!("  Index files: {}\n", kbi_files.len()));
            output.push_str(&format!(
                "  Total index size: {}\n",
                format_bytes(total_size)
            ));

            if let Ok(reader) = MmapIndexReader::open(&kbi_files[0].path()) {
                let commit_hex = commit_hash_to_hex(&reader.header().commit_hash);
                let short = &commit_hex[..commit_hex.len().min(12)];
                output.push_str(&format!("  Latest indexed commit: {}\n", short));
                output.push_str(&format!(
                    "  Files in latest index: {}\n",
                    reader.file_count()
                ));
                output.push_str(&format!(
                    "  Embeddings in latest index: {}\n",
                    reader.embedding_count()
                ));
            }
        }
    }

    Ok(output)
}

// ── kb_index tool ────────────────────────────────────────────────

async fn tool_index(args: &Value) -> Result<String> {
    let cfg = config::load_config()?;
    let api_key = cfg
        .voyage_api_key
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No Voyage API key configured. Run 'kb init' first."))?
        .clone();

    // Resolve which repos to index.
    let names: Vec<String> = match args.get("names").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => Vec::new(),
    };
    let commit = args.get("commit").and_then(|v| v.as_str());
    let compact = args
        .get("compact")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let cwd = parse_cwd(args);

    let repo_names = resolve_names(&cfg, &names)?;
    let progress = StderrProgress;
    let mut output = String::new();

    for (i, repo_name) in repo_names.iter().enumerate() {
        let (repo_cfg, worktree_path) = resolve_repo_and_worktree(&cfg, repo_name, cwd.as_deref())?;

        let result = ops::index_repo(
            repo_cfg,
            &worktree_path,
            &api_key,
            &progress,
            commit,
            compact,
        )
        .await
        .map_err(|e| anyhow::anyhow!("failed to index '{}': {:#}", repo_cfg.name, e))?;

        if i > 0 {
            output.push('\n');
        }

        if result.already_up_to_date {
            output.push_str(&format!(
                "Repository '{}': index is up to date at commit {}.",
                repo_name,
                &result.commit_hex[..result.commit_hex.len().min(12)],
            ));
        } else if result.had_error {
            output.push_str(&format!(
                "Repository '{}': indexing partially completed at commit {} ({} files, {} chunks). \
                 Some embedding requests failed; progress was saved. Re-run to resume.",
                repo_name,
                &result.commit_hex[..result.commit_hex.len().min(12)],
                result.file_count,
                result.total_chunks,
            ));
        } else {
            let mode = if result.is_delta { "delta" } else { "full" };
            output.push_str(&format!(
                "Repository '{}': {} index complete at commit {} ({} files, {} chunks).",
                repo_name,
                mode,
                &result.commit_hex[..result.commit_hex.len().min(12)],
                result.file_count,
                result.total_chunks,
            ));
        }
    }

    Ok(output)
}

// ── Helpers (shared with main.rs logic) ──────────────────────────

fn resolve_repo<'a>(
    cfg: &'a KbConfig,
    name: &str,
    cwd_override: Option<&std::path::Path>,
) -> Result<&'a RepoConfig> {
    let (repo_cfg, _) = resolve_repo_and_worktree(cfg, name, cwd_override)?;
    Ok(repo_cfg)
}

/// Like `resolve_repo`, but also returns the worktree path from
/// which to open the git repository (for HEAD resolution and
/// working-tree reads).
///
/// For a named repo, that's `repo_cfg.path` (the main worktree).
/// For `.`, it's the worktree containing `cwd_override` (or the
/// server's process cwd when `None`) — which may be a linked
/// worktree sharing the same `.kb/` as the main.
///
/// MCP clients should pass the user's working directory via
/// `cwd_override` when sending `.` as the name, since the MCP
/// server's process cwd is typically the location where it was
/// launched and may not match the client's current context.
fn resolve_repo_and_worktree<'a>(
    cfg: &'a KbConfig,
    name: &str,
    cwd_override: Option<&std::path::Path>,
) -> Result<(&'a RepoConfig, std::path::PathBuf)> {
    if name == "." {
        let cwd = match cwd_override {
            Some(p) => p.to_path_buf(),
            None => std::env::current_dir()?,
        };
        let repo = git::open_repo(&cwd).with_context(|| {
            format!(
                "'{}' is not inside a git repository (pass 'cwd' to point at a worktree)",
                cwd.display()
            )
        })?;

        let current_workdir = repo
            .workdir()
            .ok_or_else(|| anyhow::anyhow!("bare repository not supported"))?
            .to_path_buf();
        let current_workdir = std::fs::canonicalize(&current_workdir).unwrap_or(current_workdir);
        let main_path = git::main_worktree_path(&repo)
            .ok_or_else(|| anyhow::anyhow!("bare repository not supported"))?;
        let main_canon = std::fs::canonicalize(&main_path).unwrap_or(main_path);

        let repo_cfg = cfg
            .repos
            .iter()
            .find(|r| {
                std::fs::canonicalize(&r.path)
                    .map(|p| p == main_canon)
                    .unwrap_or(false)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "current directory's repository is not registered. Use 'kb add' first."
                )
            })?;

        Ok((repo_cfg, current_workdir))
    } else {
        let repo_cfg = cfg
            .repos
            .iter()
            .find(|r| r.name == name)
            .ok_or_else(|| anyhow::anyhow!("unknown repository: '{}'", name))?;
        Ok((repo_cfg, repo_cfg.path.clone()))
    }
}

/// Resolve a list of repository names, expanding aliases and
/// defaulting to all repos when the list is empty.
fn resolve_names(cfg: &KbConfig, names: &[String]) -> Result<Vec<String>> {
    if names.is_empty() {
        return Ok(cfg.repos.iter().map(|r| r.name.clone()).collect());
    }

    let mut result = Vec::new();
    for name in names {
        if let Some(alias) = cfg.aliases.iter().find(|a| a.name == *name) {
            result.extend(alias.repos.clone());
        } else {
            result.push(name.clone());
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

/// Extract the commit hash hex string from a GitHash (stored as ASCII
/// hex bytes padded with zeroes).
fn commit_hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
    String::from_utf8_lossy(&hash[..end]).into_owned()
}

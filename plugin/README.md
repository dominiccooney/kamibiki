# Kamibiki Plugin for Cline

A [Cline SDK](https://github.com/cline/cline) plugin that gives your coding agent semantic code search powered by Kamibiki. Instead of grep or text-based search, the agent can ask natural language questions like "how does authentication work" or "where are database connections configured" and get back the most relevant code chunks, ranked by meaning.

The plugin auto-indexes your workspace on each session start (using delta indexing, so only changed files are re-embedded) and exposes a `semantic_search` tool the agent can call at any time.

## Prerequisites

1. Install the `kb` binary:

```sh
git clone https://github.com/dominiccooney/kamibiki.git
cd kamibiki
cargo build --release
```

Copy `target/release/kb` somewhere on your `PATH`, or note its location for later.

2. Get a [Voyage AI](https://www.voyageai.com/) API key (used for embeddings and reranking).

3. Configure your API key:

```sh
kb init
```

This saves the key to `~/.kb.conf`.

4. Verify it works by indexing a repo manually:

```sh
cd /path/to/your/project
kb add myproject .
kb index myproject
kb search myproject "some concept"
```

If you see ranked code results, you're good to go.

## Installing the Plugin

Copy `kamibiki.ts` to one of these locations:

For a single project (plugin loads only when working in that project):
```sh
cp kamibiki.ts /path/to/your/project/.cline/plugins/kamibiki.ts
```

For all projects (plugin loads globally):
```sh
mkdir -p ~/.cline/plugins
cp kamibiki.ts ~/.cline/plugins/kamibiki.ts
```

That's it. Cline auto-discovers plugins in these directories.

## What Happens at Runtime

1. When a Cline session starts, the `beforeRun` hook makes sure the workspace is registered, then runs `kb index .` in your workspace. This creates or updates the search index at the current HEAD commit. Delta indexing means only files that changed since the last index are re-embedded, so this is fast after the initial index.

2. The plugin checks `kb status .` first. If the current repository is not registered yet, it registers it with a stable generated name like `cline-myproject-8f2c1d3e4a5b`. It does not claim the global `.` name in `~/.kb.conf`.

3. The agent gets a `semantic_search` tool it can call with a natural language query and an optional result count. Results come back with file paths, line numbers, code snippets, and relevance scores. If `kb` is not installed, the Voyage AI key is missing, or indexing has not completed, the tool returns a structured error instead of crashing the agent run.

## The `semantic_search` Tool

Parameters:
- `query` (string, required): A natural language or code search query
- `top` (integer, optional): Number of results to return, from 1 to 50 (default: 10)
- `dirs` (string[], optional): Directories to focus the search on. Accepts absolute, cwd-relative, or repository-relative paths; each is relativized against the repository root by the `kb` CLI. Omit to search the whole repository.
- `exclude_dirs` (string[], optional): Directories to exclude from results. Same path forms as `dirs`. Exclusions take precedence over `dirs`.

This mirrors the `kb` CLI flags exactly: each `dirs` entry becomes a repeatable `--dir`, and each `exclude_dirs` entry becomes a repeatable `--exclude-dir`. Scoping a search to the relevant part of a large monorepo dramatically improves result quality.

Example queries the agent might use:
- "error handling in the API layer"
- "how are WebSocket connections managed"
- "database migration logic"
- "authentication and session tokens"
- "rate limiting implementation"

Results include file paths with line numbers, the actual code chunks, and a relevance score from Voyage AI's reranker, so the agent knows exactly where to look and how confident the match is.

## Customization

The plugin is a single TypeScript file. Fork and modify it to fit your workflow:

- Change the default or maximum result count by editing `DEFAULT_TOP` or `MAX_TOP`
- Add a `rule` via `api.registerRule()` to always include certain search context in the system prompt
- Add an `afterRun` hook to re-index after the agent finishes modifying files
- Use `api.registerMessageBuilder()` to automatically inject relevant code context into every model request based on the conversation topic

See the [Cline SDK plugin documentation](https://github.com/cline/cline) for the full plugin API.

## Development

The plugin targets `@cline/core` `0.0.42`. To work on it locally:

```sh
cd plugin
bun install
bun run typecheck   # type-checks against the @cline/core SDK
bun test            # runs the unit tests
```

The tests live in `kamibiki.test.ts` and cover the pure
argument-building logic (`buildSearchArgs`) — including result-count
clamping and the `--dir` / `--exclude-dir` flag mapping — so they run
without a `kb` binary or a model.

## How It Works Under the Hood

The plugin shells out to the `kb` CLI binary. When the agent calls `semantic_search`:

1. `kb search . <query> -n <top>` runs in the workspace directory
2. Kamibiki embeds the query using Voyage AI (`voyage-code-3` model)
3. Binary-quantized Hamming distance search finds the top 200 candidates from the local index (no network call, very fast)
4. Voyage AI's reranker (`rerank-2.5-lite`) scores the candidates for final ranking
5. Results are returned with file paths, 1-based line numbers, byte offsets, and the actual code content

The search index lives in a `.kb/` directory at the repository root. Each index file (`.kbi`) is a memory-mapped binary format optimized for fast search. Delta indexes only store changed files and point to a parent index, keeping incremental updates cheap. Run `kb gc` periodically to clean up stale index files from old commits.

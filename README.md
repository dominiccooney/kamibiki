# Kamibiki (KB)

Kamibiki is a contextual search engine for agentic coding. It indexes
git repositories locally using embeddings and provides fast semantic
search from the command line or as an MCP server for coding agents.

## Getting Started

### Prerequisites

- **Rust** — Install from https://rust-lang.org/tools/install
- **Voyage AI API key** — Sign up at https://www.voyageai.com/ and
  create an API key. Kamibiki uses Voyage AI for embeddings and
  reranking.

### Install

```sh
git clone https://github.com/dominiccooney/kamibiki.git
cd kamibiki
cargo build --release
```

The binary is at `target/release/kb`. You can copy it somewhere on
your `PATH`, or run it directly.

### Set up your API key

```sh
kb init
```

This prompts for your Voyage AI API key and saves it to `~/.kb.conf`.

### Add a repository

```sh
kb add <name> <path/to/git/repo>
```

`<name>` is a shorthand alias you'll use in other commands. For
example:

```sh
kb add myproject ~/src/myproject
```

### Index the repository

```sh
kb index <name>
```

This chunks all tracked files, computes embeddings via the Voyage AI
API, and writes index files to a `.kb` folder at the repository root.
Add `.kb` to your `.gitignore`.

Indexing supports delta updates — subsequent runs only re-embed
changed files. If interrupted, re-running resumes from where it left
off.

You can index at a specific git revision with `--commit`:

```sh
kb index myproject --commit my-feature-branch
kb index myproject --commit abc1234
kb index myproject --commit HEAD~3
```

When `--commit` is omitted, it indexes at HEAD.

### Search

```sh
kb search <name> <query>
```

For example:

```sh
kb search myproject "how does authentication work"
```

Use `-n` to control the number of results (default 10), and
`--commit` to search from a specific revision:

```sh
kb search myproject "error handling" -n 5 --commit my-branch
```

Use `.` as the name to search the repository in the current directory:

```sh
kb search . "database connection pooling"
```

### Check status

```sh
kb status
kb status <name>
```

Shows indexed repositories, index file count and size, latest indexed
commit, and embedding count.

## Commands

| Command | Description |
|---------|-------------|
| `kb init` | Set up your Voyage AI API key |
| `kb add <name> <path>` | Register a git repository for indexing |
| `kb index [names...] [-c commit]` | Update the index (delta-aware, restartable) |
| `kb status [name]` | Show indexing status |
| `kb search <name> <query> [-n top] [-c commit]` | Search a repository |
| `kb alias <name> <repos...>` | Create a shorthand for a group of repositories |
| `kb drop <name>` | Delete all index files for a repository |
| `kb start` | Launch the MCP server on stdio |

## MCP Server

`kb start` launches Kamibiki as an MCP server over stdio (JSON-RPC
2.0, newline-delimited). It exposes three tools:

**kb_search** — Search an indexed repository for code relevant to a
query. Parameters:
- `name` (required): repository name, or `.` for the current repo
- `query` (required): natural language or code search query
- `top` (optional): number of results, default 10
- `commit` (optional): git revision to search from (commit hash,
  branch name, tag, `HEAD~1`, etc.). Defaults to HEAD.

**kb_status** — Show indexing status of registered repositories.
Parameters:
- `name` (optional): specific repository name; omit for all repos

**kb_index** — Update the search index for one or more repositories.
Parameters:
- `names` (optional): array of repository names; omit to index all
- `commit` (optional): git revision to index at (commit hash, branch
  name, tag, `HEAD~1`, etc.). Defaults to HEAD.

### Connecting a coding agent

Add Kamibiki to your MCP server configuration. For example, in Cline
(VS Code or CLI), edit your MCP settings:

```json
{
  "mcpServers": {
    "kamibiki": {
      "command": "/path/to/kb",
      "args": ["start"],
      "disabled": false
    }
  }
}
```

For Cline VS Code, this file is at
`~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`.
For Cline CLI, it's at `~/.config/cline-cli/cline_mcp_settings.json`.

Once configured, your coding agent will see `kb_search`, `kb_status`,
and `kb_index` as available tools. Depending on your model, you may
need to prompt it to use the tool the first time — after that it
tends to use it regularly.

## Technical Background

Kamibiki is implemented in Rust and uses gitoxide for high-bandwidth
read-only access to git repositories. Its indexes are flat files
stored in a `.kb` directory at the repository root. As your
repository changes, it writes overlay files. These are periodically
compacted to reclaim space and improve search speed.

Kamibiki skips binary files and extremely large files. Working with
multimodal embedding models (e.g. to embed images) is something we
might consider in future.

### Chunkers and Embedders

As an embeddings-based search engine, Kamibiki is sensitive to the
chunking model, which breaks a file into chunks and possibly reformats
those chunks to add context; and the embedding model, which projects
those chunks into vector space. The set of supported chunking and
embeddings models are hard-coded into the binary and written into
index metadata.

Supported chunkers:

**tsv1** — A tree-sitter based chunker which recursively divides
parse trees for popular languages until a suitable chunk size is
produced. For unsupported languages, it falls back to splitting on
newlines and recursively merging chunks until a suitable size is
reached. The exact chunk size depends on the capabilities of the
embedding model.

Supported models:

**1** = voyage-code-3@2048, binary quantization (~256 bytes/chunk)

### File Format

The Kamibiki index file format is designed to be mmap'ed with the bulk
of the data read-only. This lets us rely on the operating system's
file system cache, page cache and virtual memory manager to be
fast. Each index file is named after a git hash, and contains:

- A version number, currently one byte = 1. These files always use the
  voyage-code-3@2048, binary quantization embedder. (The index is
  insensitive to the chunker used, as long as it produces
  non-overlapping chunks that start at the first byte of the file.)

- The git hash this index was created at. 64 bytes to support
  repositories with SHA-256 hashes, but may just have 20 bytes/160 bit
  SHA-1 hash.

- A parent git hash, if this index should be overlayed on an existing
  index. This way when a repository changes we can produce a small
  index containing just the differences.

- A table of offsets. The offsets are offsets into the git index at
  the relevant revision. The offsets can be positive or negative s16
  numbers. A positive number indicates the following n items from the
  git index, in order, are included in the index. A negative number
  indicates the following -n items from the git index, in order, are
  skipped. A 0 offset indicates the end of the table.

- A table of chunk counts. For each file included (by virtue of being
  covered by a positive count in the index offsets table) a u16 number
  of chunks. Files with more than 65,534 chunks are not supported. (0
  chunks is valid, meaning there are zero chunks. 65,535 is reserved.)

- A table of lengths, per file and per chunk within each file, in
  order. The length is a u16 indicating the length of the chunk in
  bytes. For files with specific encodings, chunkers must break chunks
  on valid character boundaries.

- Padding to align embedding data to a word boundary suitable for
  vector processing. The embedder component is responsible for
  describing this alignment.

- The embedding data. This should be aligned to a suitable width for
  vector processing of the embeddings.

### Query Strategy

To serve a query, Kamibiki:

1. Embeds the query using the relevant embedding model.

2. Uses vector operations to compute cosine distance in embedding
   space. For binary quantized embeddings, this is XOR and
   POPCOUNT. It maintains a max-heap to keep the top N results with the
   smallest embedding distance. Each result is identified by its
   offset in the embeddings table. N is chosen to be close to the
   limits of the reranking model in step 5.

3. Sorts the top N results by offset in the embeddings table.

4. Scans with parallel pointers into the git index, index table, chunk
   count and offset tables. Finds the index entry and byte offsets of
   the top N chunks, and uses git to produce the chunk content.

5. Submits the query and top N chunks to a reranking model (not
   exceeding the context length limitations of the reranking model).

6. Returns the results from the reranker, in order.

When the index has a parent index, the results from steps 2–4 are
refined iteratively walking backwards over the relevant indexes.
Results from the newer indexes are preferred. If an item is deleted in
the revision being searched (not in the index but in git itself),
stale results from old indexes are suppressed before step 5.

### Local Changes

Kamibiki does not search uncommitted changes. In future, we might do
that with an ephemeral in-memory index for uncommitted changes.

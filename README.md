# Kamibiki (KB)

Kamibiki is a contextual engine for agentic coding which indexes git
repositories and documentation locally and provides fast, local
agentic search.

Kamibiki has two modes: A command-line client and a server, including
MCP server.

Commands:

`kb init` sets up a Voyage AI key to generate embeddings. 

`kb add <name> <path>` add an index for the repository at <path>.

`kb index <name> ...` updates the index of the specified
repositories. With the `--compact` flag, existing index files are
collected and flattened into a single comprehensive index file at the
current revision, and old index files are deleted.

`kb status [name]` prints the list of indexed repositories, their indexing
status, and index size.

`kb start`, `kb stop` starts (stops) the Kamibiki server.

`kb search <name> <query>` searches the specified repository. Use the
name `.` as a shorthand for the current repository.

`kb alias <name> <name ...>` sets up an alias to quickly search a set
of related repositories. Aliases are stored in ~/.kb.conf.

Kamibiki stores its index files in a .kb folder at the repository
root. It is recommended that you add this to your .gitignore file. It
also stores pointers to all of your indexed repositories in
~/.kb.conf.

## MCP server

`kb start` launches Kamibiki as an MCP server over stdio (JSON-RPC
2.0, newline-delimited). It exposes two tools:

**kb_search** — Search an indexed repository for code relevant to a
query. Parameters:
- `name` (required): repository name, or `.` for the current repo
- `query` (required): natural language or code search query
- `top` (optional): number of results, default 10

**kb_status** — Show indexing status of registered repositories.
Parameters:
- `name` (optional): specific repository name; omit for all repos

### Connecting a coding agent

Add kamibiki to your MCP server configuration. For example, in Cline
(VS Code or CLI), edit your MCP settings:

```json
{
  "mcpServers": {
    "kamibiki": {
      "command": "kb",
      "args": ["start"],
      "disabled": false
    }
  }
}
```

For Cline VS Code, this file is at
`~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`.
For Cline CLI, it's at `~/.config/cline-cli/cline_mcp_settings.json`.

Once configured, your coding agent will see `kb_search` and
`kb_status` as available tools.

## Technical Background

Kamibiki is implemented in Rust and uses gitoxide for high bandwidth
read-only access to your git repositories. Its indexes are flat
files. As your git repository changes, it writes overlay files. These
are periodically compacted to reclaim space and improve search speed.

Kamibiki skips binary files and extremely large files. Working with
multimodal embedding models, for example to embed images, is something
we might consider in future.

### Chunkers and Embedders

As an embeddings-based search engine, kamibiki is sensitive to the
chunking model, which breaks a file into chunks and possibly reformats
those chunks to add context; and the embedding model, which projects
those chunks into vector space. The set of supported chunking and
embeddings models are hard-coded into the kb server and written into
the index metadata.

Supported chunkers:

tsv1

A tree-sitter based chunker which recursively divides parse trees for
popular languages until a suitable chunk size is produced.

For unsupported languages, this falls back to splitting on newlines
and recursively merging chunks until a suitable chunk size is
produced.

The exact chunk size can depend on the capabilities of the embedding
model.

Supported models:

1 = voyage-context-3@2048, binary quantization (~256 bytes/chunk)

### File format

The Kamibiki index file format is designed to be mmap'ed with the bulk
of the data read-only. This lets us rely on the operating system's
file system cache, page cache and virtual memory manager to be
fast. Each index file is named after a git hash, and contains:

- A version number, currently one byte = 1. These files always use the
  voyage-context-3@2048, binary quantization embedder. (The index is
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
  of chunks. Files with more than 6553*4* chunks are not supported. (0
  chunks are a valid value meaning there are zero chunks. 65535 is
  reserved.)

- A table of lengths, per file and per chunk within each file, in
  order. The length is a u16 indicating the length of the chunk in
  bytes. For files with specific encodings, chunkers must break chunks
  on valid character boundaries. Alternative designs could consider
  re-invoking the chunker to produce chunks including extra context,
  or overlapping chunks, but Kamibiki only supports this scheme of
  simple non-overlapping chunks.

- Padding to align embedding data to a word boundary suitable for
  vector processing. The embedder component is responsible for
  describing this alignment.

- The embedding data. This should be aligned to a suitable width for
  vector processing of the embeddings.

### The query strategy

To serve a query, Kamibiki:

1. Embeds the query using the relevant embedding model.

2. Uses vector operations to compute cosine distance in embedding
   space. For binary quantized embeddings, this is XOR and
   POPCOUNT. It maintains a maxheap to keep the top N results with the
   smallest embedding distance. Each result is identified by its
   offset in the embeddings table. N is chosen to be close to the
   limits of the reranking model in step 5.

3. Sorts the top N results by offset in the embeddings table.

4. Scans with parallel pointers into the git index, index table, chunk
   count and offset tables. Finds the index entry and byte offsets of
   the top N chunks, and uses git to actually produce the chunk
   content.

5. Submits the query and top N chunks to a reranking model (not
   exceeding the context length limitations of the reranking model).

6. Returns the results from the reranker, in order.

When the index has a parent index, the results from steps 2-4 are
refined iteratively walking backwards over the relevant
indexes. Results from the newer indexes are preferred. If an item is
deleted in the revision being searched (not in the index but in git
itself), those stale results from the old index are suppressed before
step 5.

### Local Changes

Kamibiki does not search uncommitted changes. In future, we might do
that with an ephemeral in-memory index for uncommitted changes.

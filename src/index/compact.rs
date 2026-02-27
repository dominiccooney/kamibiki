use std::path::Path;

use anyhow::Result;

/// Compact the index chain for a repository, merging all overlay
/// indexes into a single flat index at the current HEAD.
///
/// `kb_dir` is the `.kb/` directory within the repository root.
///
/// This walks the parent chain from the newest index, merges all
/// entries, re-indexes at the current commit, writes a single root
/// index, and removes the old overlay files.
pub fn compact(_kb_dir: &Path) -> Result<()> {
    // Compaction requires:
    // 1. Finding the latest .kbi file in kb_dir
    // 2. Walking the parent chain via parent_hash in each header
    // 3. Merging overlay entries (newer entries take precedence)
    // 4. Writing a single index with no parent
    // 5. Deleting old .kbi files
    //
    // This depends on git module for resolving paths at each commit
    // and the embed module for re-embedding if needed. Deferred until
    // those modules are integrated.
    todo!("compact: depends on git + embed integration")
}

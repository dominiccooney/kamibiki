use anyhow::{Result, Context};
use gix::bstr::ByteSlice;
use std::path::Path;

/// Open a git repository at the given path.
pub fn open_repo(path: &Path) -> Result<gix::Repository> {
    Ok(gix::discover(path)?)
}

/// Get the HEAD commit hash as a hex string.
pub fn head_commit_hex(repo: &gix::Repository) -> Result<String> {
    let head = repo.head_commit()
        .context("failed to resolve HEAD commit")?;
    Ok(head.id().to_hex().to_string())
}

/// Return the entries in the git index (working tree tracked files)
/// as `(index_position, path)` pairs. The paths are relative to the
/// repository root.
pub fn index_entries(repo: &gix::Repository) -> Result<Vec<(usize, String)>> {
    let index = repo.index()
        .context("failed to read git index")?;
    let entries: Vec<(usize, String)> = index
        .entries()
        .iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            let path = entry.path(&index);
            path.to_str().ok().map(|s| (i, s.to_string()))
        })
        .collect();
    Ok(entries)
}

/// Read a blob from the repository's HEAD tree at the given path.
/// Returns the raw blob content.
pub fn read_blob_at_head(repo: &gix::Repository, path: &str) -> Result<Vec<u8>> {
    let head = repo.head_commit()
        .context("failed to resolve HEAD commit")?;
    let tree = head.tree()
        .context("failed to get HEAD tree")?;
    let entry = tree.lookup_entry_by_path(path)
        .context("failed to look up path in tree")?
        .ok_or_else(|| anyhow::anyhow!("path not found in tree: {}", path))?;
    let object = entry.object()
        .context("failed to read object")?;
    Ok(object.data.to_vec())
}

/// Read a blob from a specific commit (given as hex hash) at the
/// given path.
pub fn read_blob(repo: &gix::Repository, commit_hex: &str, path: &str) -> Result<Vec<u8>> {
    let oid = gix::ObjectId::from_hex(commit_hex.as_bytes())
        .context("invalid commit hash hex")?;
    let commit = repo.find_commit(oid)
        .context("failed to find commit")?;
    let tree = commit.tree()
        .context("failed to get tree from commit")?;
    let entry = tree.lookup_entry_by_path(path)
        .context("failed to look up path in tree")?
        .ok_or_else(|| anyhow::anyhow!("path not found in tree: {}", path))?;
    let object = entry.object()
        .context("failed to read object")?;
    Ok(object.data.to_vec())
}

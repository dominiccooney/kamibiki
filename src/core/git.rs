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

/// Result of comparing two commits: files that were added/modified
/// and files that were deleted.
pub struct TreeDiff {
    /// Files that were added or modified (exist in the new tree).
    pub changed: Vec<String>,
    /// Files that were deleted (exist only in the old tree).
    pub deleted: Vec<String>,
}

/// Find files that changed between two commits using gitoxide's
/// in-process tree diff (no subprocess fork).
pub fn changed_files_between(
    repo: &gix::Repository,
    old_commit_hex: &str,
    new_commit_hex: &str,
) -> Result<TreeDiff> {
    use gix::object::tree::diff::Change;

    let old_oid = gix::ObjectId::from_hex(old_commit_hex.as_bytes())
        .context("invalid old commit hash")?;
    let new_oid = gix::ObjectId::from_hex(new_commit_hex.as_bytes())
        .context("invalid new commit hash")?;

    let old_tree = repo.find_commit(old_oid)
        .context("failed to find old commit")?
        .tree()
        .context("failed to get old tree")?;
    let new_tree = repo.find_commit(new_oid)
        .context("failed to find new commit")?
        .tree()
        .context("failed to get new tree")?;

    let mut changed = Vec::new();
    let mut deleted = Vec::new();
    old_tree
        .changes()
        .context("failed to create tree diff platform")?
        .options(|opts| { opts.track_path(); })
        .for_each_to_obtain_tree(&new_tree, |change| {
            let path = change.location().to_string();
            match change {
                Change::Deletion { .. } => deleted.push(path),
                _ => changed.push(path),
            }
            Ok::<_, std::convert::Infallible>(std::ops::ControlFlow::Continue(()))
        })?;

    Ok(TreeDiff { changed, deleted })
}

/// Get file entries at a specific commit by traversing the commit's
/// tree in-process via gitoxide. Returns `(position, path)` pairs
/// sorted by path, matching the ordering used by `index_entries`.
pub fn entries_at_commit(
    repo: &gix::Repository,
    commit_hex: &str,
) -> Result<Vec<(usize, String)>> {
    let oid = gix::ObjectId::from_hex(commit_hex.as_bytes())
        .context("invalid commit hash")?;
    let commit = repo.find_commit(oid)
        .context("failed to find commit")?;
    let tree = commit.tree()
        .context("failed to get tree from commit")?;

    let mut paths = Vec::new();
    collect_tree_entries(repo, tree.id().into(), "", &mut paths)?;
    // Git index and ls-tree both sort entries by path.
    paths.sort();

    Ok(paths
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i, p))
        .collect())
}

/// Recursively collect all non-tree entry paths from a tree object.
/// Includes blobs, symlinks, and submodule (gitlink) entries to match
/// the enumeration produced by `index_entries()`, which includes all
/// git index entries regardless of type.
fn collect_tree_entries(
    repo: &gix::Repository,
    tree_id: gix::ObjectId,
    prefix: &str,
    paths: &mut Vec<String>,
) -> Result<()> {
    let object = repo.find_object(tree_id)
        .context("failed to find tree object")?;
    let tree = object
        .try_into_tree()
        .map_err(|_| anyhow::anyhow!("object is not a tree"))?;

    for entry_ref in tree.iter() {
        let entry = entry_ref.context("failed to decode tree entry")?;
        let name = entry.filename().to_str()
            .map_err(|_| anyhow::anyhow!("non-UTF-8 filename"))?;
        let full_path = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}/{}", prefix, name)
        };

        if entry.mode().is_tree() {
            collect_tree_entries(repo, entry.object_id(), &full_path, paths)?;
        } else {
            // Include all non-tree entries: blobs, symlinks, gitlinks.
            paths.push(full_path);
        }
    }

    Ok(())
}

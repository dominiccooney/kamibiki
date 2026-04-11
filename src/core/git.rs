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

/// Resolve a revision spec (commit hash, branch name, HEAD, HEAD~1,
/// etc.) to a commit hex string. If `spec` is `None`, resolves HEAD.
///
/// Uses gitoxide's rev-parse, which supports the same revision
/// syntax as git itself (hashes, branch names, remote refs, ~N, ^N,
/// etc.).
pub fn resolve_commit_hex(repo: &gix::Repository, spec: Option<&str>) -> Result<String> {
    match spec {
        None | Some("HEAD") => head_commit_hex(repo),
        Some(spec) => {
            let id = repo.rev_parse_single(spec)
                .with_context(|| format!("failed to resolve revision '{}'", spec))?;
            // Peel to a commit to ensure the spec actually points at
            // a commit (not a tree/blob).
            let commit = id.object()
                .with_context(|| format!("failed to find object for '{}'", spec))?
                .try_into_commit()
                .map_err(|_| anyhow::anyhow!("'{}' does not point to a commit", spec))?;
            Ok(commit.id().to_hex().to_string())
        }
    }
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

/// Describes how a snippet relates to the current file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnippetStaleness {
    /// The snippet matches the file on disk at the same byte offset.
    Current,
    /// The file exists on disk and the snippet text appears in it,
    /// but at a different byte offset — the code may have moved.
    Moved,
    /// The file exists on disk but the snippet text does not appear
    /// in it at all — the code may have been modified.
    Modified,
    /// The file was not found on disk — it may have been renamed or
    /// removed.
    Missing,
}

impl SnippetStaleness {
    /// Returns a short human-readable note, or `None` if the snippet
    /// appears current.
    pub fn note(&self) -> Option<&'static str> {
        match self {
            SnippetStaleness::Current => None,
            SnippetStaleness::Moved => {
                Some("(note: this snippet may have moved — the file on disk differs at this offset)")
            }
            SnippetStaleness::Modified => {
                Some("(note: this snippet may have changed — the file on disk differs from the indexed version)")
            }
            SnippetStaleness::Missing => {
                Some("(note: this file was not found on disk — it may have been renamed or removed)")
            }
        }
    }
}

/// Check whether a snippet from the index still matches the file on
/// disk. Compares the indexed chunk text against the current working
/// tree file at `repo_root / path`.
///
/// `chunk_text` is the snippet content as extracted from the indexed
/// commit. `byte_offset` and `chunk_len` are the byte range within
/// the file where the chunk was originally found.
pub fn check_snippet_staleness(
    repo_root: &Path,
    path: &str,
    byte_offset: usize,
    chunk_len: usize,
    chunk_text: &[u8],
) -> SnippetStaleness {
    let disk_path = repo_root.join(path);
    let disk_content = match std::fs::read(&disk_path) {
        Ok(c) => c,
        Err(_) => return SnippetStaleness::Missing,
    };

    // Check whether the chunk matches at the original byte offset.
    let end = byte_offset + chunk_len;
    if end <= disk_content.len()
        && disk_content[byte_offset..end] == *chunk_text
    {
        return SnippetStaleness::Current;
    }

    // The chunk doesn't match at the original offset. Check whether
    // the text appears anywhere in the file (it may have moved due
    // to edits above it).
    if !chunk_text.is_empty() && find_subsequence(&disk_content, chunk_text).is_some() {
        return SnippetStaleness::Moved;
    }

    SnippetStaleness::Modified
}

/// Find the first occurrence of `needle` in `haystack`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staleness_current_when_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, b"aaabbbccc").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "hello.txt",
            3, // byte_offset
            3, // chunk_len
            b"bbb",
        );
        assert_eq!(result, SnippetStaleness::Current);
        assert!(result.note().is_none());
    }

    #[test]
    fn staleness_moved_when_offset_differs() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hello.txt");
        // Original: "aaabbbccc", chunk "bbb" at offset 3.
        // Disk now has extra bytes prepended, shifting "bbb" to offset 6.
        std::fs::write(&file, b"XXXaaabbbccc").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "hello.txt",
            3, // original offset
            3,
            b"bbb",
        );
        assert_eq!(result, SnippetStaleness::Moved);
        assert!(result.note().is_some());
        assert!(result.note().unwrap().contains("moved"));
    }

    #[test]
    fn staleness_modified_when_text_absent() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, b"completely different content").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "hello.txt",
            0,
            3,
            b"bbb",
        );
        assert_eq!(result, SnippetStaleness::Modified);
        assert!(result.note().is_some());
        assert!(result.note().unwrap().contains("changed"));
    }

    #[test]
    fn staleness_missing_when_file_gone() {
        let dir = tempfile::tempdir().unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "nonexistent.txt",
            0,
            3,
            b"abc",
        );
        assert_eq!(result, SnippetStaleness::Missing);
        assert!(result.note().is_some());
        assert!(result.note().unwrap().contains("not found"));
    }

    #[test]
    fn staleness_current_at_start_of_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, b"hello world").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "f.txt",
            0,
            5,
            b"hello",
        );
        assert_eq!(result, SnippetStaleness::Current);
    }

    #[test]
    fn staleness_modified_when_file_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        // Chunk was at offset 10 len 5, but file is now only 8 bytes.
        std::fs::write(&file, b"short").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "f.txt",
            10,
            5,
            b"xxxxx",
        );
        assert_eq!(result, SnippetStaleness::Modified);
    }

    #[test]
    fn staleness_subdirectory_path() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("src");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("lib.rs"), b"fn main() {}").unwrap();

        let result = check_snippet_staleness(
            dir.path(),
            "src/lib.rs",
            0,
            12,
            b"fn main() {}",
        );
        assert_eq!(result, SnippetStaleness::Current);
    }
}

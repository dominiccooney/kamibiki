//! Index file garbage collection.
//!
//! `kb gc` inspects the `.kbi` files in a repository's `.kb/`
//! directory and removes indexes whose indexed commit is no longer
//! reachable in the git object database.
//!
//! A `.kbi` is scheduled for deletion when (and only when) the
//! commit hash in its header does not resolve in git — the rest of
//! the chain is unaffected, because a dead commit's parent commit
//! might very well still be live (e.g. you're indexing `main` and a
//! tip commit got rewritten, but its ancestors remain).
//!
//! As a consistency check, we also watch for the *anomalous* case
//! where a surviving index's `parent_hash` points at a commit that
//! is about to be deleted. This would only happen if a parent
//! commit somehow became unknown to git while its descendant
//! remained reachable — a weird state that we don't expect in
//! normal operation. When we see it, we warn loudly rather than
//! silently cascade: the user likely wants to look at what
//! happened.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::core::types::{GitHash, MAX_HASH_LEN};
use crate::index::{IndexReader, MmapIndexReader};

/// One entry in a garbage-collection plan.
#[derive(Debug, Clone)]
pub struct GcEntry {
    /// Path to the `.kbi` file.
    pub path: PathBuf,
    /// Hex-encoded commit hash from the index header.
    pub commit_hex: String,
    /// Size of the file in bytes (for reporting).
    pub size: u64,
}

/// An anomaly noticed while planning gc: a kept file whose
/// `parent_hash` points at an index that is about to be deleted.
/// Surfaced as a warning rather than acted upon, because it
/// indicates something unexpected in the repository state.
#[derive(Debug, Clone)]
pub struct ParentOrphanWarning {
    /// The live (kept) `.kbi` file with the suspicious parent.
    pub child_path: PathBuf,
    /// Commit hash of the live file.
    pub child_commit_hex: String,
    /// Commit hash of the parent that is being deleted.
    pub parent_commit_hex: String,
}

/// A plan describing which index files will be deleted and which
/// will be kept. Safe to inspect before calling [`execute`].
#[derive(Debug, Clone, Default)]
pub struct GcPlan {
    /// Files scheduled for deletion, deterministically ordered.
    pub to_delete: Vec<GcEntry>,
    /// Paths of files that will be kept ("live").
    pub kept: Vec<PathBuf>,
    /// Consistency anomalies detected during planning. These do not
    /// affect what is deleted; they are for the caller to report.
    pub warnings: Vec<ParentOrphanWarning>,
}

impl GcPlan {
    /// Total bytes that will be reclaimed.
    pub fn bytes_freed(&self) -> u64 {
        self.to_delete.iter().map(|e| e.size).sum()
    }
}

/// Compute the gc plan for `kb_dir`. `commit_is_known` returns
/// `true` for hex commit hashes that resolve in git.
///
/// Unparseable `.kbi` files are scheduled for deletion (they are
/// already unusable).
pub fn plan_gc<F>(kb_dir: &Path, mut commit_is_known: F) -> Result<GcPlan>
where
    F: FnMut(&str) -> bool,
{
    if !kb_dir.exists() {
        return Ok(GcPlan::default());
    }

    // Enumerate files, reading header metadata once.
    struct Entry {
        path: PathBuf,
        size: u64,
        commit_hex: String,
        parent_hex: String,
    }

    let mut entries: Vec<Entry> = Vec::new();
    let mut plan_entries: Vec<GcEntry> = Vec::new();

    for dirent in
        std::fs::read_dir(kb_dir).with_context(|| format!("reading {}", kb_dir.display()))?
    {
        let dirent = match dirent {
            Ok(d) => d,
            Err(_) => continue,
        };
        let path = dirent.path();
        if !path.extension().is_some_and(|e| e == "kbi") {
            continue;
        }
        let size = dirent.metadata().map(|m| m.len()).unwrap_or(0);

        match MmapIndexReader::open(&path) {
            Ok(reader) => entries.push(Entry {
                path,
                size,
                commit_hex: hash_to_hex(&reader.header().commit_hash),
                parent_hex: hash_to_hex(reader.parent_hash()),
            }),
            Err(_) => {
                // Unreadable → schedule for deletion directly.
                plan_entries.push(GcEntry {
                    path,
                    commit_hex: String::new(),
                    size,
                });
            }
        }
    }

    // Split into kept vs. scheduled, based solely on whether the
    // indexed commit is still known to git.
    let mut scheduled_commits: HashSet<String> = HashSet::new();
    let mut kept: Vec<Entry> = Vec::new();

    for e in entries {
        if e.commit_hex.is_empty() || !commit_is_known(&e.commit_hex) {
            scheduled_commits.insert(e.commit_hex.clone());
            plan_entries.push(GcEntry {
                path: e.path,
                commit_hex: e.commit_hex,
                size: e.size,
            });
        } else {
            kept.push(e);
        }
    }

    // Consistency check: any kept file referring to a deleted
    // parent is an anomaly worth surfacing.
    let mut warnings: Vec<ParentOrphanWarning> = Vec::new();
    for e in &kept {
        if !e.parent_hex.is_empty() && scheduled_commits.contains(&e.parent_hex) {
            warnings.push(ParentOrphanWarning {
                child_path: e.path.clone(),
                child_commit_hex: e.commit_hex.clone(),
                parent_commit_hex: e.parent_hex.clone(),
            });
        }
    }

    plan_entries.sort_by(|a, b| a.path.cmp(&b.path));
    let mut kept_paths: Vec<PathBuf> = kept.into_iter().map(|e| e.path).collect();
    kept_paths.sort();
    warnings.sort_by(|a, b| a.child_path.cmp(&b.child_path));

    Ok(GcPlan {
        to_delete: plan_entries,
        kept: kept_paths,
        warnings,
    })
}

/// Execute a garbage-collection plan: remove every scheduled file.
///
/// Returns the number of bytes reclaimed. Files that have already
/// disappeared between planning and execution are silently skipped.
pub fn execute(plan: &GcPlan) -> Result<u64> {
    let mut freed: u64 = 0;
    for entry in &plan.to_delete {
        match std::fs::remove_file(&entry.path) {
            Ok(()) => freed += entry.size,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "failed to remove {}: {}",
                    entry.path.display(),
                    e
                ));
            }
        }
    }
    Ok(freed)
}

/// Extract a hex commit hash from a [`GitHash`] (zero-padded ASCII).
fn hash_to_hex(hash: &GitHash) -> String {
    let end = hash.iter().position(|&b| b == 0).unwrap_or(MAX_HASH_LEN);
    String::from_utf8_lossy(&hash[..end]).into_owned()
}

/// Convenience wrapper: check whether `commit_hex` resolves to an
/// existing commit in the given repository.
pub fn commit_is_known(repo: &gix::Repository, commit_hex: &str) -> bool {
    if commit_hex.is_empty() {
        return false;
    }
    let oid = match gix::ObjectId::from_hex(commit_hex.as_bytes()) {
        Ok(o) => o,
        Err(_) => return false,
    };
    repo.find_commit(oid).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BinaryEmbedding, EMBEDDING_BYTES, IndexHeader};
    use crate::index::format::write_index;
    use crate::index::{ChunkEntry, FileEntry};

    fn make_git_hash(hex: &str) -> GitHash {
        let mut hash: GitHash = [0; MAX_HASH_LEN];
        let bytes = hex.as_bytes();
        hash[..bytes.len().min(MAX_HASH_LEN)]
            .copy_from_slice(&bytes[..bytes.len().min(MAX_HASH_LEN)]);
        hash
    }

    fn write_kbi(kb_dir: &Path, name: &str, commit: &str, parent: &str) {
        let header = IndexHeader {
            version: 1,
            commit_hash: make_git_hash(commit),
            parent_hash: make_git_hash(parent),
        };
        let entries = vec![FileEntry {
            git_index_position: 0,
            chunks: vec![ChunkEntry {
                byte_offset: 0,
                chunk_len: 1,
                embedding: [0u8; EMBEDDING_BYTES] as BinaryEmbedding,
            }],
        }];
        write_index(&kb_dir.join(format!("{}.kbi", name)), &header, &entries).unwrap();
    }

    #[test]
    fn plan_empty_directory_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let plan = plan_gc(dir.path(), |_| true).unwrap();
        assert!(plan.to_delete.is_empty());
        assert!(plan.kept.is_empty());
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn plan_missing_directory_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let plan = plan_gc(&dir.path().join("nonexistent"), |_| true).unwrap();
        assert!(plan.to_delete.is_empty());
    }

    #[test]
    fn plan_removes_unknown_commit_only() {
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        write_kbi(dir.path(), "bbbb", "bbbb", "");

        let plan = plan_gc(dir.path(), |c| c == "aaaa").unwrap();
        assert_eq!(plan.to_delete.len(), 1);
        assert_eq!(plan.to_delete[0].commit_hex, "bbbb");
        assert_eq!(plan.kept.len(), 1);
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn plan_does_not_cascade_to_children() {
        // aaaa is dead (unknown). bbbb has aaaa as parent but is
        // itself known to git. We delete only aaaa; bbbb survives
        // even though its parent is gone — and we emit a warning.
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        write_kbi(dir.path(), "bbbb", "bbbb", "aaaa");

        let plan = plan_gc(dir.path(), |c| c == "bbbb").unwrap();
        assert_eq!(plan.to_delete.len(), 1);
        assert_eq!(plan.to_delete[0].commit_hex, "aaaa");
        assert_eq!(plan.kept.len(), 1);

        assert_eq!(plan.warnings.len(), 1);
        assert_eq!(plan.warnings[0].child_commit_hex, "bbbb");
        assert_eq!(plan.warnings[0].parent_commit_hex, "aaaa");
    }

    #[test]
    fn plan_keeps_chain_when_parent_and_child_are_known() {
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        write_kbi(dir.path(), "bbbb", "bbbb", "aaaa");

        let plan = plan_gc(dir.path(), |_| true).unwrap();
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.kept.len(), 2);
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn plan_deletes_dead_parent_without_touching_dead_child() {
        // aaaa (unknown root) ← bbbb (also unknown): both are
        // scheduled for deletion. No warning: the child is also
        // dying, so there's no live file with a dangling parent.
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        write_kbi(dir.path(), "bbbb", "bbbb", "aaaa");

        let plan = plan_gc(dir.path(), |_| false).unwrap();
        assert_eq!(plan.to_delete.len(), 2);
        assert!(plan.kept.is_empty());
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn execute_removes_files_and_reports_bytes() {
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        write_kbi(dir.path(), "bbbb", "bbbb", "");

        let plan = plan_gc(dir.path(), |c| c == "aaaa").unwrap();
        let expected = plan.bytes_freed();
        let freed = execute(&plan).unwrap();
        assert_eq!(freed, expected);
        assert!(freed > 0);

        assert!(dir.path().join("aaaa.kbi").exists());
        assert!(!dir.path().join("bbbb.kbi").exists());
    }

    #[test]
    fn execute_is_idempotent_if_file_already_gone() {
        let dir = tempfile::tempdir().unwrap();
        write_kbi(dir.path(), "aaaa", "aaaa", "");
        let plan = plan_gc(dir.path(), |_| false).unwrap();

        std::fs::remove_file(dir.path().join("aaaa.kbi")).unwrap();
        let freed = execute(&plan).unwrap();
        assert_eq!(freed, 0);
    }
}

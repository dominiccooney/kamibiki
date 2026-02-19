use anyhow::Result;
use std::path::Path;

/// Open a git repository at the given path.
pub fn open_repo(path: &Path) -> Result<gix::Repository> {
    Ok(gix::discover(path)?)
}

/// Iterate over files in the git index at HEAD, yielding
/// (index_position, path) pairs.
pub fn index_entries(_repo: &gix::Repository)
    -> Result<Vec<(usize, String)>>
{
    todo!()
}

/// Read a blob from the repository at a given revision and path.
pub fn read_blob(_repo: &gix::Repository, _commit_hash: &[u8], _path: &str)
    -> Result<Vec<u8>>
{
    todo!()
}

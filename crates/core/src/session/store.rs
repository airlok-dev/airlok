//! On-disk sessions: one JSON file per conversation under
//! `<data dir>/sessions/<cwd hash>/<created>-<id>.json`. The file holds the
//! plaintext history and the value behind every placeholder, so it is
//! written mode 0600 inside a 0700 directory. Provider keys are never
//! written: redact-only entries are stored with an empty value.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::warn;

use super::{fnv1a, Session};

pub struct SessionStore {
    root: PathBuf,
}

/// One line of `airlok sessions`.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub id: String,
    pub path: PathBuf,
    pub created_at: String,
    pub updated_at: String,
    pub model: String,
    pub turns: usize,
    pub first_prompt: Option<String>,
}

impl SessionStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `$XDG_DATA_HOME/airlok`, else `~/.local/share/airlok`.
    pub fn default_root() -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
            return Some(PathBuf::from(dir).join("airlok"));
        }
        std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(|home| PathBuf::from(home).join(".local/share/airlok"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Sessions started in `cwd` share one directory, named by a stable hash
    /// of the path so two checkouts never mix.
    pub fn dir_for(&self, cwd: &Path) -> PathBuf {
        let hash = fnv1a(cwd.to_string_lossy().as_bytes());
        self.root.join("sessions").join(format!("{hash:016x}"))
    }

    /// Writes the session atomically (temp file, then rename) with private
    /// permissions. The file name is fixed at creation, so repeated saves
    /// overwrite the same file.
    pub fn save(&self, session: &Session) -> io::Result<PathBuf> {
        let dir = self.dir_for(&session.cwd);
        create_private_dirs(&self.root, &dir)?;
        let name = file_name(session);
        let path = dir.join(&name);
        let tmp = dir.join(format!(".{name}.tmp"));
        let body = serde_json::to_vec_pretty(&session.for_disk()).map_err(io::Error::other)?;
        write_private(&tmp, &body)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// Sessions for `cwd`, most recently updated first. Unreadable files
    /// are skipped with a warning.
    pub fn list(&self, cwd: &Path) -> io::Result<Vec<Summary>> {
        let mut summaries: Vec<Summary> = self
            .session_files(&self.dir_for(cwd))?
            .into_iter()
            .filter_map(|path| match read_session(&path) {
                Ok(session) => Some(Summary {
                    id: session.id.clone(),
                    path,
                    created_at: session.created_at.clone(),
                    updated_at: session.updated_at.clone(),
                    model: session.model.clone(),
                    turns: session.turns(),
                    first_prompt: session.first_prompt().map(str::to_string),
                }),
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "skipping unreadable session");
                    None
                }
            })
            .collect();
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    /// The session with `id` (a unique prefix will do), or the most recently
    /// updated one when `id` is `None`.
    pub fn load(&self, cwd: &Path, id: Option<&str>) -> io::Result<Option<Session>> {
        let Some(summary) = self.find(cwd, id)? else {
            return Ok(None);
        };
        read_session(&summary.path).map(Some)
    }

    /// Deletes one session. Returns false when there is no such session.
    pub fn remove(&self, cwd: &Path, id: &str) -> io::Result<bool> {
        match self.find(cwd, Some(id))? {
            Some(summary) => fs::remove_file(summary.path).map(|_| true),
            None => Ok(false),
        }
    }

    /// Deletes sessions from every directory whose last update is older
    /// than `age`. Returns the deleted paths.
    pub fn clean(&self, age: Duration) -> io::Result<Vec<PathBuf>> {
        let cutoff = time::OffsetDateTime::now_utc() - age;
        let sessions_dir = self.root.join("sessions");
        let mut removed = Vec::new();
        for dir in list_dirs(&sessions_dir)? {
            for path in self.session_files(&dir)? {
                let Ok(session) = read_session(&path) else {
                    continue;
                };
                let updated = time::OffsetDateTime::parse(
                    &session.updated_at,
                    &time::format_description::well_known::Rfc3339,
                );
                if matches!(updated, Ok(at) if at < cutoff) {
                    fs::remove_file(&path)?;
                    removed.push(path);
                }
            }
            if fs::read_dir(&dir)?.next().is_none() {
                let _ = fs::remove_dir(&dir);
            }
        }
        Ok(removed)
    }

    fn find(&self, cwd: &Path, id: Option<&str>) -> io::Result<Option<Summary>> {
        let summaries = self.list(cwd)?;
        let Some(wanted) = id else {
            return Ok(summaries.into_iter().next());
        };
        let mut matches = summaries.into_iter().filter(|s| s.id.starts_with(wanted));
        let first = matches.next();
        if matches.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("session id {wanted} is ambiguous"),
            ));
        }
        Ok(first)
    }

    fn session_files(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        match fs::read_dir(dir) {
            Ok(entries) => Ok(entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
                .filter(|p| {
                    !p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                })
                .collect()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }
}

fn list_dirs(path: &Path) -> io::Result<Vec<PathBuf>> {
    match fs::read_dir(path) {
        Ok(entries) => Ok(entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

fn read_session(path: &Path) -> io::Result<Session> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}

/// `20260911T120000Z-1a2b3c4d.json`: sorts by creation time, findable by id.
fn file_name(session: &Session) -> String {
    let stamp: String = session
        .created_at
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    format!("{stamp}-{}.json", session.id)
}

fn create_private_dirs(root: &Path, leaf: &Path) -> io::Result<()> {
    fs::create_dir_all(leaf)?;
    // Tighten every directory from the root down; create_dir_all uses the
    // umask, which is usually 0755.
    let mut dir = leaf;
    loop {
        set_mode(dir, 0o700)?;
        if dir == root {
            break;
        }
        match dir.parent() {
            Some(parent) if parent.starts_with(root) => dir = parent,
            _ => break,
        }
    }
    Ok(())
}

fn write_private(path: &Path, body: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(body)?;
    file.sync_all()?;
    // The mode on open only applies to a new file; an existing temp file
    // from a crashed run keeps whatever it had.
    set_mode(path, 0o600)
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

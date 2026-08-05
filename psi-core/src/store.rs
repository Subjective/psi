//! Persistence: one append-only JSONL file per session (docs/design.md,
//! "Persistence: append-only JSONL per session").
//!
//! A file is the replay of the only two mutations a session tree has: an item
//! is appended, or `head` moves. Appending an item always moves `head` onto
//! it, so an item record carries its own head move and only `set_head` needs a
//! record of its own. The last record that touches `head` wins, which is how a
//! restart recovers where the user left off without ever rewriting a line.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::item::{Item, ItemId};
use crate::session::{SessionId, SessionMeta, SessionSnapshot};

const EXTENSION: &str = "jsonl";

/// How many same-millisecond ids to try before giving up. Reaching it means
/// something other than a clock collision is wrong with the directory.
const MAX_ID_COLLISIONS: u32 = 64;

/// One line of a session log. The tag is spelled out so a log stays greppable:
/// an item line is the item's own fields alongside `"record":"item"`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Record {
    Meta(SessionMeta),
    Item(Item),
    Head { head: Option<ItemId> },
}

/// The append end of one session's log.
#[derive(Debug)]
pub struct SessionLog {
    file: File,
}

impl SessionLog {
    /// Appends one record. The line is built in memory and written with a
    /// single call to a file opened for append, so a torn write can only leave
    /// an unterminated last line, never an interleaved one. There is no
    /// `fsync`: everything written survives a process crash, and an OS crash
    /// may lose the tail, which is what `SessionStore::load` repairs.
    fn write(&mut self, record: &Record) -> io::Result<()> {
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)
    }

    pub fn append_item(&mut self, item: &Item) -> io::Result<()> {
        self.write(&Record::Item(item.clone()))
    }

    pub fn set_head(&mut self, head: Option<ItemId>) -> io::Result<()> {
        self.write(&Record::Head { head })
    }
}

/// The sessions directory: one `<session id>.jsonl` file per session.
#[derive(Debug)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// Creates the directory if it is missing, so the first run of Psi works
    /// against a fresh home.
    pub fn new(dir: PathBuf) -> io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Starts a durable session and writes its header. The id is the creation
    /// time, disambiguated when a file of that name already exists;
    /// `create_new` makes the check atomic, so two Psi processes starting in
    /// the same millisecond cannot claim one file.
    pub fn create(&self, created_at_ms: u64) -> io::Result<(SessionMeta, SessionLog)> {
        for suffix in 0..MAX_ID_COLLISIONS {
            let id = match suffix {
                0 => SessionId(format!("s{created_at_ms}")),
                n => SessionId(format!("s{created_at_ms}-{n}")),
            };
            let path = self.dir.join(format!("{}.{EXTENSION}", id.0));
            let file = match OpenOptions::new().create_new(true).append(true).open(&path) {
                Ok(file) => file,
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(err),
            };
            let meta = SessionMeta { id, created_at_ms };
            let mut log = SessionLog { file };
            log.write(&Record::Meta(meta.clone()))?;
            return Ok((meta, log));
        }
        Err(io::Error::other("no free session id"))
    }

    /// Every session on disk, newest first, so a client can offer "continue
    /// the most recent". Only headers are read: listing must not pay for the
    /// size of every log. A file whose header never landed is not yet a
    /// session and is skipped, as is anything unreadable — a listing has no
    /// way to report an error, and one bad file must not hide the rest.
    pub fn list(&self) -> Vec<SessionMeta> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut sessions: Vec<SessionMeta> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == EXTENSION))
            .filter_map(|path| header(&path))
            .collect();
        // Ids break ties so a listing is deterministic across runs.
        sessions.sort_by(|a, b| {
            b.created_at_ms
                .cmp(&a.created_at_ms)
                .then_with(|| b.id.cmp(&a.id))
        });
        sessions
    }

    /// Restores a session's tree and head and reopens its log for appending.
    pub fn load(&self, id: &SessionId) -> io::Result<(SessionSnapshot, SessionLog)> {
        let path = self.path(id)?;
        let (snapshot, file) = replay(&path)?;
        Ok((snapshot, SessionLog { file }))
    }

    /// Session ids become filenames and `load_session` takes its id from the
    /// client, so anything that is not a plain name is refused rather than
    /// resolved.
    fn path(&self, id: &SessionId) -> io::Result<PathBuf> {
        let name = &id.0;
        let plain = !name.is_empty()
            && !name.starts_with('.')
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !plain {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid session id: {name}"),
            ));
        }
        Ok(self.dir.join(format!("{name}.{EXTENSION}")))
    }
}

/// Reads a session's header line. A header with no terminating newline was
/// never committed, so it reads as no session at all — the same rule `replay`
/// applies, which keeps listing and loading agreed on what exists.
fn header(path: &Path) -> Option<SessionMeta> {
    let mut line = Vec::new();
    BufReader::new(File::open(path).ok()?)
        .read_until(b'\n', &mut line)
        .ok()?;
    if !line.ends_with(b"\n") {
        return None;
    }
    match serde_json::from_slice(&line).ok()? {
        Record::Meta(meta) => Some(meta),
        _ => None,
    }
}

/// Replays a log into a snapshot and repairs the file if it is damaged.
///
/// A log is valid up to its first defect: a final line with no terminating
/// newline (a crash mid-write), a line that is not a record, or a record that
/// contradicts the tree — a second header, a repeated item id, a parent or a
/// head that no earlier line defined. Everything from the defect on is
/// dropped and the file is truncated to that point, so the prefix that a
/// reader accepted is exactly what the next append extends. Truncating rather
/// than skipping is what keeps the rule simple: a log is a prefix, and a
/// record that survives can never reference one that did not.
///
/// The returned file is open for append, so writes go past the repaired end
/// whatever the read left the cursor at.
fn replay(path: &Path) -> io::Result<(SessionSnapshot, File)> {
    let mut file = OpenOptions::new().read(true).append(true).open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut meta: Option<SessionMeta> = None;
    let mut items: Vec<Item> = Vec::new();
    let mut seen: HashSet<ItemId> = HashSet::new();
    let mut head: Option<ItemId> = None;
    // Bytes of the valid prefix; the file is truncated here.
    let mut valid_len = 0usize;

    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        if !line.ends_with(b"\n") {
            break;
        }
        // Invalid UTF-8 is a torn write like any other, so it fails to parse
        // rather than failing the load.
        let record = std::str::from_utf8(line)
            .ok()
            .and_then(|line| serde_json::from_str::<Record>(line).ok());
        let Some(record) = record else { break };
        match record {
            Record::Meta(recorded) => {
                if meta.is_some() {
                    break;
                }
                meta = Some(recorded);
            }
            Record::Item(item) => {
                // The parent is checked before the id is claimed, so an item
                // that names itself as its parent is a defect like any other.
                if meta.is_none()
                    || item.parent_id.is_some_and(|parent| !seen.contains(&parent))
                    || !seen.insert(item.id)
                {
                    break;
                }
                head = Some(item.id);
                items.push(item);
            }
            Record::Head { head: target } => {
                if meta.is_none() || target.is_some_and(|id| !seen.contains(&id)) {
                    break;
                }
                head = target;
            }
        }
        valid_len += line.len();
    }

    let Some(meta) = meta else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("session log has no header: {}", path.display()),
        ));
    };
    if valid_len != bytes.len() {
        file.set_len(valid_len as u64)?;
    }
    Ok((SessionSnapshot { meta, items, head }, file))
}

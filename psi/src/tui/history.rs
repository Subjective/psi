//! Prompt history that outlives a run: one file beside the session logs.
//!
//! It lives in the sessions directory because that is already the one place
//! Psi is allowed to write, and it is named `history` with no extension so
//! `SessionStore::list`, which only reads `*.jsonl`, never sees it as a
//! session. Each line is one JSON string, so a prompt with line breaks in it
//! survives as one entry.
//!
//! History is a convenience, never a correctness concern: every failure to
//! read or write is swallowed, because a prompt that could not be recorded
//! must not interrupt the session it was typed into.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

/// How many prompts a new run loads. Enough to walk back through a working
/// session, few enough that the walk stays a walk.
const LOADED: usize = 100;

pub struct History {
    path: PathBuf,
    /// The last prompt on file, so a prompt submitted twice in a row is not
    /// appended twice.
    last: Option<String>,
    /// True when the file ends mid-line — a torn write. The next append then
    /// starts with its own newline, so the torn tail stays one skipped line
    /// instead of swallowing the new entry with it.
    unterminated: bool,
}

impl History {
    /// Opens the history in `dir` and reads the prompts a new composer starts
    /// with, oldest last.
    pub fn open(dir: &Path) -> (Self, Vec<String>) {
        let path = dir.join("history");
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let mut prompts: Vec<String> = text
            .lines()
            // A line that does not parse is a torn write, not an entry.
            .filter_map(|line| serde_json::from_str::<String>(line).ok())
            .collect();
        let prompts = prompts.split_off(prompts.len().saturating_sub(LOADED));
        let last = prompts.last().cloned();
        let unterminated = !text.is_empty() && !text.ends_with('\n');
        (
            Self {
                path,
                last,
                unterminated,
            },
            prompts,
        )
    }

    pub fn append(&mut self, prompt: &str) {
        if self.last.as_deref() == Some(prompt) {
            return;
        }
        self.last = Some(prompt.to_string());
        let Ok(line) = serde_json::to_string(prompt) else {
            return;
        };
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let lead = if self.unterminated { "\n" } else { "" };
            let _ = writeln!(file, "{lead}{line}");
            self.unterminated = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_survive_a_restart_with_their_line_breaks() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, prompts) = History::open(dir.path());
        assert!(prompts.is_empty());
        history.append("first");
        history.append("second\nline");

        let (_, prompts) = History::open(dir.path());
        assert_eq!(prompts, ["first", "second\nline"]);
    }

    #[test]
    fn a_prompt_repeated_in_a_row_is_recorded_once() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, _) = History::open(dir.path());
        for prompt in ["go", "go", "again", "go"] {
            history.append(prompt);
        }
        // The check spans runs too: reopening remembers the last entry.
        let (mut history, prompts) = History::open(dir.path());
        assert_eq!(prompts, ["go", "again", "go"]);
        history.append("go");
        let (_, prompts) = History::open(dir.path());
        assert_eq!(prompts.len(), 3);
    }

    #[test]
    fn only_the_last_hundred_prompts_load() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, _) = History::open(dir.path());
        for n in 0..LOADED + 10 {
            history.append(&format!("prompt {n}"));
        }
        let (_, prompts) = History::open(dir.path());
        assert_eq!(prompts.len(), LOADED);
        assert_eq!(prompts[0], "prompt 10");
        assert_eq!(prompts[LOADED - 1], format!("prompt {}", LOADED + 9));
    }

    #[test]
    fn a_torn_line_is_skipped_rather_than_read_as_a_prompt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("history"),
            "\"one\"\n{\"not\": \"a prompt\"}\n\"two",
        )
        .unwrap();
        let (_, prompts) = History::open(dir.path());
        assert_eq!(prompts, ["one"]);
    }

    /// Appending after a crash mid-write must not glue the new entry onto the
    /// torn tail, which would lose them both.
    #[test]
    fn appending_after_a_torn_tail_starts_its_own_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("history"), "\"one\"\n\"tw").unwrap();
        let (mut history, prompts) = History::open(dir.path());
        assert_eq!(prompts, ["one"]);
        history.append("three");
        let (_, prompts) = History::open(dir.path());
        assert_eq!(prompts, ["one", "three"]);
    }

    /// The store lists `*.jsonl`; the history file must not look like one.
    #[test]
    fn the_history_file_is_not_a_session() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, _) = History::open(dir.path());
        history.append("go");
        let store = psi_core::store::SessionStore::new(dir.path().to_path_buf()).unwrap();
        assert!(store.list().is_empty());
    }
}

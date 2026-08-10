//! The `@` picker's two pure pieces: a walk of the workspace, and a
//! subsequence scorer that ranks its entries against what the user typed.
//!
//! Both are here rather than in the harness because completing a path is the
//! client's business, not the session's: nothing the picker does crosses the
//! protocol, and nothing it reads becomes durable state.

use std::collections::VecDeque;
use std::path::Path;

/// Directories the walk never descends into. The same rule the harness's own
/// tools apply: `.git` is large, binary, and never what a prompt means.
const SKIPPED_DIRS: [&str; 1] = [".git"];

/// How many entries one walk collects. The walk is breadth-first, so a
/// workspace larger than this loses its deepest paths rather than everything
/// after the first big directory.
const MAX_ENTRIES: usize = 5000;

/// How many matches the picker offers. The live region shows a handful; the
/// rest exist only to be scrolled past, so they are not ranked into a list.
const MAX_MATCHES: usize = 20;

/// A character matched at the start of a path segment. Segments are what a
/// user types from memory, so `tui/comp` should beat a scatter of the same
/// characters inside one long name.
const SEGMENT_BONUS: i32 = 10;

/// A character matched directly after the previous one. Worth more than a
/// segment start, so a query typed as one word finds the name it spells rather
/// than a path whose separators happen to line up with its letters.
const RUN_BONUS: i32 = 12;

/// Any matched character at all, so a longer query outscores a shorter one.
const MATCH: i32 = 1;

/// Every file and directory under `root`, workspace-relative, shallowest
/// first. Directories carry a trailing `/` so the list says which is which and
/// an inserted directory path reads as one.
///
/// Symlinks are skipped: one can point outside the workspace or back up the
/// tree, and the walk stays on real paths.
pub fn walk(root: &Path) -> Vec<String> {
    let mut entries = Vec::new();
    let mut queue = VecDeque::from([root.to_path_buf()]);
    while let Some(dir) = queue.pop_front() {
        let Ok(read) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut children: Vec<_> = read.flatten().collect();
        children.sort_by_key(|child| child.file_name());
        for child in children {
            if entries.len() >= MAX_ENTRIES {
                return entries;
            }
            let name = child.file_name();
            if SKIPPED_DIRS.contains(&name.to_string_lossy().as_ref()) {
                continue;
            }
            let Ok(kind) = child.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            let path = child.path();
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            let mut shown = relative.to_string_lossy().to_string();
            if kind.is_dir() {
                shown.push('/');
                queue.push_back(path);
            }
            entries.push(shown);
        }
    }
    entries
}

/// The entries `query` selects, best first. Ties break toward the shorter
/// path, then alphabetically, so a listing is stable across keystrokes.
pub fn rank(entries: &[String], query: &str) -> Vec<String> {
    let mut scored: Vec<(i32, &String)> = entries
        .iter()
        .filter_map(|entry| score(query, entry).map(|score| (score, entry)))
        .collect();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(b.1))
    });
    scored
        .into_iter()
        .take(MAX_MATCHES)
        .map(|(_, entry)| entry.clone())
        .collect()
}

/// Scores `path` against `query`, or `None` when the query's characters do not
/// all appear in it in order.
///
/// The match is case-insensitive until the query contains an uppercase
/// character, which makes the whole comparison exact — a query typed in
/// lowercase means "I do not care", and one that shifts a key means the
/// opposite.
///
/// Scoring is a small dynamic program rather than a greedy scan, because a
/// greedy scan takes the first place each character fits and so misses the run
/// further along that a user aiming at a filename actually meant.
pub fn score(query: &str, path: &str) -> Option<i32> {
    let sensitive = query.chars().any(char::is_uppercase);
    let fold = |text: &str| -> Vec<char> {
        if sensitive {
            text.chars().collect()
        } else {
            text.to_lowercase().chars().collect()
        }
    };
    let needle = fold(query);
    let haystack = fold(path);
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }

    // `ends[j]` is the best score for a match of the query so far whose last
    // character landed on `haystack[j]`.
    let mut ends: Vec<Option<i32>> = vec![None; haystack.len()];
    for (index, wanted) in needle.iter().enumerate() {
        let mut next: Vec<Option<i32>> = vec![None; haystack.len()];
        // The best place the previous character could have ended, strictly
        // before `j`.
        let mut before: Option<i32> = None;
        for j in 0..haystack.len() {
            if j > 0 {
                before = before.max(ends[j - 1]);
            }
            if haystack[j] != *wanted {
                continue;
            }
            // The query's first character may start anywhere; later ones must
            // follow a match of the one before.
            let after_gap = if index == 0 { Some(0) } else { before };
            let after_run = match j {
                0 => None,
                j => ends[j - 1].map(|score| score + RUN_BONUS),
            };
            let bonus = MATCH
                + if segment_start(&haystack, j) {
                    SEGMENT_BONUS
                } else {
                    0
                };
            next[j] = after_gap.max(after_run).map(|score| score + bonus);
        }
        ends = next;
    }
    ends.into_iter().flatten().max()
}

/// Whether `at` begins a path segment: the start of the path, or just after
/// one of the separators paths are built from.
fn segment_start(path: &[char], at: usize) -> bool {
    at == 0 || matches!(path[at - 1], '/' | '_' | '-' | '.')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best(query: &str, paths: &[&str]) -> Vec<String> {
        let entries: Vec<String> = paths.iter().map(|path| path.to_string()).collect();
        rank(&entries, query)
    }

    #[test]
    fn a_query_must_appear_in_order() {
        assert!(score("tui", "psi/src/tui/app.rs").is_some());
        assert!(score("iut", "psi/src/tui/app.rs").is_none());
        assert!(score("xyz", "psi/src/tui/app.rs").is_none());
        // An empty query matches everything, which is the list a bare `@`
        // opens on.
        assert_eq!(score("", "anything"), Some(0));
    }

    #[test]
    fn case_matters_only_once_the_query_shifts_a_key() {
        assert!(score("readme", "README.md").is_some());
        assert!(score("READ", "README.md").is_some());
        assert!(score("Readme", "README.md").is_none());
    }

    #[test]
    fn a_consecutive_run_beats_the_same_characters_scattered() {
        let run = score("comp", "composer.rs").unwrap();
        let scattered = score("comp", "c_o_m_p.rs").unwrap();
        assert!(run > scattered, "{run} vs {scattered}");
    }

    #[test]
    fn a_segment_start_beats_the_middle_of_a_name() {
        let start = score("app", "src/app.rs").unwrap();
        let middle = score("app", "src/wrapper.rs").unwrap();
        assert!(start > middle, "{start} vs {middle}");
    }

    #[test]
    fn ties_break_toward_the_shorter_path() {
        assert_eq!(
            best("app", &["psi/src/deep/app.rs", "app.rs"]),
            ["app.rs", "psi/src/deep/app.rs"]
        );
    }

    #[test]
    fn ranking_puts_the_path_a_query_aims_at_first() {
        let paths = [
            "psi/src/tui/app.rs",
            "psi/src/tui/composer.rs",
            "psi/src/tui/draw.rs",
            "psi-core/src/tools/mod.rs",
            "docs/design.md",
        ];
        assert_eq!(best("tuicomp", &paths)[0], "psi/src/tui/composer.rs");
        assert_eq!(best("dsgn", &paths)[0], "docs/design.md");
        assert_eq!(best("draw", &paths)[0], "psi/src/tui/draw.rs");
    }

    #[test]
    fn the_walk_is_relative_shallowest_first_and_skips_git_and_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path();
        std::fs::create_dir_all(path.join("src/tui")).unwrap();
        std::fs::create_dir_all(path.join(".git/objects")).unwrap();
        std::fs::write(path.join("README.md"), "").unwrap();
        std::fs::write(path.join("src/tui/app.rs"), "").unwrap();
        std::fs::write(path.join(".git/objects/blob"), "").unwrap();
        std::os::unix::fs::symlink(path.join("README.md"), path.join("link.md")).unwrap();

        assert_eq!(
            walk(path),
            ["README.md", "src/", "src/tui/", "src/tui/app.rs"]
        );
    }
}

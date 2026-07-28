use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::Deserialize;
use serde_json::json;

use super::path::resolve_in_root;
use super::{SKIPPED_DIRS, blocking, failure, parse_args, success, truncate};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolSpec};

const MAX_MATCHES: usize = 100;
const MAX_LINE_CHARS: usize = 200;
const MAX_BYTES: usize = 16 * 1024;
/// Files larger than this are skipped: they are generated or binary far more
/// often than they are what the model is looking for.
const MAX_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
}

pub struct Search {
    root: PathBuf,
}

impl Search {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for Search {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search".to_string(),
            description: format!(
                "Search the workspace for a regular expression, one result per matching \
                 line, formatted as `path:line: text`. Skips `.git` and files that are not \
                 UTF-8 text. Stops after {MAX_MATCHES} matches."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust regular expression to match against each line."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search, relative to the workspace \
                             root. Defaults to the root itself."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::ReadOnly
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        let root = self.root.clone();
        blocking(move || {
            let args: Args = match parse_args(&invocation.arguments) {
                Ok(args) => args,
                Err(output) => return output,
            };
            let pattern = match Regex::new(&args.pattern) {
                Ok(pattern) => pattern,
                Err(err) => return failure(format!("invalid pattern: {err}")),
            };
            let requested = args.path.unwrap_or_else(|| ".".to_string());
            let base = match resolve_in_root(&root, &requested) {
                Ok(path) => path,
                Err(message) => return failure(message),
            };

            let mut matches = Vec::new();
            search(&root, &base, &pattern, &mut matches);
            if matches.is_empty() {
                return success(format!("no matches for {}", args.pattern), false);
            }
            let capped = matches.len() > MAX_MATCHES;
            matches.truncate(MAX_MATCHES);
            if capped {
                matches.push(format!("[stopped after {MAX_MATCHES} matches]"));
            }
            let (content, truncated) = truncate(matches.join("\n"), MAX_BYTES);
            success(content, truncated || capped)
        })
    }
}

/// Collects one match per line as `path:line: text`, with paths shown relative
/// to the workspace root so they can be handed straight back to `read_file`.
/// Unreadable files are skipped rather than failing the whole search.
fn search(root: &Path, path: &Path, pattern: &Regex, matches: &mut Vec<String>) {
    if matches.len() > MAX_MATCHES {
        return;
    }
    if path.is_dir() {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
            if SKIPPED_DIRS.contains(&name.to_string_lossy().as_ref()) {
                continue;
            }
            search(root, &entry.path(), pattern, matches);
        }
        return;
    }
    if fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or(u64::MAX)
        > MAX_FILE_BYTES
    {
        return;
    }
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let shown = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
    for (index, line) in text.lines().enumerate() {
        if !pattern.is_match(line) {
            continue;
        }
        let line: String = line.chars().take(MAX_LINE_CHARS).collect();
        matches.push(format!("{shown}:{}: {line}", index + 1));
        if matches.len() > MAX_MATCHES {
            return;
        }
    }
}

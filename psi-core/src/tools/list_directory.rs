use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::json;

use super::path::resolve_in_root;
use super::{SKIPPED_DIRS, blocking, failure, parse_args, success, truncate};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolSpec};

const MAX_BYTES: usize = 16 * 1024;
const MAX_DEPTH: usize = 8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    depth: Option<usize>,
}

pub struct ListDirectory {
    root: PathBuf,
}

impl ListDirectory {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for ListDirectory {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_directory".to_string(),
            description: "List the entries of a directory in the workspace. Directories end \
                 with a slash. `.git` is never listed."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, relative to the workspace root. \
                             Defaults to the root itself."
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_DEPTH,
                        "description": "How many directory levels to descend. Defaults to 1."
                    }
                },
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
            let requested = args.path.unwrap_or_else(|| ".".to_string());
            let base = match resolve_in_root(&root, &requested) {
                Ok(path) => path,
                Err(message) => return failure(message),
            };
            let depth = args.depth.unwrap_or(1).clamp(1, MAX_DEPTH);

            let mut lines = Vec::new();
            if let Err(err) = walk(&base, &base, depth, &mut lines) {
                return failure(format!("list_directory {requested}: {err}"));
            }
            let (content, truncated) = truncate(lines.join("\n"), MAX_BYTES);
            success(content, truncated)
        })
    }
}

/// Depth-first, name-ordered, so the same directory always lists the same way
/// and two identical calls stay comparable.
fn walk(base: &Path, dir: &Path, depth: usize, lines: &mut Vec<String>) -> std::io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_dir = entry.file_type()?.is_dir();
        if is_dir && SKIPPED_DIRS.contains(&name.as_ref()) {
            continue;
        }
        let path = entry.path();
        let shown = path.strip_prefix(base).unwrap_or(&path).to_string_lossy();
        lines.push(if is_dir {
            format!("{shown}/")
        } else {
            shown.to_string()
        });
        if is_dir && depth > 1 {
            walk(base, &path, depth - 1, lines)?;
        }
    }
    Ok(())
}

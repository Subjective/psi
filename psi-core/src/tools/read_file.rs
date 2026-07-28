use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::json;

use super::path::resolve_in_root;
use super::{blocking, failure, parse_args, success, truncate};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolSpec};

const MAX_BYTES: usize = 64 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
    /// 1-based and inclusive, matching how the model is asked to cite lines.
    #[serde(default)]
    start_line: Option<usize>,
    #[serde(default)]
    line_count: Option<usize>,
}

pub struct ReadFile {
    root: PathBuf,
}

impl ReadFile {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read a UTF-8 text file in the workspace. Returns the file's exact \
                 text, with no line numbers added, so the text can be quoted back to \
                 apply_patch verbatim."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root."
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First line to read, 1-based. Defaults to the first line."
                    },
                    "line_count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "How many lines to read. Defaults to the rest of the file."
                    }
                },
                "required": ["path"],
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
            let path = match resolve_in_root(&root, &args.path) {
                Ok(path) => path,
                Err(message) => return failure(message),
            };
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => return failure(format!("read_file {}: {err}", args.path)),
            };
            let text = match (args.start_line, args.line_count) {
                (None, None) => text,
                (start, count) => {
                    let start = start.unwrap_or(1).max(1) - 1;
                    let selected: Vec<&str> = text
                        .lines()
                        .skip(start)
                        .take(count.unwrap_or(usize::MAX))
                        .collect();
                    selected.join("\n")
                }
            };
            let (content, truncated) = truncate(text, MAX_BYTES);
            success(content, truncated)
        })
    }
}

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::json;

use super::path::resolve_in_root;
use super::{blocking, failure, parse_args, success};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolSpec};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    path: String,
    old_text: String,
    new_text: String,
}

pub struct ApplyPatch {
    root: PathBuf,
}

impl ApplyPatch {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for ApplyPatch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "apply_patch".to_string(),
            description: "Edit one file in the workspace by replacing an exact stretch of its \
                 text. `old_text` must appear exactly once in the file; pass an empty \
                 `old_text` to create a new file with `new_text` as its contents."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to replace, copied exactly from the file, with \
                             enough surrounding lines to appear only once. Empty to create a \
                             new file."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to put in its place. Empty to delete the old text."
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        }
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Mutating
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

            if args.old_text.is_empty() {
                if path.exists() {
                    return failure(format!(
                        "apply_patch {}: file already exists; pass the text to replace as \
                         old_text",
                        args.path
                    ));
                }
                if let Some(parent) = path.parent()
                    && let Err(err) = fs::create_dir_all(parent)
                {
                    return failure(format!("apply_patch {}: {err}", args.path));
                }
                return match fs::write(&path, &args.new_text) {
                    Ok(()) => success(format!("created {}", args.path), false),
                    Err(err) => failure(format!("apply_patch {}: {err}", args.path)),
                };
            }

            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => return failure(format!("apply_patch {}: {err}", args.path)),
            };
            match text.matches(&args.old_text).count() {
                0 => failure(format!(
                    "apply_patch {}: old_text is not in the file",
                    args.path
                )),
                1 => {
                    let patched = text.replace(&args.old_text, &args.new_text);
                    match fs::write(&path, patched) {
                        Ok(()) => success(format!("updated {}", args.path), false),
                        Err(err) => failure(format!("apply_patch {}: {err}", args.path)),
                    }
                }
                count => failure(format!(
                    "apply_patch {}: old_text appears {count} times; include more surrounding \
                     lines so it appears once",
                    args.path
                )),
            }
        })
    }
}

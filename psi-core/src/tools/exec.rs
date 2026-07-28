use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use tokio::time::timeout;

use super::{failure, parse_args, success, truncate};
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolSpec};

const MAX_BYTES: usize = 32 * 1024;
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 600_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    command: String,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

/// The general escape hatch. It inherits Psi's process permissions and is
/// deliberately not confined to the workspace root; run Psi in a container for
/// untrusted work (docs/design.md, "Trusted environment and hooks").
pub struct Exec {
    root: PathBuf,
}

impl Exec {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for Exec {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "exec".to_string(),
            description: format!(
                "Run a shell command with `sh -c`, from the workspace root. Returns the \
                 command's combined output and its exit status. Defaults to a \
                 {DEFAULT_TIMEOUT_MS}ms timeout, after which the command is killed."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command line to run."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TIMEOUT_MS,
                        "description": "How long to let the command run, in milliseconds."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Unknown
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        let root = self.root.clone();
        Box::pin(async move {
            let args: Args = match parse_args(&invocation.arguments) {
                Ok(args) => args,
                Err(output) => return output,
            };
            let limit = args
                .timeout_ms
                .unwrap_or(DEFAULT_TIMEOUT_MS)
                .clamp(1, MAX_TIMEOUT_MS);

            let mut command = tokio::process::Command::new("/bin/sh");
            command
                .arg("-c")
                .arg(&args.command)
                .current_dir(&root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                // The engine drops this future to cancel the turn, and the
                // timeout below drops it too; either way the child must go.
                .kill_on_drop(true);

            let output = match timeout(Duration::from_millis(limit), command.output()).await {
                Ok(Ok(output)) => output,
                Ok(Err(err)) => return failure(format!("exec: {err}")),
                Err(_) => return failure(format!("exec: killed after {limit}ms")),
            };

            let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
            if !output.stderr.is_empty() {
                if !content.is_empty() && !content.ends_with('\n') {
                    content.push('\n');
                }
                content.push_str("[stderr]\n");
                content.push_str(&String::from_utf8_lossy(&output.stderr));
            }
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str(&match output.status.code() {
                Some(code) => format!("[exit status: {code}]"),
                None => "[exit status: killed by signal]".to_string(),
            });

            // A non-zero exit is an answer, not a broken tool: the model has to
            // read failing test output. Only a command that could not run at
            // all fails the tool_result item.
            let (content, truncated) = truncate(content, MAX_BYTES);
            success(content, truncated)
        })
    }
}

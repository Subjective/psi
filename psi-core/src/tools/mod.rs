//! The five default tools (docs/design.md, "Five tools, one profile"). The
//! structured tools carry the schemas that make calls comparable — canonical
//! arguments, declared effects — and every one of them bounds its output so a
//! single call cannot flood the model's context.

mod apply_patch;
mod exec;
mod list_directory;
mod path;
mod read_file;
mod search;

pub use apply_patch::ApplyPatch;
pub use exec::Exec;
pub use list_directory::ListDirectory;
pub use read_file::ReadFile;
pub use search::Search;

use std::path::PathBuf;

use serde::de::DeserializeOwned;

use crate::tool::{ToolFuture, ToolOutput, ToolRegistry};

/// Directories no walk descends into. `.git` is large, binary, and never what
/// the model means; excluding it keeps `list_directory` and `search` usable in
/// a real repository.
const SKIPPED_DIRS: [&str; 1] = [".git"];

/// The advertised profile, in advertisement order. The agent and the predictor
/// always share it, so their calls stay comparable.
pub fn default_profile(workspace: PathBuf) -> ToolRegistry {
    let root = path::canonical_root(workspace);
    let mut registry = ToolRegistry::new();
    registry.register(ReadFile::new(root.clone()));
    registry.register(ListDirectory::new(root.clone()));
    registry.register(Search::new(root.clone()));
    registry.register(ApplyPatch::new(root.clone()));
    registry.register(Exec::new(root));
    registry
}

/// Parses the model's arguments into a tool's own shape. Unknown fields are
/// rejected: the argument space is exactly the advertised schema, which is
/// what makes two calls with the same arguments the same call.
fn parse_args<T: DeserializeOwned>(arguments: &serde_json::Value) -> Result<T, ToolOutput> {
    serde_json::from_value(arguments.clone())
        .map_err(|err| failure(format!("invalid arguments: {err}")))
}

/// Caps tool output. The marker tells the model what it is missing; the flag
/// lands on the tool_result item's `truncated` field.
fn truncate(mut content: String, limit: usize) -> (String, bool) {
    if content.len() <= limit {
        return (content, false);
    }
    let total = content.len();
    let mut cut = limit;
    while !content.is_char_boundary(cut) {
        cut -= 1;
    }
    content.truncate(cut);
    content.push_str(&format!("\n[truncated: {cut} of {total} bytes shown]"));
    (content, true)
}

/// A tool result the model can act on. Empty content reads as a broken tool
/// rather than an empty answer, so it is spelled out instead.
fn success(content: String, truncated: bool) -> ToolOutput {
    let content = if content.is_empty() {
        "(no output)".to_string()
    } else {
        content
    };
    ToolOutput {
        content,
        error: None,
        truncated,
    }
}

/// A failed call. The message is what the model sees and what marks the
/// tool_result item failed.
fn failure(message: String) -> ToolOutput {
    ToolOutput {
        content: message.clone(),
        error: Some(message),
        truncated: false,
    }
}

/// Runs a tool body that uses blocking filesystem calls off the engine's
/// runtime thread.
fn blocking(body: impl FnOnce() -> ToolOutput + Send + 'static) -> ToolFuture {
    Box::pin(async move {
        match tokio::task::spawn_blocking(body).await {
            Ok(output) => output,
            Err(err) => failure(format!("tool did not finish: {err}")),
        }
    })
}

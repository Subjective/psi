use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// What the model sees when a tool is advertised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the argument object.
    pub parameters: serde_json::Value,
}

/// A tool's declared effect on the workspace; drives revision bumps and,
/// later, the speculative allowlist (Milestone 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffect {
    /// Never mutates: the revision is untouched (`read_file`, `list_directory`, `search`).
    ReadOnly,
    /// Mutates when it succeeds: bump on success (`apply_patch`).
    Mutating,
    /// Effects unknowable: bump after every run (`exec`).
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub call_id: String,
    pub arguments: serde_json::Value,
    pub cwd: PathBuf,
}

/// What a tool returns. `content` is what the model sees either way; `error`
/// marks the tool_result item failed. Duration is measured by the engine.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub error: Option<String>,
    pub truncated: bool,
}

pub type ToolFuture = Pin<Box<dyn Future<Output = ToolOutput> + Send>>;

pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn effect(&self) -> ToolEffect;
    fn execute(&self, invocation: ToolInvocation) -> ToolFuture;
}

/// The advertised tool set, in advertisement order. One registry is one
/// profile for now; profiles separate from the registry only when it holds
/// more tools than are advertised (see docs/design.md).
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.spec().name == name)
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec()).collect()
    }
}

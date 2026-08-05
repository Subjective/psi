use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Identifies one item within its session. Ids are assigned in append order,
/// so id order equals log order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ItemId(pub u64);

/// Groups the items of one turn: one user message through the assistant
/// response that ends it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub u64);

/// Counter bumped after workspace mutations. Recorded on every tool call and
/// used to scope the speculative cache (Milestone 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceRevision(pub u64);

/// How an item or turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    Completed,
    Cancelled,
    Failed,
}

impl fmt::Display for CompletionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CompletionStatus::Completed => "completed",
            CompletionStatus::Cancelled => "cancelled",
            CompletionStatus::Failed => "failed",
        })
    }
}

/// One record in a session's append-only tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: ItemId,
    /// `None` marks a root: the first item of a session, or of a branch
    /// started from an empty head.
    pub parent_id: Option<ItemId>,
    pub turn_id: TurnId,
    pub created_at_ms: u64,
    pub status: CompletionStatus,
    /// Present when `status` is `Failed`.
    pub error: Option<String>,
    #[serde(flatten)]
    pub payload: ItemPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ItemPayload {
    UserMessage {
        text: String,
    },
    AssistantMessage {
        text: String,
    },
    Reasoning {
        text: String,
        /// The provider's own record of this reasoning, replayed verbatim so a
        /// reasoning model sees its own encrypted reasoning again. Opaque: the
        /// harness never reads inside it. `None` when the backend has nothing
        /// to replay.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_data: Option<serde_json::Value>,
    },
    ToolCall {
        tool: String,
        call_id: String,
        arguments: serde_json::Value,
        cwd: PathBuf,
        revision: WorkspaceRevision,
    },
    ToolResult {
        call_id: String,
        content: String,
        duration_ms: u64,
        truncated: bool,
    },
}

impl ItemPayload {
    pub fn kind(&self) -> ItemKind {
        match self {
            ItemPayload::UserMessage { .. } => ItemKind::UserMessage,
            ItemPayload::AssistantMessage { .. } => ItemKind::AssistantMessage,
            ItemPayload::Reasoning { .. } => ItemKind::Reasoning,
            ItemPayload::ToolCall { .. } => ItemKind::ToolCall,
            ItemPayload::ToolResult { .. } => ItemKind::ToolResult,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    UserMessage,
    AssistantMessage,
    Reasoning,
    ToolCall,
    ToolResult,
}

impl fmt::Display for ItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ItemKind::UserMessage => "user_message",
            ItemKind::AssistantMessage => "assistant_message",
            ItemKind::Reasoning => "reasoning",
            ItemKind::ToolCall => "tool_call",
            ItemKind::ToolResult => "tool_result",
        })
    }
}

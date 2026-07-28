//! The provider-neutral model boundary. The harness never sees provider wire
//! types; backends translate to this surface. Argument deltas, usage, and
//! provider passthrough data join with the Responses codec (Milestone 2),
//! which is their first consumer.

use tokio::sync::mpsc;

use crate::item::Item;
use crate::session::SessionId;
use crate::tool::ToolSpec;

/// One request to the model: the active path plus the advertised tool profile.
#[derive(Debug, Clone)]
pub struct TurnRequest {
    pub session_id: SessionId,
    pub items: Vec<Item>,
    pub tools: Vec<ToolSpec>,
}

/// A completed tool call proposed by the model.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub call_id: String,
    pub tool: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum ModelEvent {
    TextDelta { delta: String },
    ReasoningDelta { delta: String },
    ToolCallCompleted { call: ToolCallRequest },
    Completed,
    Error { message: String },
}

/// Turns one request into a stream of events. A stream that ends without
/// `Completed` or `Error` is treated as a failure — a silent early end is
/// never success (see the vLLM guards in docs/design.md).
pub trait ModelBackend: Send + Sync {
    fn stream_response(&self, request: TurnRequest) -> mpsc::Receiver<ModelEvent>;
}

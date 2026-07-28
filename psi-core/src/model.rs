//! The provider-neutral model boundary. The harness never sees provider wire
//! types; backends translate to this surface.

use serde::{Deserialize, Serialize};
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
    pub sampling: Sampling,
}

/// Sampling overrides for one request. The authoritative turn leaves both unset
/// and takes the server's own defaults; the prediction strategies set them
/// (`crate::predictor`), because the prediction budget is a cap on generated
/// tokens and branch sampling needs a temperature above zero for its samples to
/// differ. A field is only rendered into the request body when it is set, so an
/// OpenAI reasoning model — which rejects `temperature` outright — is never
/// sent one by the authoritative path.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f64>,
    /// The generated-token cap, which is what the prediction budget bounds.
    pub max_output_tokens: Option<u64>,
}

/// A completed tool call proposed by the model.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub call_id: String,
    pub tool: String,
    pub arguments: serde_json::Value,
}

/// Tokens billed for one model response. The engine sums these over a turn's
/// responses and reports the total on `turn_finished`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[derive(Debug, Clone)]
pub enum ModelEvent {
    TextDelta {
        delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    /// Ends the reasoning item and carries the provider's own record of it,
    /// encrypted reasoning included, for verbatim replay. Arrives even when
    /// the provider streamed no reasoning text.
    ReasoningCompleted {
        provider_data: serde_json::Value,
    },
    /// Tool-call arguments as they stream. `tool` repeats on every delta so a
    /// call whose arguments never finish still names the tool it meant to run.
    ToolCallArgumentsDelta {
        call_id: String,
        tool: String,
        delta: String,
    },
    ToolCallCompleted {
        call: ToolCallRequest,
    },
    Usage {
        usage: Usage,
    },
    Completed,
    Error {
        message: String,
    },
}

/// Turns one request into a stream of events. A stream that ends without
/// `Completed` or `Error` is treated as a failure — a silent early end is
/// never success (see the vLLM guards in docs/design.md). Dropping the
/// receiver cancels the request.
pub trait ModelBackend: Send + Sync {
    fn stream_response(&self, request: TurnRequest) -> mpsc::Receiver<ModelEvent>;
}

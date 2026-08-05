//! The replay oracle (docs/design.md, "Speculation"): a fake predictor used
//! only in tests and benchmarks. It is always right, so a run driven by it
//! measures the ceiling — the savings available when prediction is perfect —
//! which says whether speculation can pay at all before any real predictor
//! exists.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::fake::FakeResponse;
use crate::model::{ModelEvent, ToolCallRequest, TurnRequest};
use crate::speculation::{PredictionFuture, Predictor};

/// Predicts, for each model response, exactly the tool calls that response
/// will make.
pub struct ReplayOracle {
    rounds: Mutex<VecDeque<Vec<ToolCallRequest>>>,
}

impl ReplayOracle {
    /// Built from the same script the fake model plays. The script is the
    /// recorded session: each response's tool calls are what the recording
    /// does next, and the model and the oracle consume their rounds in
    /// lockstep, one per response.
    pub fn for_script(script: &[FakeResponse]) -> Self {
        let rounds = script
            .iter()
            .map(|response| {
                response
                    .events
                    .iter()
                    .filter_map(|event| match event {
                        ModelEvent::ToolCallCompleted { call } => Some(call.clone()),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        Self {
            rounds: Mutex::new(rounds),
        }
    }
}

impl Predictor for ReplayOracle {
    fn predict(&self, _request: &TurnRequest) -> PredictionFuture {
        let calls = self
            .rounds
            .lock()
            .expect("oracle lock")
            .pop_front()
            .unwrap_or_default();
        Box::pin(async move { calls })
    }
}

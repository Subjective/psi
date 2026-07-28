//! The two prediction strategies (docs/design.md, "Speculation", Milestone 7).
//! Both are `Predictor`s over an ordinary `ModelBackend`, so a predictor is a
//! consumer of the same backend and codec the agent uses rather than a second
//! model boundary, and both send the authoritative request's context and tool
//! profile unchanged — the agent and the predictor share a profile, so their
//! calls are comparable.
//!
//! Direct proposal asks once for the calls the assistant is about to make.
//! Branch sampling asks nothing: it samples the predictor continuing the turn
//! itself, `samples` times at temperature, and keeps the tool calls each
//! continuation makes. Agreement across the samples is its ranking.
//!
//! Branch sampling issues its samples as concurrent requests, because vLLM's
//! `/v1/responses` has no `n`: `ResponsesRequest`
//! (vllm/entrypoints/openai/responses/protocol.py) declares no such field and
//! its `to_sampling_params` never sets `SamplingParams.n`, while the base model
//! it parses into allows unknown fields and only logs them — so an `n` sent
//! there would be dropped in silence. The server batches the requests it receives together,
//! and its prefix cache serves the shared prompt to the samples that arrive
//! behind the first — so the prefix is shared through the cache rather than, as
//! `n` would give, through one forked sequence, and every sample bills its own
//! input tokens. Neither shape shares anything with the authoritative model:
//! that is a different model answering a different request.
//!
//! The prediction budget caps generated tokens. Direct proposal spends it on
//! its one request; branch sampling divides it across the samples, so a round
//! of either costs the same generation at most. What each round really billed
//! rides back on the `Prediction` and into the trace.
//!
//! A strategy never fails a turn. A request that fails, times out, or answers
//! with nothing usable yields an empty prediction — a missed round, recorded
//! with its reason.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::item::{CompletionStatus, Item, ItemId, ItemPayload, TurnId};
use crate::model::{ModelBackend, ModelEvent, Sampling, ToolCallRequest, TurnRequest, Usage};
use crate::speculation::{Prediction, PredictionFuture, Predictor, canonical_json};

/// Appended to the shared context as the last user message. The tools are
/// already on the request, so it names no profile; it asks for calls and
/// nothing else, because every other token it draws is budget spent on output
/// the runtime discards.
const PROPOSAL_INSTRUCTION: &str = "\
Do not answer this yourself. Predict what the assistant does next: call the \
tools you expect its next response to call, with the exact arguments you expect \
it to use, most likely first. Call them now and write nothing else.";

/// The temperature branch sampling draws its continuations at. One is the
/// model's own distribution, which is what "sample the model continuing" means;
/// below it the samples collapse toward each other and agreement stops ranking
/// anything.
const SAMPLE_TEMPERATURE: f64 = 1.0;

/// One predictor request asking for the calls the assistant will make next.
pub struct DirectProposal {
    model: Arc<dyn ModelBackend>,
}

impl DirectProposal {
    pub fn new(model: Arc<dyn ModelBackend>) -> Self {
        Self { model }
    }
}

impl Predictor for DirectProposal {
    fn predict(&self, request: &TurnRequest, budget: u64) -> PredictionFuture {
        let mut request = request.clone();
        request.items.push(instruction_item(&request.items));
        request.sampling = Sampling {
            // The most likely calls, not a sample: this strategy asks once and
            // takes the model's own best guess.
            temperature: Some(0.0),
            max_output_tokens: Some(budget.max(1)),
        };
        let events = self.model.stream_response(request);
        Box::pin(async move {
            let sample = harvest(events).await;
            Prediction {
                // The predictor's own order is its ranking.
                calls: deduplicated(sample.calls),
                usage: sample.usage,
                error: sample.error,
            }
        })
    }
}

/// `samples` temperature-sampled continuations, ranked by how many of them
/// propose each call.
pub struct BranchSampling {
    model: Arc<dyn ModelBackend>,
    samples: usize,
}

impl BranchSampling {
    /// `samples` is clamped to at least one: a round with no samples is not a
    /// cheaper strategy, it is no strategy.
    pub fn new(model: Arc<dyn ModelBackend>, samples: usize) -> Self {
        Self {
            model,
            samples: samples.max(1),
        }
    }
}

impl Predictor for BranchSampling {
    fn predict(&self, request: &TurnRequest, budget: u64) -> PredictionFuture {
        let mut request = request.clone();
        request.sampling = Sampling {
            temperature: Some(SAMPLE_TEMPERATURE),
            max_output_tokens: Some((budget / self.samples as u64).max(1)),
        };
        // Every request is issued here, before any of them is read, so the
        // server has all the samples to batch together. Draining them one after
        // another does not serialize them: each one's events buffer in its own
        // channel while the others are being read.
        let streams: Vec<_> = (0..self.samples)
            .map(|_| self.model.stream_response(request.clone()))
            .collect();
        Box::pin(async move {
            let mut usage = Usage::default();
            let mut error = None;
            let mut samples = Vec::new();
            for stream in streams {
                let sample = harvest(stream).await;
                usage.add(sample.usage);
                // The first failure is the one reported: a round whose samples
                // all fail the same way says so once.
                error = error.or(sample.error);
                samples.push(sample.calls);
            }
            Prediction {
                calls: ranked(samples),
                usage,
                error,
            }
        })
    }
}

/// One predictor response, drained.
#[derive(Default)]
struct Sample {
    calls: Vec<ToolCallRequest>,
    usage: Usage,
    error: Option<String>,
}

/// Drains one response into the calls it made and what it billed. Calls that
/// arrived before a failure are kept: they are guesses like any others, and the
/// worst a wrong one costs is a speculative execution nothing adopts.
async fn harvest(mut events: mpsc::Receiver<ModelEvent>) -> Sample {
    let mut sample = Sample::default();
    let mut ended = false;
    while let Some(event) = events.recv().await {
        match event {
            ModelEvent::ToolCallCompleted { call } => sample.calls.push(call),
            ModelEvent::Usage { usage } => sample.usage.add(usage),
            ModelEvent::Error { message } => {
                sample.error = Some(message);
                ended = true;
                break;
            }
            ModelEvent::Completed => {
                ended = true;
                break;
            }
            _ => {}
        }
    }
    if !ended {
        sample.error = Some("the predictor stream ended without completing".to_string());
    }
    sample
}

/// The identity two guesses have to share to be the same guess: the cache key's
/// tool and canonical arguments, without the working directory and revision the
/// runtime supplies itself.
fn identity(call: &ToolCallRequest) -> String {
    format!("{} {}", call.tool, canonical_json(&call.arguments))
}

/// Keeps the first of each distinct call, in the order they were proposed.
fn deduplicated(calls: Vec<ToolCallRequest>) -> Vec<ToolCallRequest> {
    let mut seen = HashSet::new();
    calls
        .into_iter()
        .filter(|call| seen.insert(identity(call)))
        .collect()
}

/// Orders distinct calls by how many samples proposed them, ties broken by
/// first appearance. A call proposed twice inside one sample counts once for
/// it, so a repetitive sample cannot outvote agreement.
fn ranked(samples: Vec<Vec<ToolCallRequest>>) -> Vec<ToolCallRequest> {
    let mut agreement: HashMap<String, usize> = HashMap::new();
    let mut order: Vec<ToolCallRequest> = Vec::new();
    let mut seen = HashSet::new();
    for sample in samples {
        let mut counted = HashSet::new();
        for call in sample {
            let key = identity(&call);
            if !counted.insert(key.clone()) {
                continue;
            }
            *agreement.entry(key.clone()).or_default() += 1;
            if seen.insert(key) {
                order.push(call);
            }
        }
    }
    // A stable sort leaves calls with equal agreement in first-appearance
    // order, which is the tiebreak.
    order.sort_by_key(|call| std::cmp::Reverse(agreement[&identity(call)]));
    order
}

/// The proposal instruction as an item the codec can render. It never enters a
/// session: the codecs read an item's payload and status and nothing else, so
/// its ids only have to continue the path it is appended to.
fn instruction_item(items: &[Item]) -> Item {
    let last = items.last();
    Item {
        id: ItemId(last.map_or(0, |item| item.id.0 + 1)),
        parent_id: last.map(|item| item.id),
        turn_id: last.map_or(TurnId(0), |item| item.turn_id),
        created_at_ms: last.map_or(0, |item| item.created_at_ms),
        status: CompletionStatus::Completed,
        error: None,
        payload: ItemPayload::UserMessage {
            text: PROPOSAL_INSTRUCTION.to_string(),
        },
    }
}

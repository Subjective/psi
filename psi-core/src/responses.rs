//! The OpenAI Responses codec: harness items and tool specs in, request JSON
//! out; stream events in, `ModelEvent`s out. Both backends speak this wire
//! format, so the codec is shared and the backends differ only in transport
//! and capabilities (docs/design.md, "Model backends: one Responses codec").
//!
//! Requests are stateless. Psi owns history, so nothing is stored provider
//! side and reasoning replays only through the encrypted reasoning content
//! carried opaquely on reasoning items.

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::item::{CompletionStatus, Item, ItemPayload};
use crate::model::{ModelEvent, ToolCallRequest, TurnRequest, Usage};
use crate::tool::ToolSpec;

/// Builds one streaming `/responses` request body.
pub fn build_request(model: &str, instructions: &str, request: &TurnRequest) -> Value {
    json!({
        "model": model,
        "instructions": instructions,
        "input": build_input(&request.items),
        "tools": request.tools.iter().map(tool_json).collect::<Vec<_>>(),
        "tool_choice": "auto",
        // The engine runs a response's tool calls one after another, so asking
        // for parallel proposals would only blur where a turn's time went.
        "parallel_tool_calls": false,
        "reasoning": { "summary": "auto" },
        // Psi owns history; provider-side storage is never authoritative.
        "store": false,
        // With `store: false` this is the only way a reasoning model sees its
        // own earlier reasoning again.
        "include": ["reasoning.encrypted_content"],
        "stream": true
    })
}

fn tool_json(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
    })
}

fn build_input(items: &[Item]) -> Vec<Value> {
    // A function call the provider cannot pair with its output is rejected on
    // replay, so a call whose arguments never finished streaming is dropped.
    let answered: HashSet<&str> = items
        .iter()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();

    let mut input = Vec::new();
    for (index, item) in items.iter().enumerate() {
        match &item.payload {
            ItemPayload::UserMessage { text } => input.push(json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }],
            })),
            ItemPayload::AssistantMessage { text } => {
                // A cancelled turn can leave an assistant message with nothing
                // in it, which the provider rejects as empty content.
                if !text.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }],
                    }));
                }
            }
            ItemPayload::Reasoning { provider_data, .. } => {
                if let Some(data) = provider_data
                    && reasoning_has_output(items, index, &answered)
                {
                    input.push(data.clone());
                }
            }
            ItemPayload::ToolCall {
                tool,
                call_id,
                arguments,
                ..
            } => {
                if answered.contains(call_id.as_str()) {
                    input.push(json!({
                        "type": "function_call",
                        "name": tool,
                        "call_id": call_id,
                        // The Responses API carries call arguments as a JSON
                        // string, not as an object.
                        "arguments": arguments.to_string(),
                    }));
                }
            }
            ItemPayload::ToolResult {
                call_id, content, ..
            } => input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output_text(item, content),
            })),
        }
    }
    input
}

/// A reasoning item must be followed by the output it reasoned toward, or the
/// provider rejects the whole input. A turn cancelled just after reasoning
/// leaves one dangling, so it is dropped instead of replayed.
fn reasoning_has_output(items: &[Item], index: usize, answered: &HashSet<&str>) -> bool {
    match items.get(index + 1).map(|item| &item.payload) {
        Some(ItemPayload::AssistantMessage { text }) => !text.is_empty(),
        Some(ItemPayload::ToolCall { call_id, .. }) => answered.contains(call_id.as_str()),
        Some(ItemPayload::Reasoning { .. }) => reasoning_has_output(items, index + 1, answered),
        _ => false,
    }
}

/// A cancelled call records no content, and an empty output reads as a broken
/// tool rather than an empty answer.
fn tool_output_text(item: &Item, content: &str) -> String {
    if !content.is_empty() {
        return content.to_string();
    }
    match item.status {
        CompletionStatus::Cancelled => "call cancelled before it ran".to_string(),
        _ => "(no output)".to_string(),
    }
}

/// Assembles stream events into `ModelEvent`s. Argument deltas name only the
/// output item they belong to, so the decoder remembers each function call's
/// id and tool name from `response.output_item.added`.
#[derive(Default)]
pub struct Decoder {
    calls: HashMap<String, PendingCall>,
}

struct PendingCall {
    call_id: String,
    tool: String,
}

impl Decoder {
    /// Decodes one stream event. Event types Psi does not branch on decode to
    /// nothing; `response.completed` decodes to usage followed by completion.
    pub fn decode(&mut self, event: &Value) -> Vec<ModelEvent> {
        match string(event, "type").unwrap_or_default() {
            "response.output_text.delta" => string(event, "delta")
                .map(|delta| {
                    vec![ModelEvent::TextDelta {
                        delta: delta.to_string(),
                    }]
                })
                .unwrap_or_default(),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                string(event, "delta")
                    .map(|delta| {
                        vec![ModelEvent::ReasoningDelta {
                            delta: delta.to_string(),
                        }]
                    })
                    .unwrap_or_default()
            }
            "response.output_item.added" => {
                self.remember_call(event.get("item"));
                Vec::new()
            }
            "response.function_call_arguments.delta" => {
                let (Some(item_id), Some(delta)) =
                    (string(event, "item_id"), string(event, "delta"))
                else {
                    return Vec::new();
                };
                self.calls
                    .get(item_id)
                    .map(|pending| {
                        vec![ModelEvent::ToolCallArgumentsDelta {
                            call_id: pending.call_id.clone(),
                            tool: pending.tool.clone(),
                            delta: delta.to_string(),
                        }]
                    })
                    .unwrap_or_default()
            }
            "response.output_item.done" => self.finish_item(event.get("item")),
            "response.completed" => {
                let mut events = Vec::new();
                if let Some(usage) = event.get("response").and_then(|r| r.get("usage")) {
                    events.push(ModelEvent::Usage {
                        usage: Usage {
                            input_tokens: number(usage, "input_tokens"),
                            output_tokens: number(usage, "output_tokens"),
                        },
                    });
                }
                events.push(ModelEvent::Completed);
                events
            }
            "response.failed" => {
                let message = event
                    .get("response")
                    .and_then(|response| response.get("error"))
                    .and_then(|error| string(error, "message"))
                    .unwrap_or("response failed");
                vec![ModelEvent::Error {
                    message: message.to_string(),
                }]
            }
            "response.incomplete" => {
                let reason = event
                    .get("response")
                    .and_then(|response| response.get("incomplete_details"))
                    .and_then(|details| string(details, "reason"))
                    .unwrap_or("unknown");
                vec![ModelEvent::Error {
                    message: format!("response incomplete: {reason}"),
                }]
            }
            "error" => vec![ModelEvent::Error {
                message: string(event, "message")
                    .unwrap_or("stream error")
                    .to_string(),
            }],
            _ => Vec::new(),
        }
    }

    fn remember_call(&mut self, item: Option<&Value>) {
        let Some(item) = item else { return };
        if string(item, "type") != Some("function_call") {
            return;
        }
        let (Some(item_id), Some(call_id), Some(tool)) = (
            string(item, "id"),
            string(item, "call_id"),
            string(item, "name"),
        ) else {
            return;
        };
        self.calls.insert(
            item_id.to_string(),
            PendingCall {
                call_id: call_id.to_string(),
                tool: tool.to_string(),
            },
        );
    }

    fn finish_item(&mut self, item: Option<&Value>) -> Vec<ModelEvent> {
        let Some(item) = item else {
            return Vec::new();
        };
        if let Some(item_id) = string(item, "id") {
            self.calls.remove(item_id);
        }
        match string(item, "type") {
            Some("function_call") => {
                let (Some(call_id), Some(tool)) = (string(item, "call_id"), string(item, "name"))
                else {
                    return Vec::new();
                };
                let raw = string(item, "arguments").unwrap_or("");
                let arguments = if raw.trim().is_empty() {
                    json!({})
                } else {
                    match serde_json::from_str(raw) {
                        Ok(arguments) => arguments,
                        // Arguments that are not JSON are a protocol violation
                        // with no sane recovery: the call cannot be run and the
                        // model cannot be told which call failed.
                        Err(err) => {
                            return vec![ModelEvent::Error {
                                message: format!("tool call {tool} sent invalid arguments: {err}"),
                            }];
                        }
                    }
                };
                vec![ModelEvent::ToolCallCompleted {
                    call: ToolCallRequest {
                        call_id: call_id.to_string(),
                        tool: tool.to_string(),
                        arguments,
                    },
                }]
            }
            Some("reasoning") => {
                let mut provider_data = item.clone();
                // A stateless response has no server-side item to refer back
                // to, so the id goes before the item is stored for replay.
                if let Some(object) = provider_data.as_object_mut() {
                    object.remove("id");
                }
                vec![ModelEvent::ReasoningCompleted { provider_data }]
            }
            _ => Vec::new(),
        }
    }
}

fn string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn number(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Reassembles server-sent events from a byte stream. Responses puts one JSON
/// object in each event's `data` lines, and its `event` line only repeats that
/// object's `type`, so nothing but data is read.
#[derive(Default)]
pub struct SseBuffer {
    buffer: Vec<u8>,
}

impl SseBuffer {
    /// Appends bytes and returns the data payload of every event now complete.
    /// A frame is held until its blank-line terminator arrives, so a character
    /// split across two chunks is never decoded from half its bytes.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some((end, terminator)) = frame_end(&self.buffer) {
            let frame: Vec<u8> = self.buffer.drain(..end + terminator).collect();
            if let Ok(text) = std::str::from_utf8(&frame[..end])
                && let Some(payload) = data_lines(text)
            {
                payloads.push(payload);
            }
        }
        payloads
    }
}

/// Locates the blank line ending the first frame: its length, then the length
/// of the terminator to discard.
fn frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    (0..buffer.len()).find_map(|index| {
        let rest = &buffer[index..];
        if rest.starts_with(b"\n\n") {
            Some((index, 2))
        } else if rest.starts_with(b"\r\n\r\n") {
            Some((index, 4))
        } else {
            None
        }
    })
}

fn data_lines(frame: &str) -> Option<String> {
    let mut data = String::new();
    for line in frame.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    (!data.is_empty()).then_some(data)
}

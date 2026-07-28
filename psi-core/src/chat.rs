//! The Chat Completions codec, which exists only as a predictor-side config
//! switch for model-parser combinations whose Responses streaming misbehaves
//! (docs/design.md, "Model backends: one Responses codec, explicit
//! capabilities"). `VllmBackend` picks it when its config names this endpoint;
//! nothing else in Psi speaks it, and the authoritative loop never does.
//!
//! It is non-streaming. The predictor needs the whole proposal before it can
//! rank or deduplicate anything, so streaming would buy an event-assembly layer
//! whose output is thrown away, and dropping the response still cancels the
//! request.
//!
//! Two differences from the Responses body are deliberate. Reasoning items are
//! dropped, because Chat has no slot that replays them and vLLM re-derives
//! reasoning from the prompt anyway. And `parallel_tool_calls` is true, because
//! vLLM truncates a reply to its first tool call when it is explicitly false
//! (`maybe_filter_parallel_tool_calls`), which would cap every proposal at one
//! call — the opposite of what a predictor is asked for.

use std::collections::HashSet;

use serde_json::{Value, json};

use crate::item::{CompletionStatus, Item, ItemPayload};
use crate::model::{ModelEvent, ToolCallRequest, TurnRequest, Usage};
use crate::tool::ToolSpec;

/// Builds one non-streaming `/chat/completions` request body.
pub fn build_request(model: &str, instructions: &str, request: &TurnRequest) -> Value {
    let mut body = json!({
        "model": model,
        "messages": build_messages(instructions, &request.items),
        "tools": request.tools.iter().map(tool_json).collect::<Vec<_>>(),
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "stream": false,
    });
    if let Some(temperature) = request.sampling.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_output_tokens) = request.sampling.max_output_tokens {
        body["max_completion_tokens"] = json!(max_output_tokens);
    }
    body
}

fn tool_json(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    })
}

fn build_messages(instructions: &str, items: &[Item]) -> Vec<Value> {
    // A tool call the reply cannot pair with its output confuses every chat
    // template that renders one, so a call whose result never landed is
    // dropped — the same rule the Responses codec applies.
    let answered: HashSet<&str> = items
        .iter()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();

    let mut messages = vec![json!({ "role": "system", "content": instructions })];
    for item in items {
        match &item.payload {
            ItemPayload::UserMessage { text } => {
                messages.push(json!({ "role": "user", "content": text }));
            }
            ItemPayload::AssistantMessage { text } => {
                if !text.is_empty() {
                    messages.push(json!({ "role": "assistant", "content": text }));
                }
            }
            ItemPayload::Reasoning { .. } => {}
            ItemPayload::ToolCall {
                tool,
                call_id,
                arguments,
                ..
            } => {
                if answered.contains(call_id.as_str()) {
                    messages.push(json!({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool,
                                // Chat carries call arguments as a JSON string.
                                "arguments": arguments.to_string(),
                            },
                        }],
                    }));
                }
            }
            ItemPayload::ToolResult {
                call_id, content, ..
            } => messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_output_text(item, content),
            })),
        }
    }
    messages
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

/// Turns one complete response body into the same `ModelEvent`s the streaming
/// codec produces, so the predictor harvests both endpoints the same way. Only
/// the first choice is read: the predictor sends one sample per request, and a
/// second choice would be a second sample conflated with the first.
pub fn decode_response(body: &Value) -> Vec<ModelEvent> {
    let mut events = Vec::new();
    let message = body
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"));
    let calls = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for call in calls {
        let function = call.get("function");
        let (Some(call_id), Some(tool)) = (
            string(call, "id"),
            function.and_then(|function| string(function, "name")),
        ) else {
            continue;
        };
        let raw = function
            .and_then(|function| string(function, "arguments"))
            .unwrap_or("");
        let arguments = if raw.trim().is_empty() {
            json!({})
        } else {
            match serde_json::from_str(raw) {
                Ok(arguments) => arguments,
                // Arguments that are not JSON name a call that cannot be run.
                Err(err) => {
                    return vec![ModelEvent::Error {
                        message: format!("tool call {tool} sent invalid arguments: {err}"),
                    }];
                }
            }
        };
        events.push(ModelEvent::ToolCallCompleted {
            call: ToolCallRequest {
                call_id: call_id.to_string(),
                tool: tool.to_string(),
                arguments,
            },
        });
    }
    if let Some(usage) = body.get("usage") {
        events.push(ModelEvent::Usage {
            usage: Usage {
                input_tokens: number(usage, "prompt_tokens"),
                output_tokens: number(usage, "completion_tokens"),
            },
        });
    }
    events.push(ModelEvent::Completed);
    events
}

fn string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn number(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

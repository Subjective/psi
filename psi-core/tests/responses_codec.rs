//! The Responses codec against fixture event streams: SSE framing, event
//! decoding, and the request bodies built from harness items (Milestone 2).

use psi_core::item::{CompletionStatus, Item, ItemId, ItemPayload, TurnId, WorkspaceRevision};
use psi_core::model::{ModelEvent, Sampling, TurnRequest, Usage};
use psi_core::responses::{Capabilities, Decoder, SseBuffer, build_request};
use psi_core::session::SessionId;
use psi_core::tool::ToolSpec;
use serde_json::{Value, json};

/// One reasoning-then-tool-call response, as the Responses API streams it.
const TOOL_CALL_STREAM: &str = "\
event: response.created
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}

event: response.output_item.added
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}

event: response.reasoning_summary_text.delta
data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"summary_index\":0,\"delta\":\"Check \"}

event: response.reasoning_summary_text.delta
data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"summary_index\":0,\"delta\":\"the file.\"}

event: response.output_item.done
data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Check the file.\"}],\"encrypted_content\":\"enc-blob\"}}

event: response.output_item.added
data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"\"}}

event: response.function_call_arguments.delta
data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"{\\\"path\\\":\"}

event: response.function_call_arguments.delta
data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"\\\"README.md\\\"}\"}

event: response.output_item.done
data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":120,\"output_tokens\":34,\"total_tokens\":154}}}

";

/// Feeds a stream through the framer and decoder one byte at a time, which is
/// the worst case for both: every frame and every character arrives split.
fn decode_stream(stream: &str) -> Vec<ModelEvent> {
    let mut frames = SseBuffer::default();
    let mut decoder = Decoder::default();
    let mut events = Vec::new();
    for byte in stream.as_bytes() {
        for payload in frames.push(&[*byte]) {
            let event: Value = serde_json::from_str(&payload).expect("event is JSON");
            events.extend(decoder.decode(&event));
        }
    }
    events
}

fn summarize(event: &ModelEvent) -> String {
    match event {
        ModelEvent::TextDelta { delta } => format!("text:{delta}"),
        ModelEvent::ReasoningDelta { delta } => format!("reasoning:{delta}"),
        ModelEvent::ReasoningCompleted { .. } => "reasoning_completed".to_string(),
        ModelEvent::ToolCallArgumentsDelta {
            call_id,
            tool,
            delta,
        } => {
            format!("arguments:{tool}:{call_id}:{delta}")
        }
        ModelEvent::ToolCallCompleted { call } => format!("call:{}:{}", call.tool, call.arguments),
        ModelEvent::Usage { usage } => {
            format!("usage:{}:{}", usage.input_tokens, usage.output_tokens)
        }
        ModelEvent::Completed => "completed".to_string(),
        ModelEvent::Error { message } => format!("error:{message}"),
    }
}

fn item(id: u64, turn: u64, payload: ItemPayload) -> Item {
    Item {
        id: ItemId(id),
        parent_id: id.checked_sub(1).map(ItemId),
        turn_id: TurnId(turn),
        created_at_ms: 0,
        status: CompletionStatus::Completed,
        error: None,
        payload,
    }
}

fn tool_call(id: u64, turn: u64, call_id: &str) -> Item {
    item(
        id,
        turn,
        ItemPayload::ToolCall {
            tool: "read_file".to_string(),
            call_id: call_id.to_string(),
            arguments: json!({ "path": "README.md" }),
            cwd: "/fixture".into(),
            revision: WorkspaceRevision(0),
        },
    )
}

fn tool_result(id: u64, turn: u64, call_id: &str) -> Item {
    item(
        id,
        turn,
        ItemPayload::ToolResult {
            call_id: call_id.to_string(),
            content: "# fixture".to_string(),
            duration_ms: 3,
            truncated: false,
        },
    )
}

fn request(items: Vec<Item>) -> Value {
    request_for(Capabilities::OPENAI, items)
}

fn request_for(capabilities: Capabilities, items: Vec<Item>) -> Value {
    build_request(
        "test-model",
        "be helpful",
        capabilities,
        &TurnRequest {
            session_id: SessionId("s0".to_string()),
            items,
            tools: vec![ToolSpec {
                name: "read_file".to_string(),
                description: "read a file".to_string(),
                parameters: json!({ "type": "object" }),
            }],
            sampling: Sampling::default(),
        },
    )
}

#[test]
fn a_reasoning_and_tool_call_stream_decodes_in_order() {
    let events = decode_stream(TOOL_CALL_STREAM);
    let summary: Vec<String> = events.iter().map(summarize).collect();
    assert_eq!(
        summary,
        [
            "reasoning:Check ",
            "reasoning:the file.",
            "reasoning_completed",
            "arguments:read_file:call_1:{\"path\":",
            "arguments:read_file:call_1:\"README.md\"}",
            "call:read_file:{\"path\":\"README.md\"}",
            "usage:120:34",
            "completed",
        ]
    );

    // The reasoning item is carried through opaquely, minus the id: a
    // stateless response has no server-side item to point back at.
    let ModelEvent::ReasoningCompleted { provider_data } = &events[2] else {
        panic!("expected reasoning_completed");
    };
    assert_eq!(provider_data["encrypted_content"], "enc-blob");
    assert_eq!(provider_data["type"], "reasoning");
    assert!(provider_data.get("id").is_none());
}

#[test]
fn text_output_and_terminal_events_decode() {
    let stream = "\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"All \"}\r\n\
\r\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"done.\"}\r\n\
\r\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\"}}\r\n\
\r\n";
    let summary: Vec<String> = decode_stream(stream).iter().map(summarize).collect();
    // Carriage returns are framing, and a response with no usage block reports
    // none rather than zero.
    assert_eq!(summary, ["text:All ", "text:done.", "completed"]);
}

#[test]
fn failure_events_become_model_errors() {
    let failed = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\"}}}\n\n";
    assert_eq!(
        decode_stream(failed)
            .iter()
            .map(summarize)
            .collect::<Vec<_>>(),
        ["error:slow down"]
    );

    let incomplete = "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n";
    assert_eq!(
        decode_stream(incomplete)
            .iter()
            .map(summarize)
            .collect::<Vec<_>>(),
        ["error:response incomplete: max_output_tokens"]
    );

    let bad_arguments = "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_9\",\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"search\",\"arguments\":\"{not json\"}}\n\n";
    let events = decode_stream(bad_arguments);
    assert!(matches!(events[0], ModelEvent::Error { .. }));
}

#[test]
fn unknown_events_and_partial_frames_decode_to_nothing() {
    let mut frames = SseBuffer::default();
    // A frame is held until its terminator arrives.
    assert!(
        frames
            .push(b"data: {\"type\":\"response.created\"")
            .is_empty()
    );
    assert_eq!(
        frames.push(b",\"response\":{}}\n\n"),
        vec!["{\"type\":\"response.created\",\"response\":{}}".to_string()]
    );

    let mut decoder = Decoder::default();
    let created = json!({ "type": "response.created", "response": {} });
    assert!(decoder.decode(&created).is_empty());
    let unknown = json!({ "type": "response.some_future_event" });
    assert!(decoder.decode(&unknown).is_empty());
}

#[test]
fn requests_are_stateless_and_ask_for_encrypted_reasoning() {
    let body = request(vec![item(
        0,
        0,
        ItemPayload::UserMessage {
            text: "hello".to_string(),
        },
    )]);
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["instructions"], "be helpful");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(
        body["tools"],
        json!([{
            "type": "function",
            "name": "read_file",
            "description": "read a file",
            "parameters": { "type": "object" },
        }])
    );
    assert_eq!(
        body["input"],
        json!([{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "hello" }],
        }])
    );
}

#[test]
fn a_completed_turn_replays_reasoning_and_paired_calls() {
    let reasoning = json!({
        "type": "reasoning",
        "summary": [{ "type": "summary_text", "text": "Check the file." }],
        "encrypted_content": "enc-blob",
    });
    let body = request(vec![
        item(
            0,
            0,
            ItemPayload::UserMessage {
                text: "read it".to_string(),
            },
        ),
        item(
            1,
            0,
            ItemPayload::Reasoning {
                text: "Check the file.".to_string(),
                provider_data: Some(reasoning.clone()),
            },
        ),
        tool_call(2, 0, "call_1"),
        tool_result(3, 0, "call_1"),
        item(
            4,
            0,
            ItemPayload::AssistantMessage {
                text: "It is a fixture.".to_string(),
            },
        ),
    ]);
    assert_eq!(
        body["input"],
        json!([
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "read it" }] },
            reasoning,
            {
                "type": "function_call",
                "name": "read_file",
                "call_id": "call_1",
                // Responses carries arguments as a JSON string, not an object.
                "arguments": "{\"path\":\"README.md\"}",
            },
            { "type": "function_call_output", "call_id": "call_1", "output": "# fixture" },
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "It is a fixture." }] },
        ])
    );
}

#[test]
fn a_cancelled_turn_replays_without_the_items_the_provider_rejects() {
    let mut dangling_call = tool_call(2, 0, "call_1");
    dangling_call.status = CompletionStatus::Failed;
    let mut empty_message = item(
        3,
        0,
        ItemPayload::AssistantMessage {
            text: String::new(),
        },
    );
    empty_message.status = CompletionStatus::Cancelled;

    let body = request(vec![
        item(
            0,
            0,
            ItemPayload::UserMessage {
                text: "read it".to_string(),
            },
        ),
        item(
            1,
            0,
            ItemPayload::Reasoning {
                text: "Check the file.".to_string(),
                provider_data: Some(json!({ "type": "reasoning", "encrypted_content": "enc" })),
            },
        ),
        // Arguments that never finished streaming: no result was ever
        // recorded for this call.
        dangling_call,
        empty_message,
    ]);
    // The unpaired call, the reasoning that led to it, and the empty assistant
    // message are all dropped, leaving a request the provider accepts.
    assert_eq!(
        body["input"],
        json!([{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "read it" }],
        }])
    );
}

#[test]
fn a_cancelled_call_still_answers_the_model() {
    let mut cancelled = tool_result(3, 0, "call_1");
    cancelled.status = CompletionStatus::Cancelled;
    cancelled.payload = ItemPayload::ToolResult {
        call_id: "call_1".to_string(),
        content: String::new(),
        duration_ms: 0,
        truncated: false,
    };
    let body = request(vec![
        item(
            0,
            0,
            ItemPayload::UserMessage {
                text: "read it".to_string(),
            },
        ),
        tool_call(1, 0, "call_1"),
        cancelled,
    ]);
    assert_eq!(
        body["input"][2],
        json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "call cancelled before it ran",
        })
    );
}

/// A target that cannot replay encrypted reasoning is neither asked for it nor
/// sent it back. vLLM raises "Encrypted content is not supported." on a
/// replayed reasoning item that carries any, and provider data is opaque, so
/// the codec drops every reasoning item rather than inspect the blobs.
#[test]
fn a_target_without_encrypted_reasoning_neither_asks_for_it_nor_replays_it() {
    let items = vec![
        item(
            0,
            0,
            ItemPayload::UserMessage {
                text: "read it".to_string(),
            },
        ),
        item(
            1,
            0,
            ItemPayload::Reasoning {
                text: "Check the file.".to_string(),
                provider_data: Some(json!({ "type": "reasoning", "encrypted_content": "enc" })),
            },
        ),
        tool_call(2, 0, "call_1"),
        tool_result(3, 0, "call_1"),
    ];

    let body = request_for(Capabilities::VLLM, items.clone());
    assert!(body.get("include").is_none(), "{body}");
    assert_eq!(
        body["input"],
        json!([
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "read it" }] },
            {
                "type": "function_call",
                "name": "read_file",
                "call_id": "call_1",
                "arguments": "{\"path\":\"README.md\"}",
            },
            { "type": "function_call_output", "call_id": "call_1", "output": "# fixture" },
        ])
    );

    // Everything else about the request is the one shared codec's output.
    let openai = request_for(Capabilities::OPENAI, items);
    assert_eq!(openai["store"], false);
    assert_eq!(openai["tools"], body["tools"]);
    assert_eq!(openai["input"].as_array().unwrap().len(), 4);
}

#[test]
fn reasoning_without_provider_data_is_not_replayed() {
    // The fake model produces reasoning with nothing to replay; sending it as
    // plain text would not be the provider's own record.
    let body = request(vec![
        item(
            0,
            0,
            ItemPayload::UserMessage {
                text: "hi".to_string(),
            },
        ),
        item(
            1,
            0,
            ItemPayload::Reasoning {
                text: "thinking".to_string(),
                provider_data: None,
            },
        ),
        item(
            2,
            0,
            ItemPayload::AssistantMessage {
                text: "hello".to_string(),
            },
        ),
    ]);
    assert_eq!(body["input"].as_array().unwrap().len(), 2);
}

#[test]
fn usage_sums_the_way_the_engine_reports_it() {
    let mut total = Usage::default();
    total.add(Usage {
        input_tokens: 120,
        output_tokens: 34,
    });
    total.add(Usage {
        input_tokens: 200,
        output_tokens: 12,
    });
    assert_eq!(
        total,
        Usage {
            input_tokens: 320,
            output_tokens: 46
        }
    );
}

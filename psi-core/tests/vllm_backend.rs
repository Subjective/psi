//! The vLLM backend over a local server (Milestone 7): an end-to-end streaming
//! tool call, the zero-tool-call configuration guard, cancellation by dropping
//! the stream, and the capability-gated request body on the wire. Plus an
//! ignored live smoke test against a real vLLM server.
//!
//! The scripted streams are shaped the way vLLM emits them: reasoning arrives
//! as `response.reasoning_text.delta` and closes with a reasoning item holding
//! `content` and no `encrypted_content`, and a function call arrives as
//! `response.output_item.added`, argument deltas, then `response.output_item.done`.

use std::sync::Arc;
use std::time::Duration;

use psi_core::hook::HookRegistry;
use psi_core::item::{CompletionStatus, ItemPayload};
use psi_core::model::{ModelBackend, ModelEvent, Sampling, TurnRequest};
use psi_core::protocol::{Command, EventPayload};
use psi_core::session::SessionId;
use psi_core::tool::ToolSpec;
use psi_core::tools::default_profile;
use psi_core::vllm::{VllmBackend, VllmConfig};
use psi_core::{Harness, HarnessConfig};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

const TOOL_CALL_STREAM: &str = "\
event: response.created
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}

event: response.output_item.added
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"reasoning\",\"summary\":[],\"status\":\"in_progress\"}}

event: response.reasoning_text.delta
data: {\"type\":\"response.reasoning_text.delta\",\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"Read it.\"}

event: response.output_item.done
data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"Read it.\"}],\"status\":\"completed\"}}

event: response.output_item.added
data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"\",\"status\":\"in_progress\"}}

event: response.function_call_arguments.delta
data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"{\\\"path\\\":\\\"README.md\\\"}\"}

event: response.function_call_arguments.done
data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}

event: response.output_item.done
data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\",\"status\":\"completed\"}}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":118,\"output_tokens\":21,\"total_tokens\":139}}}

";

/// What a server with no tool parser sends back: the model's tool call, but as
/// ordinary output text and with no function call item anywhere.
const UNPARSED_STREAM: &str = "\
event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"<tool_call>{\\\"name\\\": \\\"read_file\\\"}</tool_call>\"}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":90,\"output_tokens\":14}}}

";

const TEXT_STREAM: &str = "\
event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"Done.\"}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"status\":\"completed\"}}

";

/// Answers `responses.len()` requests with the canned bodies in order, and
/// hands back every request it read. When `hold` is set the last socket stays
/// open after its body is written, so the test can watch what the client does
/// next; the returned channel fires once that client hangs up.
async fn serve(responses: Vec<String>, hold: bool) -> (String, oneshot::Receiver<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    let (done_tx, done_rx) = oneshot::channel();
    tokio::spawn(async move {
        let mut requests = Vec::new();
        let last = responses.len() - 1;
        for (index, response) in responses.into_iter().enumerate() {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            // The body follows the headers in the same stream; every request
            // here declares a content-length, so read until it is satisfied.
            loop {
                match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(read) => request.extend_from_slice(&chunk[..read]),
                }
                if request_is_complete(&request) {
                    break;
                }
            }
            requests.push(String::from_utf8_lossy(&request).into_owned());
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
            if hold && index == last {
                // Resolves with zero bytes once the client closes the connection.
                let _ = socket.read(&mut chunk).await;
            }
        }
        let _ = done_tx.send(requests);
    });
    (base_url, done_rx)
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(head_end) = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
    else {
        return false;
    };
    let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
    let length: usize = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0);
    request.len() >= head_end + length
}

fn sse_response(body: &str) -> String {
    format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{body}")
}

fn backend(base_url: String) -> VllmBackend {
    VllmBackend::new(VllmConfig {
        base_url,
        model: "Qwen/Qwen3-8B".to_string(),
        request_timeout: Duration::from_secs(5),
        idle_timeout: Duration::from_secs(5),
        ..VllmConfig::default()
    })
    .expect("backend")
}

fn request(tools: Vec<ToolSpec>) -> TurnRequest {
    TurnRequest {
        session_id: SessionId("s0".to_string()),
        items: Vec::new(),
        tools,
        sampling: Sampling::default(),
    }
}

fn read_file_tool() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: "read_file".to_string(),
        description: "read a file".to_string(),
        parameters: json!({ "type": "object" }),
    }]
}

async fn drain(events: &mut mpsc::Receiver<ModelEvent>) -> Vec<String> {
    let mut collected = Vec::new();
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
        collected.push(match event {
            ModelEvent::TextDelta { delta } => format!("text:{delta}"),
            ModelEvent::ReasoningDelta { delta } => format!("reasoning:{delta}"),
            ModelEvent::ReasoningCompleted { .. } => "reasoning_completed".to_string(),
            ModelEvent::ToolCallArgumentsDelta { tool, delta, .. } => {
                format!("arguments:{tool}:{delta}")
            }
            ModelEvent::ToolCallCompleted { call } => {
                format!("call:{}:{}", call.tool, call.arguments)
            }
            ModelEvent::Usage { usage } => {
                format!("usage:{}:{}", usage.input_tokens, usage.output_tokens)
            }
            ModelEvent::Completed => "completed".to_string(),
            ModelEvent::Error { message } => format!("error:{message}"),
        });
    }
    collected
}

#[tokio::test]
async fn a_streamed_tool_call_reaches_the_model_boundary() {
    let (base_url, _requests) = serve(vec![sse_response(TOOL_CALL_STREAM)], false).await;
    let backend = backend(base_url);

    let mut events = backend.stream_response(request(read_file_tool()));
    assert_eq!(
        drain(&mut events).await,
        [
            "reasoning:Read it.",
            "reasoning_completed",
            "arguments:read_file:{\"path\":\"README.md\"}",
            "call:read_file:{\"path\":\"README.md\"}",
            "usage:118:21",
            "completed",
        ]
    );
}

/// The first guard: a server started without `--enable-auto-tool-choice` and a
/// matching `--tool-call-parser` returns the tool call as text and no calls.
#[tokio::test]
async fn a_response_with_no_tool_calls_is_a_configuration_error() {
    let (base_url, _requests) = serve(vec![sse_response(UNPARSED_STREAM)], false).await;
    let backend = backend(base_url);

    let mut events = backend.stream_response(request(read_file_tool()));
    let collected = drain(&mut events).await;
    let error = collected.last().expect("an event");
    assert!(
        error.starts_with("error:vllm") && error.contains("--tool-call-parser"),
        "{collected:?}"
    );
    // The text and usage still arrive; only the terminal event changes, so the
    // turn fails loudly instead of completing as a plain answer.
    assert!(collected.iter().any(|event| event.starts_with("text:")));
    assert!(!collected.iter().any(|event| event == "completed"));
}

/// Once the target has completed one tool call the parser is proven present,
/// and a later response that answers without calling anything is honest.
#[tokio::test]
async fn a_target_that_has_called_a_tool_may_then_answer_without_one() {
    let (base_url, _requests) = serve(
        vec![sse_response(TOOL_CALL_STREAM), sse_response(TEXT_STREAM)],
        false,
    )
    .await;
    let backend = backend(base_url);

    let mut events = backend.stream_response(request(read_file_tool()));
    assert!(drain(&mut events).await.contains(&"completed".to_string()));

    let mut events = backend.stream_response(request(read_file_tool()));
    assert_eq!(drain(&mut events).await, ["text:Done.", "completed"]);
}

#[tokio::test]
async fn a_request_that_advertised_no_tools_is_never_a_configuration_error() {
    let (base_url, _requests) = serve(vec![sse_response(TEXT_STREAM)], false).await;
    let backend = backend(base_url);

    let mut events = backend.stream_response(request(Vec::new()));
    assert_eq!(drain(&mut events).await, ["text:Done.", "completed"]);
}

/// The second guard: cancellation is the connection going away. vLLM's
/// `/v1/responses/{id}/cancel` refuses a synchronous response, so there is
/// nothing else for the backend to do.
#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    let opening = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Read\"}\n\n";
    let (base_url, closed) = serve(vec![sse_response(opening)], true).await;
    let backend = backend(base_url);

    let mut events = backend.stream_response(request(read_file_tool()));
    assert!(matches!(
        events.recv().await,
        Some(ModelEvent::TextDelta { .. })
    ));
    drop(events);

    tokio::time::timeout(Duration::from_secs(5), closed)
        .await
        .expect("the request was not cancelled")
        .expect("server task");
}

#[tokio::test]
async fn the_request_is_keyless_and_asks_for_nothing_vllm_rejects() {
    let (base_url, requests) = serve(vec![sse_response(TOOL_CALL_STREAM)], false).await;
    // Naming a variable nobody sets is how the keyless case is reached without
    // mutating the environment out from under the other tests in this binary.
    let backend = VllmBackend::new(VllmConfig {
        base_url,
        api_key_env: "PSI_TEST_VLLM_KEY_THAT_IS_NEVER_SET".to_string(),
        ..VllmConfig::default()
    })
    .expect("backend");

    let mut events = backend.stream_response(request(read_file_tool()));
    drain(&mut events).await;

    let requests = tokio::time::timeout(Duration::from_secs(5), requests)
        .await
        .expect("server task")
        .expect("server task");
    let sent = &requests[0];
    assert!(
        !sent.to_lowercase().contains("authorization:"),
        "a keyless server was sent credentials: {sent}"
    );
    let body: Value = serde_json::from_str(sent.split("\r\n\r\n").nth(1).expect("body")).unwrap();
    assert!(body.get("include").is_none(), "{body}");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    // An empty model name is how vLLM is told to use the model it loaded.
    assert_eq!(body["model"], "");
}

/// A live end-to-end turn against a real vLLM server: run with
/// `cargo test -- --ignored` and `PSI_VLLM_BASE_URL` pointing at it, e.g.
/// `http://localhost:8000/v1`. `PSI_VLLM_MODEL` names the served model when
/// the server rejects an empty one. The server must have been started with
/// `--enable-auto-tool-choice` and a `--tool-call-parser` for its model, which
/// is exactly what the configuration guard checks.
#[tokio::test]
#[ignore = "needs PSI_VLLM_BASE_URL and a running vLLM server"]
async fn live_smoke_test_reads_a_fixture_file() {
    let base_url = std::env::var("PSI_VLLM_BASE_URL").expect("PSI_VLLM_BASE_URL");
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("ANSWER.txt"), "the answer is 42\n").unwrap();
    let workspace = dir.path().to_path_buf();

    let mut config = VllmConfig {
        base_url,
        ..VllmConfig::default()
    };
    if let Ok(model) = std::env::var("PSI_VLLM_MODEL") {
        config.model = model;
    }
    config.instructions = format!(
        "{}\n\nThe workspace root is {}.",
        config.instructions,
        workspace.display()
    );
    let backend = VllmBackend::new(config).expect("backend");

    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(backend),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    commands.send(Command::CreateSession).await.unwrap();
    let session_id = match events.recv().await.unwrap().payload {
        EventPayload::SessionCreated { meta } => meta.id,
        other => panic!("expected session_created, got {other:?}"),
    };
    commands
        .send(Command::SubmitMessage {
            session_id,
            text: "Read ANSWER.txt and reply with only the number it names.".to_string(),
        })
        .await
        .unwrap();

    let mut tool_results = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(180), events.recv())
            .await
            .expect("timed out")
            .expect("event channel closed");
        match event.payload {
            EventPayload::ItemFinished { item } => {
                if let ItemPayload::ToolResult { content, .. } = item.payload {
                    tool_results.push(content);
                }
            }
            EventPayload::TurnFinished { status, error, .. } => {
                assert_eq!(status, CompletionStatus::Completed, "{error:?}");
                break;
            }
            _ => {}
        }
    }
    assert!(
        tool_results.iter().any(|result| result.contains("42")),
        "the model never read the file: {tool_results:?}"
    );
}

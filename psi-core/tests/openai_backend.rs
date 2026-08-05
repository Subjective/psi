//! The OpenAI backend over a local server: streaming, cancellation by dropping
//! the stream, the idle timeout, and HTTP failures. Plus an ignored live smoke
//! test against the real API (Milestone 2).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use psi_core::Harness;
use psi_core::hook::HookRegistry;
use psi_core::item::{CompletionStatus, ItemPayload};
use psi_core::model::{ModelBackend, ModelEvent, TurnRequest};
use psi_core::openai::{OpenAiBackend, OpenAiConfig};
use psi_core::protocol::{Command, EventPayload};
use psi_core::session::SessionId;
use psi_core::tools::default_profile;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

const STREAM: &str = "\
event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}

";

/// Answers one request with a canned response. When `hold` is set the socket
/// stays open after the body is written, so the test can watch what the client
/// does next; the returned channel fires once the client hangs up.
async fn serve(response: String, hold: bool) -> (String, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    let (closed_tx, closed_rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            match socket.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(read) => request.extend_from_slice(&chunk[..read]),
            }
        }
        let _ = socket.write_all(response.as_bytes()).await;
        let _ = socket.flush().await;
        if hold {
            // Resolves with zero bytes once the client closes the connection.
            let _ = socket.read(&mut chunk).await;
        }
        let _ = closed_tx.send(());
    });
    (base_url, closed_rx)
}

fn sse_response(body: &str) -> String {
    format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{body}")
}

fn backend(base_url: String, idle_timeout: Duration) -> OpenAiBackend {
    OpenAiBackend::with_api_key(
        OpenAiConfig {
            base_url,
            idle_timeout,
            request_timeout: Duration::from_secs(5),
            ..OpenAiConfig::default()
        },
        "test-key".to_string(),
    )
    .expect("backend")
}

fn empty_request() -> TurnRequest {
    TurnRequest {
        session_id: SessionId("s0".to_string()),
        items: Vec::new(),
        tools: Vec::new(),
    }
}

async fn drain(events: &mut mpsc::Receiver<ModelEvent>) -> Vec<String> {
    let mut collected = Vec::new();
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
        collected.push(match event {
            ModelEvent::TextDelta { delta } => format!("text:{delta}"),
            ModelEvent::Usage { usage } => {
                format!("usage:{}:{}", usage.input_tokens, usage.output_tokens)
            }
            ModelEvent::Completed => "completed".to_string(),
            ModelEvent::Error { message } => format!("error:{message}"),
            other => format!("{other:?}"),
        });
    }
    collected
}

#[tokio::test]
async fn a_streamed_response_reaches_the_model_boundary() {
    let (base_url, _closed) = serve(sse_response(STREAM), false).await;
    let backend = backend(base_url, Duration::from_secs(5));

    let mut events = backend.stream_response(empty_request());
    assert_eq!(
        drain(&mut events).await,
        ["text:Hello", "usage:5:2", "completed"]
    );
}

#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    // The server writes one event and then holds the socket open, so the
    // backend is parked waiting for more when the receiver goes away.
    let opening = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n";
    let (base_url, closed) = serve(sse_response(opening), true).await;
    let backend = backend(base_url, Duration::from_secs(30));

    let mut events = backend.stream_response(empty_request());
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
async fn a_silent_stream_times_out() {
    let (base_url, _closed) = serve(sse_response(""), true).await;
    let backend = backend(base_url, Duration::from_millis(200));

    let mut events = backend.stream_response(empty_request());
    let collected = drain(&mut events).await;
    assert_eq!(collected.len(), 1);
    assert!(collected[0].starts_with("error:openai stream idle for"));
}

#[tokio::test]
async fn a_stream_that_ends_early_is_never_success() {
    // No terminal event: the body just stops. The backend reports nothing and
    // the engine turns the silence into a failed turn.
    let partial = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n";
    let (base_url, _closed) = serve(sse_response(partial), false).await;
    let backend = backend(base_url, Duration::from_secs(5));

    let mut events = backend.stream_response(empty_request());
    assert_eq!(drain(&mut events).await, ["text:Hel"]);
}

#[tokio::test]
async fn an_http_failure_becomes_a_model_error() {
    let body = "{\"error\":{\"message\":\"slow down\"}}";
    let response = format!(
        "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let (base_url, _closed) = serve(response, false).await;
    let backend = backend(base_url, Duration::from_secs(5));

    let mut events = backend.stream_response(empty_request());
    let collected = drain(&mut events).await;
    assert_eq!(collected.len(), 1);
    assert!(
        collected[0].contains("429") && collected[0].contains("slow down"),
        "{collected:?}"
    );
}

/// A live end-to-end turn: run with `cargo test -- --ignored` and an
/// `OPENAI_API_KEY` in the environment. `PSI_MODEL` overrides the model.
#[tokio::test]
#[ignore = "needs OPENAI_API_KEY and network"]
async fn live_smoke_test_reads_a_fixture_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("ANSWER.txt"), "the answer is 42\n").unwrap();
    let workspace = dir.path().to_path_buf();

    let mut config = OpenAiConfig::default();
    if let Ok(model) = std::env::var("PSI_MODEL") {
        config.model = model;
    }
    config.instructions = format!(
        "{}\n\nThe workspace root is {}.",
        config.instructions,
        workspace.display()
    );
    let backend = OpenAiBackend::new(config).expect("OPENAI_API_KEY");

    let (commands, mut events) = Harness::spawn(
        Arc::new(backend),
        default_profile(workspace.clone()),
        HookRegistry::new(),
        workspace,
    );
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
    let mut reply = String::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(180), events.recv())
            .await
            .expect("timed out")
            .expect("event channel closed");
        match event.payload {
            EventPayload::ItemFinished { item } => match item.payload {
                ItemPayload::ToolResult { content, .. } => tool_results.push(content),
                ItemPayload::AssistantMessage { text } => reply.push_str(&text),
                _ => {}
            },
            EventPayload::TurnFinished {
                status,
                error,
                usage,
                ..
            } => {
                assert_eq!(status, CompletionStatus::Completed, "{error:?}");
                assert!(usage.expect("usage reported").input_tokens > 0);
                break;
            }
            _ => {}
        }
    }
    assert!(
        tool_results.iter().any(|result| result.contains("42")),
        "the model never read the file: {tool_results:?}"
    );
    assert!(reply.contains("42"), "unexpected reply: {reply}");
}

/// The default profile has to be constructible for a workspace that does not
/// exist yet, because the binary builds it before touching the filesystem.
#[test]
fn the_profile_builds_for_a_missing_workspace() {
    let tools = default_profile(PathBuf::from("/no/such/workspace"));
    assert_eq!(tools.specs().len(), 5);
}

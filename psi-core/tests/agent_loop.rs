//! Milestone 2's verification: the headless agent inspects a fixture
//! repository, changes a file, runs a test and finishes; a structured-tool
//! access outside the workspace root is refused; and a blocking hook surfaces
//! to the model as a refused call.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use psi_core::fake::{FakeModel, FakeResponse};
use psi_core::hook::{Hook, HookDecision, HookRegistry};
use psi_core::item::{CompletionStatus, ItemPayload};
use psi_core::model::{ModelEvent, ToolCallRequest};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::{SessionId, SessionSnapshot};
use psi_core::tool::{ToolInvocation, ToolOutput};
use psi_core::tools::default_profile;
use psi_core::{Harness, HarnessConfig};
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// A fixture repository the agent can inspect, edit, and test: a shell script
/// standing in for a test suite, which passes only after the fix.
fn fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.sh"), "answer() {\n  echo 41\n}\n").unwrap();
    fs::write(
        root.join("test.sh"),
        ". ./src/lib.sh\n[ \"$(answer)\" = \"42\" ] || { echo 'FAIL: expected 42'; exit 1; }\necho 'PASS'\n",
    )
    .unwrap();
    dir
}

fn call(tool: &str, call_id: &str, arguments: serde_json::Value) -> ModelEvent {
    ModelEvent::ToolCallCompleted {
        call: ToolCallRequest {
            call_id: call_id.into(),
            tool: tool.into(),
            arguments,
        },
    }
}

fn one_call(tool: &str, call_id: &str, arguments: serde_json::Value) -> FakeResponse {
    FakeResponse::new(vec![call(tool, call_id, arguments), ModelEvent::Completed])
}

fn text(message: &str) -> FakeResponse {
    FakeResponse::new(vec![
        ModelEvent::TextDelta {
            delta: message.into(),
        },
        ModelEvent::Completed,
    ])
}

async fn recv(events: &mut mpsc::Receiver<Event>) -> Event {
    tokio::time::timeout(Duration::from_secs(10), events.recv())
        .await
        .expect("timed out waiting for event")
        .expect("event channel closed")
}

async fn create_session(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
) -> SessionId {
    commands.send(Command::CreateSession).await.unwrap();
    match recv(events).await.payload {
        EventPayload::SessionCreated { meta } => meta.id,
        other => panic!("expected session_created, got {other:?}"),
    }
}

async fn run_turn(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
    session_id: &SessionId,
    prompt: &str,
) -> CompletionStatus {
    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: prompt.into(),
        })
        .await
        .unwrap();
    loop {
        if let EventPayload::TurnFinished { status, .. } = recv(events).await.payload {
            return status;
        }
    }
}

async fn snapshot(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
    session_id: &SessionId,
) -> SessionSnapshot {
    commands
        .send(Command::LoadSession {
            session_id: session_id.clone(),
        })
        .await
        .unwrap();
    match recv(events).await.payload {
        EventPayload::SessionLoaded { snapshot } => snapshot,
        other => panic!("expected session_loaded, got {other:?}"),
    }
}

/// Every tool_result in the session, as `(content, failed)`.
fn results(snapshot: &SessionSnapshot) -> Vec<(String, bool)> {
    snapshot
        .items
        .iter()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolResult { content, .. } => {
                Some((content.clone(), item.status == CompletionStatus::Failed))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn the_agent_inspects_edits_and_tests_a_fixture_repository() {
    let dir = fixture();
    let workspace = dir.path().to_path_buf();
    let model = FakeModel::new([
        one_call("list_directory", "call-1", json!({ "depth": 2 })),
        one_call("search", "call-2", json!({ "pattern": "echo 41" })),
        one_call("read_file", "call-3", json!({ "path": "src/lib.sh" })),
        one_call(
            "apply_patch",
            "call-4",
            json!({ "path": "src/lib.sh", "old_text": "echo 41", "new_text": "echo 42" }),
        ),
        one_call("exec", "call-5", json!({ "command": "sh test.sh" })),
        text("Fixed: answer now returns 42 and the test passes."),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace: workspace.clone(),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    let status = run_turn(&commands, &mut events, &session_id, "make the test pass").await;
    assert_eq!(status, CompletionStatus::Completed);

    let snapshot = snapshot(&commands, &mut events, &session_id).await;
    let results = results(&snapshot);
    assert!(
        results.iter().all(|(_, failed)| !failed),
        "a tool call failed: {results:?}"
    );
    assert_eq!(results[0].0, "src/\nsrc/lib.sh\ntest.sh");
    assert_eq!(results[1].0, "src/lib.sh:2:   echo 41");
    assert_eq!(results[3].0, "updated src/lib.sh");
    assert!(results[4].0.contains("PASS"), "{}", results[4].0);
    assert!(results[4].0.contains("[exit status: 0]"));

    // The edit landed on disk, and the revision advanced once for the patch
    // and once for the exec that followed it.
    assert_eq!(
        fs::read_to_string(workspace.join("src/lib.sh")).unwrap(),
        "answer() {\n  echo 42\n}\n"
    );
    let revisions: Vec<u64> = snapshot
        .items
        .iter()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolCall { revision, .. } => Some(revision.0),
            _ => None,
        })
        .collect();
    assert_eq!(revisions, vec![0, 0, 0, 0, 1]);

    match &snapshot.items.last().unwrap().payload {
        ItemPayload::AssistantMessage { text } => {
            assert_eq!(text, "Fixed: answer now returns 42 and the test passes.")
        }
        other => panic!("expected the turn to end with an assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn a_read_outside_the_workspace_root_is_refused_to_the_model() {
    let dir = fixture();
    let workspace = dir.path().to_path_buf();
    let outside = workspace.parent().unwrap().join("outside-secret.txt");
    fs::write(&outside, "secret\n").unwrap();

    let model = FakeModel::new([
        one_call(
            "read_file",
            "call-1",
            json!({ "path": "../outside-secret.txt" }),
        ),
        text("I cannot read outside the workspace."),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    let status = run_turn(&commands, &mut events, &session_id, "read the secret").await;
    assert_eq!(status, CompletionStatus::Completed);

    // The refusal is a failed tool_result the model reads and answers around;
    // the turn itself is unharmed.
    let snapshot = snapshot(&commands, &mut events, &session_id).await;
    let (content, failed) = results(&snapshot).remove(0);
    assert!(failed);
    assert!(content.contains("escapes the workspace root"), "{content}");
    assert!(!content.contains("secret\n"));

    fs::remove_file(outside).unwrap();
}

/// Blocks one tool by name and counts the edges it sees, so the test can
/// assert that a blocked call never reaches the tool or the after-hook.
struct BlockTool {
    blocked: &'static str,
    before_calls: Arc<AtomicUsize>,
    after_calls: Arc<Mutex<Vec<String>>>,
}

impl Hook for BlockTool {
    fn before(&self, tool: &str, _invocation: &ToolInvocation) -> HookDecision {
        self.before_calls.fetch_add(1, Ordering::SeqCst);
        if tool == self.blocked {
            HookDecision::Block {
                reason: "exec is not allowed in this harness".to_string(),
            }
        } else {
            HookDecision::Continue
        }
    }

    fn after(&self, tool: &str, _invocation: &ToolInvocation, output: &ToolOutput) {
        self.after_calls
            .lock()
            .unwrap()
            .push(format!("{tool}:{}", output.error.is_none()));
    }
}

#[tokio::test]
async fn a_blocking_hook_surfaces_to_the_model_as_a_refused_call() {
    let dir = fixture();
    let workspace = dir.path().to_path_buf();
    let before_calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(Mutex::new(Vec::new()));

    let model = FakeModel::new([
        one_call("read_file", "call-1", json!({ "path": "test.sh" })),
        one_call("exec", "call-2", json!({ "command": "rm -rf /" })),
        text("I was not allowed to run that."),
    ]);
    let mut hooks = HookRegistry::new();
    hooks.register(BlockTool {
        blocked: "exec",
        before_calls: before_calls.clone(),
        after_calls: after_calls.clone(),
    });
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks,
        workspace,
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    let status = run_turn(&commands, &mut events, &session_id, "clean up").await;
    assert_eq!(status, CompletionStatus::Completed);

    let snapshot = snapshot(&commands, &mut events, &session_id).await;
    let results = results(&snapshot);
    assert!(!results[0].1, "the allowed call ran: {results:?}");
    let (refusal, failed) = &results[1];
    assert!(failed);
    assert_eq!(refusal, "exec refused: exec is not allowed in this harness");

    // Both calls reached a before-hook; only the one that ran reached the
    // after-hook, and the blocked one never touched the workspace revision.
    assert_eq!(before_calls.load(Ordering::SeqCst), 2);
    assert_eq!(*after_calls.lock().unwrap(), vec!["read_file:true"]);
    let revisions: Vec<u64> = snapshot
        .items
        .iter()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolCall { revision, .. } => Some(revision.0),
            _ => None,
        })
        .collect();
    assert_eq!(revisions, vec![0, 0]);
}

#[tokio::test]
async fn streamed_tool_call_arguments_reach_the_event_stream() {
    let model = FakeModel::new([
        FakeResponse::new(vec![
            ModelEvent::ToolCallArgumentsDelta {
                call_id: "call-1".into(),
                tool: "list_directory".into(),
                delta: "{\"depth\":".into(),
            },
            ModelEvent::ToolCallArgumentsDelta {
                call_id: "call-1".into(),
                tool: "list_directory".into(),
                delta: "1}".into(),
            },
            call("list_directory", "call-1", json!({ "depth": 1 })),
            ModelEvent::Completed,
        ]),
        text("Listed."),
    ]);
    let dir = fixture();
    let workspace = dir.path().to_path_buf();
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: "list it".into(),
        })
        .await
        .unwrap();

    let mut sequence = Vec::new();
    loop {
        let event = recv(&mut events).await;
        let done = matches!(event.payload, EventPayload::TurnFinished { .. });
        sequence.push(match event.payload {
            EventPayload::ItemStarted { kind, .. } => format!("started:{kind}"),
            EventPayload::ItemDelta { delta, .. } => format!("delta:{delta}"),
            EventPayload::ItemFinished { item } => format!("finished:{}", item.payload.kind()),
            EventPayload::TurnFinished { status, .. } => format!("turn:{status}"),
            _ => continue,
        });
        if done {
            break;
        }
    }
    // The arguments stream under the tool_call item's own id, so a client can
    // render a call while it is still being written.
    assert_eq!(
        &sequence[2..6],
        [
            "started:tool_call",
            "delta:{\"depth\":",
            "delta:1}",
            "finished:tool_call",
        ]
    );
}

#[tokio::test]
async fn a_response_that_dies_mid_arguments_leaves_no_open_item() {
    let model = FakeModel::new([FakeResponse::new(vec![
        ModelEvent::ToolCallArgumentsDelta {
            call_id: "call-1".into(),
            tool: "read_file".into(),
            delta: "{\"path\":".into(),
        },
        ModelEvent::Error {
            message: "connection reset".into(),
        },
    ])]);
    let dir = fixture();
    let workspace = dir.path().to_path_buf();
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    let status = run_turn(&commands, &mut events, &session_id, "read it").await;
    assert_eq!(status, CompletionStatus::Failed);

    // The half-written call is a failed record with no arguments, and no tool
    // ran for it.
    let snapshot = snapshot(&commands, &mut events, &session_id).await;
    let call = snapshot.items.last().unwrap();
    assert_eq!(call.status, CompletionStatus::Failed);
    match &call.payload {
        ItemPayload::ToolCall {
            tool, arguments, ..
        } => {
            assert_eq!(tool, "read_file");
            assert!(arguments.is_null());
        }
        other => panic!("expected a tool_call, got {other:?}"),
    }
    assert!(results(&snapshot).is_empty());
}

#[tokio::test]
async fn reasoning_provider_data_survives_onto_the_item() {
    let model = FakeModel::new([FakeResponse::new(vec![
        ModelEvent::ReasoningDelta {
            delta: "Thinking.".into(),
        },
        ModelEvent::ReasoningCompleted {
            provider_data: json!({ "type": "reasoning", "encrypted_content": "enc-blob" }),
        },
        ModelEvent::TextDelta {
            delta: "Done.".into(),
        },
        ModelEvent::Usage {
            usage: psi_core::model::Usage {
                input_tokens: 10,
                output_tokens: 4,
            },
        },
        ModelEvent::Completed,
    ])]);
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(PathBuf::from("/fixture")),
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
        speculation: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: "think".into(),
        })
        .await
        .unwrap();
    let usage = loop {
        if let EventPayload::TurnFinished { usage, .. } = recv(&mut events).await.payload {
            break usage;
        }
    };
    assert_eq!(usage.unwrap().input_tokens, 10);

    let snapshot = snapshot(&commands, &mut events, &session_id).await;
    match &snapshot.items[1].payload {
        ItemPayload::Reasoning {
            text,
            provider_data,
        } => {
            assert_eq!(text, "Thinking.");
            assert_eq!(
                provider_data.as_ref().unwrap()["encrypted_content"],
                "enc-blob"
            );
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
}

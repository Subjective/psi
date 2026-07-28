//! Headless tests driving the harness over the interface protocol, per
//! Milestone 1: a complete fake turn with its exact event sequence, forking
//! via set_head, cancellation, and workspace revision bumps.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use psi_core::fake::{FakeModel, FakeResponse, FakeTool};
use psi_core::hook::HookRegistry;
use psi_core::item::{CompletionStatus, ItemId, ItemPayload, TurnId};
use psi_core::model::{ModelEvent, ToolCallRequest};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::{SessionId, SessionSnapshot};
use psi_core::tool::{ToolEffect, ToolRegistry};
use psi_core::{Harness, HarnessConfig};
use serde_json::json;
use tokio::sync::mpsc;

fn text_response(text: &str) -> FakeResponse {
    FakeResponse::new(vec![
        ModelEvent::TextDelta { delta: text.into() },
        ModelEvent::Completed,
    ])
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

async fn recv(events: &mut mpsc::Receiver<Event>) -> Event {
    tokio::time::timeout(Duration::from_secs(5), events.recv())
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

async fn submit_and_collect(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
    session_id: &SessionId,
    text: &str,
) -> Vec<Event> {
    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: text.into(),
        })
        .await
        .unwrap();
    let mut collected = Vec::new();
    loop {
        let event = recv(events).await;
        let done = matches!(event.payload, EventPayload::TurnFinished { .. });
        collected.push(event);
        if done {
            break;
        }
    }
    collected
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

fn summarize(event: &Event) -> String {
    match &event.payload {
        EventPayload::TurnStarted { .. } => "turn_started".into(),
        EventPayload::ItemStarted { kind, .. } => format!("item_started:{kind}"),
        EventPayload::ItemDelta { .. } => "item_delta".into(),
        EventPayload::ItemFinished { item } => {
            format!("item_finished:{}:{}", item.payload.kind(), item.status)
        }
        EventPayload::TurnFinished { status, .. } => format!("turn_finished:{status}"),
        other => format!("{other:?}"),
    }
}

fn leaves(snapshot: &SessionSnapshot) -> Vec<ItemId> {
    let parents: Vec<ItemId> = snapshot.items.iter().filter_map(|i| i.parent_id).collect();
    snapshot
        .items
        .iter()
        .map(|i| i.id)
        .filter(|id| !parents.contains(id))
        .collect()
}

#[tokio::test]
async fn golden_fake_turn_event_sequence() {
    let model = FakeModel::new([
        FakeResponse::new(vec![
            ModelEvent::ReasoningDelta {
                delta: "Let me read that file.".into(),
            },
            call("read_file", "call-1", json!({ "path": "README.md" })),
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "All".into(),
            },
            ModelEvent::TextDelta {
                delta: " done.".into(),
            },
            ModelEvent::Completed,
        ]),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(FakeTool::canned(
        "read_file",
        ToolEffect::ReadOnly,
        "fake file contents",
    ));
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools,
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
    })
    .unwrap();

    let session_id = create_session(&commands, &mut events).await;
    let turn = submit_and_collect(&commands, &mut events, &session_id, "Read the readme").await;

    let sequence: Vec<String> = turn.iter().map(summarize).collect();
    assert_eq!(
        sequence,
        [
            "turn_started",
            "item_started:user_message",
            "item_finished:user_message:completed",
            "item_started:reasoning",
            "item_delta",
            "item_finished:reasoning:completed",
            "item_started:tool_call",
            "item_finished:tool_call:completed",
            "item_started:tool_result",
            "item_finished:tool_result:completed",
            "item_started:assistant_message",
            "item_delta",
            "item_delta",
            "item_finished:assistant_message:completed",
            "turn_finished:completed",
        ]
    );

    // Every event is stamped with the session and sequenced monotonically.
    assert!(
        turn.iter()
            .all(|e| e.session_id.as_ref() == Some(&session_id))
    );
    assert!(turn.windows(2).all(|pair| pair[0].seq < pair[1].seq));

    // The durable records carry the complete content.
    let snap = snapshot(&commands, &mut events, &session_id).await;
    assert_eq!(snap.items.len(), 5);
    match &snap.items[0].payload {
        ItemPayload::UserMessage { text } => assert_eq!(text, "Read the readme"),
        other => panic!("expected user_message, got {other:?}"),
    }
    match &snap.items[1].payload {
        ItemPayload::Reasoning { text, .. } => assert_eq!(text, "Let me read that file."),
        other => panic!("expected reasoning, got {other:?}"),
    }
    match &snap.items[2].payload {
        ItemPayload::ToolCall {
            tool,
            arguments,
            revision,
            ..
        } => {
            assert_eq!(tool, "read_file");
            assert_eq!(arguments, &json!({ "path": "README.md" }));
            assert_eq!(revision.0, 0);
        }
        other => panic!("expected tool_call, got {other:?}"),
    }
    match &snap.items[3].payload {
        ItemPayload::ToolResult { content, .. } => assert_eq!(content, "fake file contents"),
        other => panic!("expected tool_result, got {other:?}"),
    }
    match &snap.items[4].payload {
        ItemPayload::AssistantMessage { text } => assert_eq!(text, "All done."),
        other => panic!("expected assistant_message, got {other:?}"),
    }
    assert!(snap.items.iter().all(|i| i.turn_id == TurnId(0)));
    assert_eq!(snap.head, Some(snap.items[4].id));
}

#[tokio::test]
async fn set_head_forks_the_item_tree() {
    let model = FakeModel::new([
        text_response("one"),
        text_response("two"),
        text_response("three"),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: ToolRegistry::new(),
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    submit_and_collect(&commands, &mut events, &session_id, "first").await;
    submit_and_collect(&commands, &mut events, &session_id, "second").await;
    let snap = snapshot(&commands, &mut events, &session_id).await;
    // Linear so far: u0 -> a1 -> u2 -> a3.
    assert_eq!(snap.items.len(), 4);
    let first_assistant = snap.items[1].id;
    assert_eq!(snap.items[2].parent_id, Some(first_assistant));
    assert_eq!(snap.head, Some(snap.items[3].id));

    // Rewind to the first assistant reply and submit again: a fork.
    commands
        .send(Command::SetHead {
            session_id: session_id.clone(),
            item_id: Some(first_assistant),
        })
        .await
        .unwrap();
    submit_and_collect(&commands, &mut events, &session_id, "third").await;

    let snap = snapshot(&commands, &mut events, &session_id).await;
    assert_eq!(snap.items.len(), 6);
    let forked_user = &snap.items[4];
    assert_eq!(forked_user.parent_id, Some(first_assistant));
    // Both branch tips survive as leaves; head follows the new one.
    assert_eq!(leaves(&snap), vec![snap.items[3].id, snap.items[5].id]);
    assert_eq!(snap.head, Some(snap.items[5].id));
    // Turn grouping: each submit is one turn.
    assert_eq!(snap.items[4].turn_id, snap.items[5].turn_id);
    assert_ne!(snap.items[0].turn_id, snap.items[2].turn_id);
}

#[tokio::test]
async fn cancel_turn_preserves_partial_output() {
    let model = FakeModel::new([FakeResponse::hanging(vec![ModelEvent::TextDelta {
        delta: "partial".into(),
    }])]);
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: ToolRegistry::new(),
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: "stop me".into(),
        })
        .await
        .unwrap();
    // Consume through the streamed delta so the cancel lands mid-response.
    loop {
        let event = recv(&mut events).await;
        if matches!(event.payload, EventPayload::ItemDelta { .. }) {
            break;
        }
    }
    commands
        .send(Command::CancelTurn {
            session_id: session_id.clone(),
        })
        .await
        .unwrap();

    match recv(&mut events).await.payload {
        EventPayload::ItemFinished { item } => {
            assert_eq!(item.status, CompletionStatus::Cancelled);
            match &item.payload {
                ItemPayload::AssistantMessage { text } => assert_eq!(text, "partial"),
                other => panic!("expected assistant_message, got {other:?}"),
            }
        }
        other => panic!("expected item_finished, got {other:?}"),
    }
    match recv(&mut events).await.payload {
        EventPayload::TurnFinished { status, .. } => {
            assert_eq!(status, CompletionStatus::Cancelled)
        }
        other => panic!("expected turn_finished, got {other:?}"),
    }

    // The partial assistant message is durable.
    let snap = snapshot(&commands, &mut events, &session_id).await;
    assert_eq!(snap.items.len(), 2);
    assert_eq!(snap.items[1].status, CompletionStatus::Cancelled);
}

#[tokio::test]
async fn workspace_revision_bumps_after_exec() {
    let model = FakeModel::new([
        FakeResponse::new(vec![
            call("exec", "call-1", json!({ "command": "touch x" })),
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            call("read_file", "call-2", json!({ "path": "x" })),
            ModelEvent::Completed,
        ]),
        text_response("done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(FakeTool::canned("exec", ToolEffect::Unknown, "ok"));
    tools.register(FakeTool::canned(
        "read_file",
        ToolEffect::ReadOnly,
        "contents",
    ));
    let sessions = tempfile::tempdir().unwrap();
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools,
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.path().to_path_buf(),
        trace: None,
    })
    .unwrap();
    let session_id = create_session(&commands, &mut events).await;

    submit_and_collect(&commands, &mut events, &session_id, "touch then read").await;

    let snap = snapshot(&commands, &mut events, &session_id).await;
    let revisions: Vec<u64> = snap
        .items
        .iter()
        .filter_map(|i| match &i.payload {
            ItemPayload::ToolCall { revision, .. } => Some(revision.0),
            _ => None,
        })
        .collect();
    // exec runs at revision 0 and bumps it; the read that follows sees 1.
    assert_eq!(revisions, vec![0, 1]);
}

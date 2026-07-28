//! Milestone 3's verification: restarting Psi preserves the item tree, and
//! both sides of a fork resume independently. Every test drives a real harness
//! against a temporary sessions directory, drops it, and spawns a fresh one
//! over the same directory, so nothing but the JSONL files crosses a restart.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use psi_core::fake::{FakeModel, FakeResponse, FakeTool};
use psi_core::hook::HookRegistry;
use psi_core::item::{CompletionStatus, ItemId, ItemPayload};
use psi_core::model::{ModelEvent, ToolCallRequest};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::{SessionId, SessionMeta, SessionSnapshot};
use psi_core::tool::{ToolEffect, ToolRegistry};
use psi_core::{Harness, HarnessConfig};
use serde_json::json;
use tokio::sync::mpsc;

type Client = (mpsc::Sender<Command>, mpsc::Receiver<Event>);

fn text_response(text: &str) -> FakeResponse {
    FakeResponse::new(vec![
        ModelEvent::TextDelta { delta: text.into() },
        ModelEvent::Completed,
    ])
}

fn spawn_with(sessions: &Path, model: FakeModel, tools: ToolRegistry) -> Client {
    Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools,
        hooks: HookRegistry::new(),
        workspace: PathBuf::from("/fixture"),
        sessions_dir: sessions.to_path_buf(),
    })
    .unwrap()
}

/// A harness whose model answers each turn with one line of text.
fn spawn(sessions: &Path, script: impl IntoIterator<Item = &'static str>) -> Client {
    let model = FakeModel::new(script.into_iter().map(text_response));
    spawn_with(sessions, model, ToolRegistry::new())
}

async fn recv(events: &mut mpsc::Receiver<Event>) -> Event {
    tokio::time::timeout(Duration::from_secs(5), events.recv())
        .await
        .expect("timed out waiting for event")
        .expect("event channel closed")
}

async fn create_session(client: &mut Client) -> SessionId {
    client.0.send(Command::CreateSession).await.unwrap();
    match recv(&mut client.1).await.payload {
        EventPayload::SessionCreated { meta } => meta.id,
        other => panic!("expected session_created, got {other:?}"),
    }
}

async fn submit(client: &mut Client, session_id: &SessionId, text: &str) {
    client
        .0
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: text.into(),
        })
        .await
        .unwrap();
    loop {
        if let EventPayload::TurnFinished { status, error, .. } = recv(&mut client.1).await.payload
        {
            assert_eq!(status, CompletionStatus::Completed, "{error:?}");
            return;
        }
    }
}

/// Also the only way to observe that a `set_head` has been handled, since it
/// emits no event of its own: commands are answered in order.
async fn load(client: &mut Client, session_id: &SessionId) -> SessionSnapshot {
    client
        .0
        .send(Command::LoadSession {
            session_id: session_id.clone(),
        })
        .await
        .unwrap();
    match recv(&mut client.1).await.payload {
        EventPayload::SessionLoaded { snapshot } => snapshot,
        other => panic!("expected session_loaded, got {other:?}"),
    }
}

async fn set_head(client: &mut Client, session_id: &SessionId, item_id: Option<ItemId>) {
    client
        .0
        .send(Command::SetHead {
            session_id: session_id.clone(),
            item_id,
        })
        .await
        .unwrap();
}

async fn list(client: &mut Client) -> Vec<SessionMeta> {
    client.0.send(Command::ListSessions).await.unwrap();
    match recv(&mut client.1).await.payload {
        EventPayload::SessionsListed { sessions } => sessions,
        other => panic!("expected sessions_listed, got {other:?}"),
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

fn texts(snapshot: &SessionSnapshot) -> Vec<&str> {
    snapshot
        .items
        .iter()
        .map(|item| match &item.payload {
            ItemPayload::UserMessage { text } | ItemPayload::AssistantMessage { text } => {
                text.as_str()
            }
            other => panic!("unexpected payload {other:?}"),
        })
        .collect()
}

fn log_path(sessions: &Path, session_id: &SessionId) -> PathBuf {
    sessions.join(format!("{}.jsonl", session_id.0))
}

#[tokio::test]
async fn a_restart_preserves_the_tree_and_both_forks_resume_independently() {
    let sessions = tempfile::tempdir().unwrap();

    // Two turns, then a fork off the first assistant reply.
    let mut first = spawn(sessions.path(), ["one", "two", "three"]);
    let session_id = create_session(&mut first).await;
    submit(&mut first, &session_id, "first").await;
    submit(&mut first, &session_id, "second").await;
    let snapshot = load(&mut first, &session_id).await;
    let branch_point = snapshot.items[1].id;
    set_head(&mut first, &session_id, Some(branch_point)).await;
    submit(&mut first, &session_id, "third").await;
    let before = load(&mut first, &session_id).await;
    assert_eq!(before.items.len(), 6);
    assert_eq!(
        leaves(&before),
        vec![before.items[3].id, before.items[5].id]
    );
    drop(first);

    // Restart: the tree, the head, and every item's content come back from
    // the log alone.
    let mut second = spawn(sessions.path(), ["fourth", "fifth"]);
    assert_eq!(
        list(&mut second)
            .await
            .into_iter()
            .map(|meta| meta.id)
            .collect::<Vec<_>>(),
        vec![session_id.clone()]
    );
    let after = load(&mut second, &session_id).await;
    assert_eq!(after.items, before.items);
    assert_eq!(after.head, before.head);
    assert_eq!(
        texts(&after),
        ["first", "one", "second", "two", "third", "three"]
    );

    // Each fork leaf continues on its own, with fresh item ids and turn ids.
    let (left, right) = (after.items[3].id, after.items[5].id);
    set_head(&mut second, &session_id, Some(left)).await;
    submit(&mut second, &session_id, "under left").await;
    set_head(&mut second, &session_id, Some(right)).await;
    submit(&mut second, &session_id, "under right").await;
    let grown = load(&mut second, &session_id).await;
    assert_eq!(grown.items.len(), 10);
    assert_eq!(grown.items[6].parent_id, Some(left));
    assert_eq!(grown.items[8].parent_id, Some(right));
    assert_eq!(leaves(&grown), vec![grown.items[7].id, grown.items[9].id]);
    assert_eq!(grown.items[6].turn_id.0, 3);
    assert_eq!(grown.items[8].turn_id.0, 4);
    drop(second);

    // And the resumed session's own appends are durable in turn.
    let mut third = spawn(sessions.path(), []);
    let reloaded = load(&mut third, &session_id).await;
    assert_eq!(reloaded.items, grown.items);
    assert_eq!(reloaded.head, grown.head);
}

#[tokio::test]
async fn a_head_moved_without_a_turn_survives_a_restart() {
    let sessions = tempfile::tempdir().unwrap();

    let mut first = spawn(sessions.path(), ["one", "two"]);
    let session_id = create_session(&mut first).await;
    submit(&mut first, &session_id, "first").await;
    submit(&mut first, &session_id, "second").await;
    let snapshot = load(&mut first, &session_id).await;
    let rewound = snapshot.items[1].id;
    set_head(&mut first, &session_id, Some(rewound)).await;
    // Reading the snapshot back is what proves the set_head reached the log
    // before the harness goes away.
    assert_eq!(load(&mut first, &session_id).await.head, Some(rewound));
    drop(first);

    let mut second = spawn(sessions.path(), ["three"]);
    let resumed = load(&mut second, &session_id).await;
    assert_eq!(resumed.items.len(), 4);
    assert_eq!(resumed.head, Some(rewound));

    // Submitting from the recovered head forks, exactly as it would have
    // before the restart.
    submit(&mut second, &session_id, "third").await;
    let forked = load(&mut second, &session_id).await;
    assert_eq!(forked.items[4].parent_id, Some(rewound));
    assert_eq!(
        leaves(&forked),
        vec![forked.items[3].id, forked.items[5].id]
    );
}

#[tokio::test]
async fn an_unterminated_trailing_line_is_dropped_and_the_log_stays_appendable() {
    let sessions = tempfile::tempdir().unwrap();

    let mut first = spawn(sessions.path(), ["one"]);
    let session_id = create_session(&mut first).await;
    submit(&mut first, &session_id, "first").await;
    drop(first);

    // A crash between the record and its newline: the JSON is complete, but
    // the record was never committed.
    let path = log_path(sessions.path(), &session_id);
    let committed = fs::read_to_string(&path).unwrap();
    let torn = r#"{"record":"item","id":2,"parent_id":1,"turn_id":1,"created_at_ms":7,"status":"completed","error":null,"kind":"user_message","text":"torn"}"#;
    fs::write(&path, format!("{committed}{torn}")).unwrap();

    let mut second = spawn(sessions.path(), ["two"]);
    let repaired = load(&mut second, &session_id).await;
    assert_eq!(texts(&repaired), ["first", "one"]);
    assert_eq!(repaired.head, Some(repaired.items[1].id));
    // The log was truncated back to its committed prefix, so the next append
    // does not extend a line no reader accepted.
    assert_eq!(fs::read_to_string(&path).unwrap(), committed);

    submit(&mut second, &session_id, "second").await;
    drop(second);

    let mut third = spawn(sessions.path(), []);
    let grown = load(&mut third, &session_id).await;
    assert_eq!(texts(&grown), ["first", "one", "second", "two"]);
    // The dropped record's id was never used, so the repaired log reuses it.
    assert_eq!(grown.items[2].id, ItemId(2));
    assert!(fs::read_to_string(&path).unwrap().ends_with('\n'));
}

#[tokio::test]
async fn a_corrupt_line_takes_everything_after_it() {
    let sessions = tempfile::tempdir().unwrap();

    let mut first = spawn(sessions.path(), ["one", "two"]);
    let session_id = create_session(&mut first).await;
    submit(&mut first, &session_id, "first").await;
    submit(&mut first, &session_id, "second").await;
    drop(first);

    // Damage the middle of the log: a header, three good items, garbage, then
    // a record that would have been fine on its own.
    let path = log_path(sessions.path(), &session_id);
    let lines: Vec<String> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(lines.len(), 5);
    let prefix = format!("{}\n", lines[..4].join("\n"));
    fs::write(&path, format!("{prefix}{{\"record\":\n{}\n", lines[4])).unwrap();

    let mut second = spawn(sessions.path(), ["three"]);
    let repaired = load(&mut second, &session_id).await;
    // The record after the defect is gone even though it parsed: a log is
    // valid only as a prefix.
    assert_eq!(texts(&repaired), ["first", "one", "second"]);
    assert_eq!(repaired.head, Some(repaired.items[2].id));
    assert_eq!(fs::read_to_string(&path).unwrap(), prefix);

    submit(&mut second, &session_id, "third").await;
    let grown = load(&mut second, &session_id).await;
    assert_eq!(texts(&grown), ["first", "one", "second", "third", "three"]);
    assert_eq!(grown.items[3].parent_id, Some(repaired.items[2].id));
}

#[tokio::test]
async fn sessions_are_listed_newest_first_and_load_by_id() {
    let sessions = tempfile::tempdir().unwrap();

    let mut first = spawn(sessions.path(), ["one", "two"]);
    let older = create_session(&mut first).await;
    submit(&mut first, &older, "in the older session").await;
    let newer = create_session(&mut first).await;
    submit(&mut first, &newer, "in the newer session").await;
    drop(first);

    let mut second = spawn(sessions.path(), []);
    let listed: Vec<SessionId> = list(&mut second)
        .await
        .into_iter()
        .map(|meta| meta.id)
        .collect::<Vec<_>>();
    assert_eq!(listed, vec![newer.clone(), older.clone()]);

    // Each one loads its own history, and an id that names no file is
    // refused rather than resolved into a path.
    assert_eq!(
        texts(&load(&mut second, &older).await)[0],
        "in the older session"
    );
    assert_eq!(
        texts(&load(&mut second, &newer).await)[0],
        "in the newer session"
    );
    second
        .0
        .send(Command::LoadSession {
            session_id: SessionId("../escape".into()),
        })
        .await
        .unwrap();
    second.0.send(Command::ListSessions).await.unwrap();
    assert!(matches!(
        recv(&mut second.1).await.payload,
        EventPayload::SessionsListed { .. }
    ));
}

#[tokio::test]
async fn every_item_kind_survives_a_restart_unchanged() {
    let sessions = tempfile::tempdir().unwrap();

    // One turn covering all five kinds, including the opaque provider data a
    // reasoning model needs replayed and the integer fields a tool call and
    // its result carry.
    let model = FakeModel::new([
        FakeResponse::new(vec![
            ModelEvent::ReasoningDelta {
                delta: "Checking.".into(),
            },
            ModelEvent::ReasoningCompleted {
                provider_data: json!({ "encrypted": "opaque", "index": 3 }),
            },
            ModelEvent::ToolCallCompleted {
                call: ToolCallRequest {
                    call_id: "call-1".into(),
                    tool: "read_file".into(),
                    arguments: json!({ "path": "README.md", "start_line": 12 }),
                },
            },
            ModelEvent::Completed,
        ]),
        text_response("done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(FakeTool::canned(
        "read_file",
        ToolEffect::ReadOnly,
        "contents",
    ));

    let mut first = spawn_with(sessions.path(), model, tools);
    let session_id = create_session(&mut first).await;
    submit(&mut first, &session_id, "read it").await;
    let before = load(&mut first, &session_id).await;
    drop(first);

    let mut second = spawn(sessions.path(), []);
    let after = load(&mut second, &session_id).await;
    assert_eq!(after.items, before.items);
    assert_eq!(after.head, before.head);
    assert_eq!(after.meta.created_at_ms, before.meta.created_at_ms);
    assert_eq!(
        after
            .items
            .iter()
            .map(|item| item.payload.kind().to_string())
            .collect::<Vec<_>>(),
        [
            "user_message",
            "reasoning",
            "tool_call",
            "tool_result",
            "assistant_message"
        ]
    );
}

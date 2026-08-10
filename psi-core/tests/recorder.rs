//! Recorded benchmark tasks: a live run (played here by the fake model, which
//! is a `ModelBackend` like any other) becomes a recording, the recording
//! becomes a task, and the task replays the run — same calls, same final
//! workspace — with its timing taken from the recording instead of injected
//! distributions.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use psi_core::bench::{
    BenchConfig, Latency, RecordedDurations, Speculation, SpeculationStats, Strategy, record_task,
    recorded_task, run_trial, script_from_items,
};
use psi_core::fake::{FakeModel, FakeResponse, FakeTool};
use psi_core::item::{CompletionStatus, Item, ItemId, ItemPayload, TurnId, WorkspaceRevision};
use psi_core::model::{ModelEvent, ToolCallRequest, Usage};
use psi_core::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolRegistry, ToolSpec};
use psi_core::trace::RunTrace;
use serde_json::json;

fn call(tool: &str, call_id: &str, arguments: serde_json::Value) -> ModelEvent {
    ModelEvent::ToolCallCompleted {
        call: ToolCallRequest {
            call_id: call_id.into(),
            tool: tool.into(),
            arguments,
        },
    }
}

/// The live run this suite records: a multi-call read round, a mutating round
/// that patches and verifies, and a closing answer — two turns, four rounds,
/// exercising round grouping, the mutating path, and final-state capture.
fn live_script() -> Vec<FakeResponse> {
    vec![
        FakeResponse::new(vec![
            ModelEvent::ReasoningDelta {
                delta: "Read both files first.".into(),
            },
            call("read_file", "c1", json!({ "path": "notes.txt" })),
            call("read_file", "c2", json!({ "path": "config.txt" })),
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            call(
                "apply_patch",
                "c3",
                json!({ "path": "config.txt", "old_text": "limit=1", "new_text": "limit=2" }),
            ),
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "Raised the limit.".into(),
            },
            ModelEvent::Usage {
                usage: Usage {
                    input_tokens: 1_200,
                    output_tokens: 40,
                },
            },
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            call("read_file", "c4", json!({ "path": "config.txt" })),
            ModelEvent::Completed,
        ]),
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: "It reads limit=2 now.".into(),
            },
            ModelEvent::Usage {
                usage: Usage {
                    input_tokens: 900,
                    output_tokens: 25,
                },
            },
            ModelEvent::Completed,
        ]),
    ]
}

/// A snapshot of a real workspace is not all UTF-8; `logo.bin` keeps the
/// round trip honest about that. The `loop` symlink points back at the
/// fixture itself: snapshotting must skip it rather than recurse into it.
fn fixture(dir: &Path) {
    std::fs::write(dir.join("notes.txt"), "remember the limit\n").unwrap();
    std::fs::write(dir.join("config.txt"), "limit=1\n").unwrap();
    std::fs::write(dir.join("logo.bin"), [0u8, 159, 146, 150]).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir, dir.join("loop")).unwrap();
}

async fn record_live_run(root: &Path) -> std::path::PathBuf {
    let source = root.join("source-fixture");
    std::fs::create_dir_all(&source).unwrap();
    fixture(&source);
    let out = root.join("recording");
    record_task(
        "raise_limit",
        &source,
        &["raise the limit".into(), "check it".into()],
        Arc::new(FakeModel::new(live_script())),
        &out,
    )
    .await
    .unwrap();
    out
}

#[tokio::test]
async fn a_recorded_run_replays_itself() {
    let root = tempfile::tempdir().unwrap();
    let out = record_live_run(root.path()).await;

    let task = recorded_task(&out).unwrap();
    assert_eq!(task.name, "raise_limit");
    assert_eq!(task.prompts, ["raise the limit", "check it"]);
    assert!(
        task.fixture
            .iter()
            .any(|(path, contents)| path == "config.txt" && contents.as_slice() == b"limit=1\n")
    );
    // The snapshot's binary file loads and replays as bytes.
    assert!(
        task.fixture
            .iter()
            .any(|(path, contents)| path == "logo.bin" && contents.as_slice() == [0, 159, 146, 150])
    );
    // The fixture's self-referential symlink was skipped, not followed.
    assert!(
        task.fixture
            .iter()
            .all(|(path, _)| !path.starts_with("loop"))
    );

    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(0),
        model_delay_ms: 0,
        speculate: None,
    };
    let trace_path = run_trial(&task, 0, &config, &root.path().join("replay"))
        .await
        .unwrap();
    let replay = RunTrace::read(&trace_path).unwrap();
    assert!(
        replay.success,
        "the replayed workspace matches the recording"
    );

    // The replay makes exactly the recorded calls, in order.
    let calls: Vec<(String, String)> = replay
        .tool_calls()
        .into_iter()
        .map(|call| (call.tool, call.arguments.to_string()))
        .collect();
    assert_eq!(
        calls,
        [
            ("read_file".into(), json!({"path": "notes.txt"}).to_string()),
            (
                "read_file".into(),
                json!({"path": "config.txt"}).to_string()
            ),
            (
                "apply_patch".into(),
                json!({"new_text": "limit=2", "old_text": "limit=1", "path": "config.txt"})
                    .to_string()
            ),
            (
                "read_file".into(),
                json!({"path": "config.txt"}).to_string()
            ),
        ]
    );
    assert_eq!(replay.turns.len(), 2);

    // Usage is not persisted as an item; it comes back from the trace, so the
    // replayed turns report what the live ones billed.
    let tokens: Vec<Option<Usage>> = replay.turns.iter().map(|turn| turn.usage).collect();
    assert_eq!(
        tokens,
        [
            Some(Usage {
                input_tokens: 1_200,
                output_tokens: 40,
            }),
            Some(Usage {
                input_tokens: 900,
                output_tokens: 25,
            }),
        ]
    );
}

#[tokio::test]
async fn a_tampered_final_state_fails_the_replay() {
    let root = tempfile::tempdir().unwrap();
    let out = record_live_run(root.path()).await;
    // The recording claims a final state the replay cannot reach.
    std::fs::write(out.join("final/config.txt"), "limit=3\n").unwrap();

    let task = recorded_task(&out).unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(0),
        model_delay_ms: 0,
        speculate: None,
    };
    let trace_path = run_trial(&task, 0, &config, &root.path().join("replay"))
        .await
        .unwrap();
    assert!(!RunTrace::read(&trace_path).unwrap().success);
}

#[tokio::test]
async fn the_oracle_reaches_its_ceiling_on_a_recording() {
    let root = tempfile::tempdir().unwrap();
    let out = record_live_run(root.path()).await;
    let task = recorded_task(&out).unwrap();

    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(0),
        model_delay_ms: 0,
        speculate: Some(Speculation {
            strategy: Strategy::Oracle,
            prediction_budget: 256,
            execution_budget: 4,
        }),
    };
    let trace_path = run_trial(&task, 0, &config, &root.path().join("replay"))
        .await
        .unwrap();
    let replay = RunTrace::read(&trace_path).unwrap();
    assert!(replay.success);
    let stats = SpeculationStats::of(std::slice::from_ref(&replay)).unwrap();
    // The three reads are allowlisted; the patch stays authoritative.
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 1);
}

fn item(
    id: u64,
    turn: u64,
    created_at_ms: u64,
    status: CompletionStatus,
    payload: ItemPayload,
) -> Item {
    Item {
        id: ItemId(id),
        parent_id: id.checked_sub(1).map(ItemId),
        turn_id: TurnId(turn),
        created_at_ms,
        status,
        error: None,
        payload,
    }
}

fn user(id: u64, turn: u64, at: u64) -> Item {
    item(
        id,
        turn,
        at,
        CompletionStatus::Completed,
        ItemPayload::UserMessage {
            text: format!("prompt {turn}"),
        },
    )
}

fn tool_call(id: u64, turn: u64, at: u64, call_id: &str, path: &str) -> Item {
    item(
        id,
        turn,
        at,
        CompletionStatus::Completed,
        ItemPayload::ToolCall {
            tool: "read_file".into(),
            call_id: call_id.into(),
            arguments: json!({ "path": path }),
            cwd: "/fixture".into(),
            revision: WorkspaceRevision(0),
        },
    )
}

fn tool_result(id: u64, turn: u64, at: u64, call_id: &str, duration_ms: u64) -> Item {
    item(
        id,
        turn,
        at,
        CompletionStatus::Completed,
        ItemPayload::ToolResult {
            call_id: call_id.into(),
            content: "contents".into(),
            duration_ms,
            truncated: false,
        },
    )
}

#[test]
fn round_boundaries_and_delays_recover_from_the_item_log() {
    // One turn, two rounds: [reasoning, two calls] then [text], with real
    // timestamps. The first response's generation delay spans the prompt to
    // its last streamed item; the second's spans the last result to its text.
    let items = vec![
        user(0, 0, 1_000),
        item(
            1,
            0,
            1_800,
            CompletionStatus::Completed,
            ItemPayload::Reasoning {
                text: "look twice".into(),
                provider_data: None,
            },
        ),
        tool_call(2, 0, 2_000, "c1", "a.txt"),
        tool_call(3, 0, 2_050, "c2", "b.txt"),
        tool_result(4, 0, 2_300, "c1", 200),
        tool_result(5, 0, 2_500, "c2", 180),
        item(
            6,
            0,
            4_000,
            CompletionStatus::Completed,
            ItemPayload::AssistantMessage {
                text: "done".into(),
            },
        ),
    ];

    let script = script_from_items(
        &items,
        &[Some(Usage {
            input_tokens: 100,
            output_tokens: 9,
        })],
    )
    .unwrap();
    assert_eq!(script.len(), 2);
    assert_eq!(
        script[0].delay_ms, 1_050,
        "prompt at 1000, last call at 2050"
    );
    assert_eq!(
        script[1].delay_ms, 1_500,
        "last result at 2500, text at 4000"
    );

    let kinds: Vec<&str> = script[0]
        .events
        .iter()
        .map(|event| match event {
            ModelEvent::ReasoningDelta { .. } => "reasoning",
            ModelEvent::ToolCallCompleted { .. } => "call",
            ModelEvent::TextDelta { .. } => "text",
            ModelEvent::Completed => "completed",
            other => panic!("unexpected event {other:?}"),
        })
        .collect();
    assert_eq!(kinds, ["reasoning", "call", "call", "completed"]);
    // The turn's usage rides on its last response, and only that one.
    assert!(matches!(
        &script[1].events[..],
        [ModelEvent::TextDelta { delta }, ModelEvent::Usage { usage }, ModelEvent::Completed]
            if delta == "done" && usage.input_tokens == 100 && usage.output_tokens == 9
    ));
}

#[test]
fn an_interrupted_recording_is_refused() {
    let items = vec![
        user(0, 0, 1_000),
        item(
            1,
            0,
            1_500,
            CompletionStatus::Cancelled,
            ItemPayload::AssistantMessage {
                text: "partial".into(),
            },
        ),
    ];
    let err = script_from_items(&items, &[]).unwrap_err();
    assert!(err.to_string().contains("interrupted"), "{err}");
}

#[tokio::test]
async fn recorded_durations_replay_by_identity_not_call_order() {
    let items = vec![
        user(0, 0, 0),
        tool_call(1, 0, 0, "c1", "slow.txt"),
        tool_result(2, 0, 0, "c1", 80),
    ];
    let durations = RecordedDurations::of(&items);
    let mut registry = ToolRegistry::new();
    registry.register(FakeTool::canned("read_file", ToolEffect::ReadOnly, "x"));
    let replayed = psi_core::bench::replay_durations(registry, durations);
    let tool = replayed.get("read_file").unwrap().clone();

    let run = |arguments: serde_json::Value| {
        let tool = tool.clone();
        async move {
            let started = Instant::now();
            tool.execute(ToolInvocation {
                call_id: "x".into(),
                arguments,
                cwd: "/fixture".into(),
            })
            .await;
            started.elapsed().as_millis() as u64
        }
    };

    // A guess (an id the recording never issued) costs what the recorded call
    // it stands for cost — every time it runs, statelessly.
    assert!(run(json!({ "path": "slow.txt" })).await >= 80);
    assert!(run(json!({ "path": "slow.txt" })).await >= 80);
    // A call the recording never made adds nothing.
    assert!(run(json!({ "path": "other.txt" })).await < 40);
}

/// A tool that takes real time, standing in for a live `exec`.
struct SlowTool {
    inner: FakeTool,
    delay: std::time::Duration,
}

impl Tool for SlowTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn effect(&self) -> ToolEffect {
        self.inner.effect()
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        let future = self.inner.execute(invocation);
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            future.await
        })
    }
}

async fn timed(registry: &ToolRegistry, call_id: &str, arguments: serde_json::Value) -> u64 {
    let tool = registry.get("read_file").unwrap().clone();
    let started = Instant::now();
    tool.execute(ToolInvocation {
        call_id: call_id.into(),
        arguments,
        cwd: "/fixture".into(),
    })
    .await;
    started.elapsed().as_millis() as u64
}

/// The recorded duration contains the live execution, so the replay reaches
/// it rather than adds to it: a call whose replayed execution takes 50ms of a
/// recorded 90ms sleeps only the remaining 40.
#[tokio::test]
async fn a_replayed_execution_counts_toward_its_recorded_duration() {
    let items = vec![
        user(0, 0, 0),
        tool_call(1, 0, 0, "c1", "slow.txt"),
        tool_result(2, 0, 0, "c1", 90),
    ];
    let mut registry = ToolRegistry::new();
    registry.register(SlowTool {
        inner: FakeTool::canned("read_file", ToolEffect::ReadOnly, "x"),
        delay: std::time::Duration::from_millis(50),
    });
    let replayed = psi_core::bench::replay_durations(registry, RecordedDurations::of(&items));

    let elapsed = timed(&replayed, "c1", json!({ "path": "slow.txt" })).await;
    assert!((90..135).contains(&elapsed), "took {elapsed}ms");
}

/// Authoritative calls replay under the recording's own call ids, so each
/// takes exactly the time that call took live — in any order, repeatedly,
/// across profiles. Nothing consumes anything: a discarded guess or an
/// earlier trial cannot shift a later call's replayed time.
#[tokio::test]
async fn recorded_call_ids_replay_their_exact_durations_statelessly() {
    let items = vec![
        user(0, 0, 0),
        tool_call(1, 0, 0, "c1", "twice.txt"),
        tool_result(2, 0, 0, "c1", 80),
        tool_call(3, 0, 0, "c2", "twice.txt"),
        tool_result(4, 0, 0, "c2", 10),
    ];
    let durations = RecordedDurations::of(&items);
    let profile = || {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool::canned("read_file", ToolEffect::ReadOnly, "x"));
        psi_core::bench::replay_durations(registry, durations.clone())
    };

    // Out of recorded order, and again after a guess ran in between: each
    // recorded id keeps its own duration.
    let first = profile();
    assert!(timed(&first, "c2", json!({ "path": "twice.txt" })).await < 40);
    assert!(timed(&first, "guess-1", json!({ "path": "twice.txt" })).await >= 80);
    assert!(timed(&first, "c1", json!({ "path": "twice.txt" })).await >= 80);
    assert!(timed(&first, "c2", json!({ "path": "twice.txt" })).await < 40);
    // A second profile replays identically.
    let second = profile();
    assert!(timed(&second, "c1", json!({ "path": "twice.txt" })).await >= 80);
    assert!(timed(&second, "c2", json!({ "path": "twice.txt" })).await < 40);
}

/// `--record` pointed inside `--fixture` would snapshot the recording into
/// itself; it is refused before any copying starts.
#[tokio::test]
async fn a_recording_nested_under_its_fixture_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("fixture");
    std::fs::create_dir_all(&source).unwrap();
    fixture(&source);

    let out = source.join("recordings/run");
    let err = record_task(
        "nested",
        &source,
        &["go".into()],
        Arc::new(FakeModel::new(Vec::new())),
        &out,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("inside the fixture"), "{err}");
}

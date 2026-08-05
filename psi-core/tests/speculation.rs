//! Milestone 6's verification: an oracle run shows measured latency reduction
//! against the Milestone 5 baseline with agent-visible results unchanged —
//! plus the correctness properties the reduction must not cost: stale results
//! are never adopted, budgets and hooks bound what runs, and wasted work is
//! recorded.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use psi_core::bench::{
    BenchConfig, Latency, ReplayOracle, Speculation, SpeculationStats, Strategy, run_trial, tasks,
};
use psi_core::fake::{FakeModel, FakeResponse};
use psi_core::hook::{Hook, HookDecision, HookRegistry};
use psi_core::item::{CompletionStatus, ItemPayload};
use psi_core::model::{ModelEvent, ToolCallRequest};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::SessionId;
use psi_core::speculation::{Predictor, SpeculationConfig, v0_allowlist};
use psi_core::tool::{ToolInvocation, ToolOutput};
use psi_core::tools::default_profile;
use psi_core::trace::{DiscardReason, RunTrace, TraceRecord, TraceWriter};
use psi_core::{Harness, HarnessConfig};
use serde_json::json;
use tokio::sync::mpsc;

fn read(path: &str, call_id: &str) -> ModelEvent {
    ModelEvent::ToolCallCompleted {
        call: ToolCallRequest {
            call_id: call_id.into(),
            tool: "read_file".into(),
            arguments: json!({ "path": path }),
        },
    }
}

fn fixture(dir: &Path, files: &[(&str, &str)]) {
    for (path, contents) in files {
        std::fs::write(dir.join(path), contents).unwrap();
    }
}

struct Client {
    commands: mpsc::Sender<Command>,
    events: mpsc::Receiver<Event>,
}

/// A harness over the real tool profile in `workspace`, speculating with the
/// given predictor when one is passed.
fn spawn(
    workspace: &Path,
    script: Vec<FakeResponse>,
    predictor: Option<Arc<dyn Predictor>>,
    trace: Option<TraceWriter>,
    hooks: HookRegistry,
) -> Client {
    let (commands, events) = Harness::spawn(HarnessConfig {
        model: Arc::new(FakeModel::new(script)),
        tools: default_profile(workspace.to_path_buf()),
        hooks,
        workspace: workspace.to_path_buf(),
        sessions_dir: workspace.join(".sessions"),
        trace,
        speculation: predictor.map(|predictor| SpeculationConfig {
            predictor,
            allowlist: v0_allowlist(),
            // The oracle reads a recording rather than a model, so no budget
            // it is handed changes what it guesses.
            prediction_budget: 0,
            execution_budget: 2,
        }),
    })
    .unwrap();
    Client { commands, events }
}

async fn recv(client: &mut Client) -> Event {
    tokio::time::timeout(Duration::from_secs(5), client.events.recv())
        .await
        .expect("timed out waiting for event")
        .expect("event channel closed")
}

async fn create_session(client: &mut Client) -> SessionId {
    client.commands.send(Command::CreateSession).await.unwrap();
    match recv(client).await.payload {
        EventPayload::SessionCreated { meta } => meta.id,
        other => panic!("expected session_created, got {other:?}"),
    }
}

async fn run_turn(client: &mut Client, session_id: &SessionId, text: &str) -> Vec<Event> {
    client
        .commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: text.into(),
        })
        .await
        .unwrap();
    let mut collected = Vec::new();
    loop {
        let event = recv(client).await;
        let done = matches!(event.payload, EventPayload::TurnFinished { .. });
        collected.push(event);
        if done {
            break;
        }
    }
    collected
}

/// Everything the agent sees, minus timing: event kinds in order, item
/// contents, and statuses. Tool durations are excluded — they are the latency
/// speculation changes.
fn visible(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .map(|event| match &event.payload {
            EventPayload::TurnStarted { turn_id } => format!("turn_started:{turn_id:?}"),
            EventPayload::ItemStarted { kind, .. } => format!("item_started:{kind}"),
            EventPayload::ItemDelta { delta, .. } => format!("delta:{delta}"),
            EventPayload::ItemFinished { item } => {
                let content = match &item.payload {
                    ItemPayload::UserMessage { text } => text.clone(),
                    ItemPayload::AssistantMessage { text } => text.clone(),
                    ItemPayload::Reasoning { text, .. } => text.clone(),
                    ItemPayload::ToolCall {
                        tool, arguments, ..
                    } => format!("{tool} {arguments}"),
                    ItemPayload::ToolResult { content, .. } => content.clone(),
                };
                format!(
                    "item_finished:{}:{}:{content}",
                    item.payload.kind(),
                    item.status
                )
            }
            EventPayload::TurnFinished { status, .. } => format!("turn_finished:{status}"),
            other => format!("{other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn speculation_leaves_agent_visible_results_unchanged() {
    let workspace = tempfile::tempdir().unwrap();
    fixture(workspace.path(), &[("a.txt", "alpha"), ("b.txt", "beta")]);
    let script = || {
        vec![
            FakeResponse::new(vec![
                ModelEvent::ReasoningDelta {
                    delta: "Reading both.".into(),
                },
                read("a.txt", "c1"),
                read("b.txt", "c2"),
                ModelEvent::Completed,
            ])
            .delayed(50),
            FakeResponse::new(vec![
                ModelEvent::TextDelta {
                    delta: "both read".into(),
                },
                ModelEvent::Completed,
            ]),
        ]
    };

    let mut baseline = spawn(workspace.path(), script(), None, None, HookRegistry::new());
    let session = create_session(&mut baseline).await;
    let baseline_turn = run_turn(&mut baseline, &session, "read them").await;

    let oracle: Arc<dyn Predictor> = Arc::new(ReplayOracle::for_script(&script()));
    let mut speculated = spawn(
        workspace.path(),
        script(),
        Some(oracle),
        None,
        HookRegistry::new(),
    );
    let session = create_session(&mut speculated).await;
    let speculated_turn = run_turn(&mut speculated, &session, "read them").await;

    assert_eq!(visible(&baseline_turn), visible(&speculated_turn));
    // Without a trace no sequence numbers are consumed by speculation either,
    // so even the seq column is identical.
    let seqs = |events: &[Event]| events.iter().map(|e| e.seq).collect::<Vec<_>>();
    assert_eq!(seqs(&baseline_turn), seqs(&speculated_turn));
}

#[tokio::test]
async fn oracle_run_beats_the_baseline_and_hits_every_call() {
    let task = tasks()
        .iter()
        .find(|task| task.name == "read_and_answer")
        .expect("the read-only benchmark task");
    let dir = tempfile::tempdir().unwrap();
    let config = BenchConfig {
        trials: 1,
        latency: Latency::fixed(250),
        model_delay_ms: 600,
        speculate: None,
    };

    let baseline_path = run_trial(task, 0, &config, &dir.path().join("baseline"))
        .await
        .unwrap();
    let baseline = RunTrace::read(&baseline_path).unwrap();

    let mut speculated_config = config.clone();
    speculated_config.speculate = Some(Speculation {
        strategy: Strategy::Oracle,
        prediction_budget: 0,
        execution_budget: 4,
    });
    let speculated_path = run_trial(task, 0, &speculated_config, &dir.path().join("speculated"))
        .await
        .unwrap();
    let speculated = RunTrace::read(&speculated_path).unwrap();

    // Same task, same success, same answers.
    assert!(baseline.success && speculated.success);

    // Perfect prediction: every authoritative call is served from the cache.
    let stats = SpeculationStats::of(std::slice::from_ref(&speculated)).expect("records");
    assert_eq!(stats.misses, 0);
    let calls = speculated
        .turns
        .iter()
        .map(|turn| turn.tool_ms())
        .filter(|ms| *ms > 0)
        .count();
    assert!(stats.hits >= 1, "the task makes tool calls");
    assert_eq!(stats.hits + stats.misses, baseline.tool_calls().len());
    assert!(calls <= stats.hits);

    // The reduction: injected latency hides under the model delay. Each
    // baseline call costs a fixed 250ms of turn wall time that the oracle run
    // does not pay; leave half as scheduling slack.
    let wall = |run: &RunTrace| run.turns.iter().map(|turn| turn.wall_ms()).sum::<u64>();
    let saved = wall(&baseline).saturating_sub(wall(&speculated));
    let expected = 250 * baseline.tool_calls().len() as u64;
    assert!(
        saved * 2 >= expected,
        "expected roughly {expected}ms saved, measured {saved}ms"
    );
}

#[tokio::test]
async fn a_mutation_invalidates_stale_speculation() {
    let workspace = tempfile::tempdir().unwrap();
    fixture(workspace.path(), &[("x.txt", "old")]);
    let script = || {
        vec![
            FakeResponse::new(vec![
                ModelEvent::ToolCallCompleted {
                    call: ToolCallRequest {
                        call_id: "c1".into(),
                        tool: "apply_patch".into(),
                        arguments: json!({ "path": "x.txt", "old_text": "old", "new_text": "new" }),
                    },
                },
                read("x.txt", "c2"),
                ModelEvent::Completed,
            ])
            .delayed(100),
            FakeResponse::new(vec![
                ModelEvent::TextDelta {
                    delta: "done".into(),
                },
                ModelEvent::Completed,
            ]),
        ]
    };

    let trace_path = workspace.path().join("run.jsonl");
    let trace = TraceWriter::create(&trace_path).unwrap();
    trace
        .write(&TraceRecord::Run {
            task: "invalidation".into(),
            trial: 0,
            started_at_ms: 0,
        })
        .unwrap();

    let oracle: Arc<dyn Predictor> = Arc::new(ReplayOracle::for_script(&script()));
    let mut client = spawn(
        workspace.path(),
        script(),
        Some(oracle),
        Some(trace.clone()),
        HookRegistry::new(),
    );
    let session = create_session(&mut client).await;
    let turn = run_turn(&mut client, &session, "patch then read").await;
    trace
        .write(&TraceRecord::Outcome { success: true })
        .unwrap();

    // The authoritative read sees the patched file: the speculative read of
    // the old contents was never adopted.
    let read_result = turn
        .iter()
        .filter_map(|event| match &event.payload {
            EventPayload::ItemFinished { item } => match &item.payload {
                ItemPayload::ToolResult {
                    call_id, content, ..
                } if call_id == "c2" => Some(content.clone()),
                _ => None,
            },
            _ => None,
        })
        .next()
        .expect("the read's result");
    assert_eq!(read_result, "new");

    let run = RunTrace::read(&trace_path).unwrap();
    let stats = SpeculationStats::of(std::slice::from_ref(&run)).unwrap();
    // Only the read is allowlisted; the mutation bumped the revision out from
    // under it, so it was discarded and both authoritative calls missed.
    assert_eq!(stats.executed, 1);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.wasted, 1);
    let invalidated = run.turns.iter().flat_map(|turn| &turn.speculation).any(
        |record| matches!(record, TraceRecord::SpeculativeDiscard { reason, .. } if *reason == DiscardReason::Invalidated),
    );
    assert!(invalidated, "the discard names the mutation as its reason");
}

/// Blocks reads of one path, for the selection test.
struct BlockPath(&'static str);

impl Hook for BlockPath {
    fn before(&self, tool: &str, invocation: &ToolInvocation) -> HookDecision {
        if tool == "read_file" && invocation.arguments["path"] == self.0 {
            return HookDecision::Block {
                reason: format!("{} is off limits", self.0),
            };
        }
        HookDecision::Continue
    }

    fn after(&self, _tool: &str, _invocation: &ToolInvocation, _output: &ToolOutput) {}
}

#[tokio::test]
async fn the_budget_caps_fanout_and_blocked_predictions_never_run() {
    let workspace = tempfile::tempdir().unwrap();
    fixture(
        workspace.path(),
        &[
            ("a.txt", "a"),
            ("b.txt", "b"),
            ("c.txt", "c"),
            ("d.txt", "d"),
        ],
    );
    // Selection order under a budget of 2: a spawns, c is blocked by the
    // hook, b spawns, d is over budget.
    let script = || {
        vec![
            FakeResponse::new(vec![
                read("a.txt", "c1"),
                read("c.txt", "c2"),
                read("b.txt", "c3"),
                read("d.txt", "c4"),
                ModelEvent::Completed,
            ])
            .delayed(100),
            FakeResponse::new(vec![
                ModelEvent::TextDelta {
                    delta: "done".into(),
                },
                ModelEvent::Completed,
            ]),
        ]
    };

    let trace_path = workspace.path().join("run.jsonl");
    let trace = TraceWriter::create(&trace_path).unwrap();
    trace
        .write(&TraceRecord::Run {
            task: "budget".into(),
            trial: 0,
            started_at_ms: 0,
        })
        .unwrap();

    let mut hooks = HookRegistry::new();
    hooks.register(BlockPath("c.txt"));
    let oracle: Arc<dyn Predictor> = Arc::new(ReplayOracle::for_script(&script()));
    let mut client = spawn(
        workspace.path(),
        script(),
        Some(oracle),
        Some(trace.clone()),
        hooks,
    );
    let session = create_session(&mut client).await;
    let turn = run_turn(&mut client, &session, "read around").await;
    trace
        .write(&TraceRecord::Outcome { success: true })
        .unwrap();

    let run = RunTrace::read(&trace_path).unwrap();
    let stats = SpeculationStats::of(std::slice::from_ref(&run)).unwrap();
    assert_eq!(stats.proposed, 4);
    assert_eq!(stats.executed, 2, "the budget is the fanout cap");
    assert_eq!(stats.hits, 2);
    // d missed; c was refused outright, so it neither hit nor missed.
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.wasted, 0);

    let refused = turn.iter().any(|event| match &event.payload {
        EventPayload::ItemFinished { item } => {
            matches!(&item.payload, ItemPayload::ToolResult { call_id, .. } if call_id == "c2")
                && item.status == CompletionStatus::Failed
        }
        _ => false,
    });
    assert!(refused, "the blocked call surfaces as a refused call");
}

#[tokio::test]
async fn unused_entries_are_wasted_at_turn_end() {
    let workspace = tempfile::tempdir().unwrap();
    fixture(workspace.path(), &[("a.txt", "a"), ("b.txt", "b")]);
    // The model only reads a; the predictor also guesses b, which parks
    // unused until the turn ends.
    let model_script = || {
        vec![
            FakeResponse::new(vec![read("a.txt", "c1"), ModelEvent::Completed]).delayed(100),
            FakeResponse::new(vec![
                ModelEvent::TextDelta {
                    delta: "done".into(),
                },
                ModelEvent::Completed,
            ]),
        ]
    };
    let predicted = vec![FakeResponse::new(vec![
        read("a.txt", "p1"),
        read("b.txt", "p2"),
        ModelEvent::Completed,
    ])];

    let trace_path = workspace.path().join("run.jsonl");
    let trace = TraceWriter::create(&trace_path).unwrap();
    trace
        .write(&TraceRecord::Run {
            task: "unused".into(),
            trial: 0,
            started_at_ms: 0,
        })
        .unwrap();

    let oracle: Arc<dyn Predictor> = Arc::new(ReplayOracle::for_script(&predicted));
    let mut client = spawn(
        workspace.path(),
        model_script(),
        Some(oracle),
        Some(trace.clone()),
        HookRegistry::new(),
    );
    let session = create_session(&mut client).await;
    run_turn(&mut client, &session, "read a").await;
    trace
        .write(&TraceRecord::Outcome { success: true })
        .unwrap();

    let run = RunTrace::read(&trace_path).unwrap();
    let stats = SpeculationStats::of(std::slice::from_ref(&run)).unwrap();
    assert_eq!(stats.executed, 2);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.wasted, 1);
    let unused = run.turns.iter().flat_map(|turn| &turn.speculation).any(
        |record| matches!(record, TraceRecord::SpeculativeDiscard { reason, tool, .. } if *reason == DiscardReason::Unused && tool == "read_file"),
    );
    assert!(unused, "the parked read of b is recorded as unused");
}

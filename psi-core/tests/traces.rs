//! Milestone 5's verification: a run can be reconstructed from its trace and
//! compared across repeated trials.
//!
//! Reconstruction is a precise claim, so these tests spell it out: the trace's
//! item sequence is the session's own durable record, turn boundaries and
//! timings are on it, every tool call carries its arguments and duration,
//! statuses and usage survive, and every number a baseline reports comes back
//! out of the file.

use std::collections::BTreeMap;
use std::path::Path;

use psi_core::bench::{
    BenchConfig, BenchTask, Latency, LatencyProfile, LatencyStream, Stats, TaskReport, run_task,
    run_trial, tasks,
};
use psi_core::item::{CompletionStatus, ItemKind, ItemPayload, TurnId};
use psi_core::model::Usage;
use psi_core::store::SessionStore;
use psi_core::trace::{RunTrace, TraceRecord, TraceWriter};

/// Fast enough to run in a test, slow enough that injected latency dominates
/// what the fixture tools really cost.
fn test_config(trials: u32) -> BenchConfig {
    BenchConfig {
        trials,
        latency: Latency::fixed(10),
        model_delay_ms: 5,
    }
}

fn task(name: &str) -> &'static BenchTask {
    tasks()
        .iter()
        .find(|task| task.name == name)
        .expect("unknown task")
}

fn kinds(items: impl Iterator<Item = ItemKind>) -> Vec<String> {
    items.map(|kind| kind.to_string()).collect()
}

/// The session the trial wrote, loaded straight from its log — the harness's
/// own durable record, which the trace must agree with item for item.
fn only_session(dir: &Path) -> Vec<psi_core::item::Item> {
    let store = SessionStore::new(dir.join("sessions")).unwrap();
    let sessions = store.list();
    assert_eq!(sessions.len(), 1, "one trial writes one session");
    store.load(&sessions[0].id).unwrap().0.items
}

#[tokio::test]
async fn a_run_is_reconstructed_from_its_trace() {
    let dir = tempfile::tempdir().unwrap();
    let task = task("read_and_answer");
    let path = run_trial(task, 0, &test_config(1), dir.path())
        .await
        .unwrap();
    let run = RunTrace::read(&path).unwrap();

    // Which run this was, and whether it did what the task asked.
    assert_eq!(run.task, "read_and_answer");
    assert_eq!(run.trial, 0);
    assert!(run.success);

    // The item sequence: exactly the session's own durable record.
    assert_eq!(
        run.items().cloned().collect::<Vec<_>>(),
        only_session(dir.path())
    );

    // Turn boundaries: one turn per prompt, each holding its own items.
    assert_eq!(run.turns.len(), task.prompts.len());
    assert_eq!(run.turns[0].turn_id, TurnId(0));
    assert_eq!(run.turns[1].turn_id, TurnId(1));
    assert_eq!(
        kinds(run.turns[0].items.iter().map(|item| item.payload.kind())),
        [
            "user_message",
            "tool_call",
            "tool_result",
            "tool_call",
            "tool_result",
            "assistant_message",
        ]
    );
    assert!(
        run.turns
            .iter()
            .all(|turn| turn.items.iter().all(|item| item.turn_id == turn.turn_id))
    );

    // Timings: every turn is bounded, and its wall time splits into the time
    // inside tools and the time outside them.
    for turn in &run.turns {
        assert!(turn.started_at_ms >= run.started_at_ms);
        assert!(turn.finished_at_ms >= turn.started_at_ms);
        assert_eq!(turn.model_ms() + turn.tool_ms(), turn.wall_ms());
        // Two calls at 10ms each, and the model delays sit outside them.
        assert!(turn.tool_ms() >= 20, "tool time {}", turn.tool_ms());
        assert!(turn.model_ms() >= 15, "model time {}", turn.model_ms());
    }

    // Every tool call, with its arguments and what it cost.
    let calls = run.tool_calls();
    let called: Vec<(&str, &serde_json::Value)> = calls
        .iter()
        .map(|call| (call.tool.as_str(), &call.arguments))
        .collect();
    assert_eq!(
        called,
        [
            ("search", &serde_json::json!({ "pattern": "RetryBudget" })),
            ("read_file", &serde_json::json!({ "path": "src/budget.rs" })),
            ("read_file", &serde_json::json!({ "path": "src/budget.rs" })),
            ("search", &serde_json::json!({ "pattern": "max_retries" })),
        ]
    );
    assert!(
        calls.iter().all(|call| call.duration_ms >= 10),
        "injected latency is missing from {calls:?}"
    );

    // Statuses and usage.
    assert!(
        run.turns
            .iter()
            .all(|turn| turn.status == CompletionStatus::Completed && turn.error.is_none())
    );
    assert!(
        run.items()
            .all(|item| item.status == CompletionStatus::Completed)
    );
    assert_eq!(
        run.turns[0].usage,
        Some(Usage {
            input_tokens: 3_900,
            output_tokens: 200,
        })
    );
    assert_eq!(
        run.turns[1].usage,
        Some(Usage {
            input_tokens: 6_300,
            output_tokens: 190,
        })
    );
}

/// The shape a trial ran, with everything a clock touched left out: two trials
/// of one task must agree on this exactly.
fn shape(run: &RunTrace) -> Vec<String> {
    let mut shape = vec![format!("task:{} success:{}", run.task, run.success)];
    for turn in &run.turns {
        shape.push(format!("turn:{} status:{}", turn.turn_id.0, turn.status));
        for item in &turn.items {
            shape.push(match &item.payload {
                ItemPayload::ToolCall {
                    tool, arguments, ..
                } => format!("call:{tool}:{arguments}"),
                ItemPayload::ToolResult { content, .. } => format!("result:{content}"),
                other => format!("{}:{}", other.kind(), item.status),
            });
        }
        shape.push(format!("usage:{:?}", turn.usage));
    }
    shape
}

#[tokio::test]
async fn repeated_trials_of_a_task_are_compared_from_their_traces() {
    let dir = tempfile::tempdir().unwrap();
    let task = task("read_and_answer");
    let report = run_task(task, &test_config(3), dir.path()).await.unwrap();

    let runs: Vec<RunTrace> = (0..3)
        .map(|trial| RunTrace::read(&RunTrace::path(dir.path(), task.name, trial)).unwrap())
        .collect();

    // With fixed latencies every trial runs the same run: same items, same
    // calls, same arguments, same results, same tokens.
    assert_eq!(shape(&runs[0]), shape(&runs[1]));
    assert_eq!(shape(&runs[1]), shape(&runs[2]));

    // The report is a function of the traces alone, so recomputing it from the
    // files gives the same report.
    assert_eq!(TaskReport::of(&runs), report);

    assert_eq!(report.task, "read_and_answer");
    assert_eq!(report.trials, 3);
    assert_eq!(report.successes, 3);
    assert!(report.errors.is_empty());
    // Three trials of a two-turn task.
    assert_eq!(report.turn_ms.count, 6);
    assert_eq!(report.model_ms.count, 6);
    assert_eq!(report.tool_ms.count, 6);
    assert_eq!(
        report.tokens,
        Usage {
            input_tokens: 3 * (3_900 + 6_300),
            output_tokens: 3 * (200 + 190),
        }
    );

    // Per-tool latency, aggregated over every trial's calls.
    let by_tool: BTreeMap<&str, &psi_core::bench::ToolStats> = report
        .tools
        .iter()
        .map(|stats| (stats.tool.as_str(), stats))
        .collect();
    assert_eq!(
        by_tool.keys().copied().collect::<Vec<_>>(),
        ["read_file", "search"]
    );
    for stats in by_tool.values() {
        assert_eq!(stats.calls, 6);
        assert_eq!(stats.latency.count, 6);
        // The injected 10ms is the floor; the fixture work above it is small.
        assert!(
            stats.latency.median_ms >= 10 && stats.latency.median_ms < 260,
            "{}: {:?}",
            stats.tool,
            stats.latency
        );
    }
    // Every turn's wall time is accounted for on both sides of the split.
    assert!(report.tool_ms.median_ms >= 20);
    assert!(report.model_ms.median_ms >= 15);
}

#[tokio::test]
async fn a_task_that_edits_its_workspace_succeeds_and_is_traced() {
    let dir = tempfile::tempdir().unwrap();
    let task = task("fix_and_test");
    let path = run_trial(task, 0, &test_config(1), dir.path())
        .await
        .unwrap();
    let run = RunTrace::read(&path).unwrap();

    // Success here is a claim about the workspace, not about what was said.
    assert!(run.success);
    assert!(
        std::fs::read_to_string(dir.path().join("fix_and_test-0.workspace/src/lib.sh"))
            .unwrap()
            .contains("echo 42")
    );
    assert_eq!(
        run.tool_calls()
            .iter()
            .map(|call| call.tool.clone())
            .collect::<Vec<_>>(),
        [
            "list_directory",
            "search",
            "read_file",
            "apply_patch",
            "exec"
        ]
    );
    // The mutating calls bumped the workspace revision the speculative cache
    // will be keyed by, and the trace kept it.
    let revisions: Vec<u64> = run
        .items()
        .filter_map(|item| match &item.payload {
            ItemPayload::ToolCall { revision, .. } => Some(revision.0),
            _ => None,
        })
        .collect();
    assert_eq!(revisions, [0, 0, 0, 0, 1]);
}

#[test]
fn an_unterminated_trace_is_refused_rather_than_measured() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.jsonl");
    let trace = TraceWriter::create(&path).unwrap();
    trace
        .write(&TraceRecord::Run {
            task: "read_and_answer".to_string(),
            trial: 0,
            started_at_ms: 1,
        })
        .unwrap();
    trace
        .write(&TraceRecord::TurnStarted {
            seq: 0,
            timestamp_ms: 2,
            turn_id: TurnId(0),
        })
        .unwrap();

    let err = RunTrace::read(&path).unwrap_err();
    assert!(err.to_string().contains("no outcome"), "{err}");
}

#[test]
fn stats_summarize_the_samples_they_were_measured_from() {
    let empty = Stats::of(&[]);
    assert_eq!(empty.count, 0);
    assert_eq!(empty.median_ms, 0);

    let one = Stats::of(&[7]);
    assert_eq!(
        (one.count, one.median_ms, one.p95_ms, one.mean_ms),
        (1, 7, 7, 7.0)
    );

    // Nearest rank, no interpolation: every number reported was measured.
    let four = Stats::of(&[40, 10, 30, 20]);
    assert_eq!(
        (four.count, four.median_ms, four.p95_ms, four.mean_ms),
        (4, 20, 40, 25.0)
    );

    let twenty: Vec<u64> = (1..=20).rev().collect();
    let twenty = Stats::of(&twenty);
    assert_eq!((twenty.median_ms, twenty.p95_ms), (10, 19));
}

#[test]
fn injected_latency_reproduces_the_measured_tail() {
    let profile = LatencyProfile {
        median_ms: 40,
        p95_ms: 2_000,
    };
    let mut stream = LatencyStream::new("read_file", profile);
    let draws: Vec<u64> = (0..10_000).map(|_| stream.next_ms()).collect();
    let stats = Stats::of(&draws);
    assert!(
        (36..=44).contains(&stats.median_ms),
        "median {}",
        stats.median_ms
    );
    assert!(
        (1_800..=2_200).contains(&stats.p95_ms),
        "p95 {}",
        stats.p95_ms
    );

    // A tool's stream is seeded from its name, so the nth call to a tool takes
    // the same time in every trial, and two tools do not share a schedule.
    let repeat: Vec<u64> = (0..8)
        .map(|_| LatencyStream::new("read_file", profile).next_ms())
        .collect();
    assert!(repeat.iter().all(|ms| *ms == draws[0]));
    assert_ne!(LatencyStream::new("search", profile).next_ms(), draws[0]);

    // A profile with no spread is a fixed latency.
    let mut fixed = LatencyStream::new("exec", LatencyProfile::fixed(7));
    assert_eq!([fixed.next_ms(), fixed.next_ms()], [7, 7]);
}

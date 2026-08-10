//! Recorded benchmark tasks. Hand-written scripts judge a predictor against
//! argument choices no model made, so hit rates against them measure script
//! realism as much as predictor quality. A recording replaces the script with
//! a real model's own trajectory: the recorder drives the live agent once over
//! a snapshotted fixture, and the loader turns the stored session into a task
//! the existing replay, oracle, and strategy machinery consumes unchanged
//! (docs/design.md, "Benchmark tasks and injected latency").
//!
//! A recording is a directory:
//!
//! ```text
//! recording.json   the task's name and prompts
//! fixture/         the workspace as the live run started
//! final/           the workspace as the live run left it
//! sessions/        the live run's session log, exactly as the harness wrote it
//! trace.jsonl      the live run's trace: per-turn usage for the replay, the
//!                  rest for later analysis
//! ```
//!
//! Replay is timed by the recording rather than by injected distributions: the
//! script carries each response's real generation delay, and every tool call
//! is stretched to its recorded duration — authoritative calls by their
//! recorded call ids, guesses by canonical identity — so a speculative
//! execution costs exactly what the call it stands for cost live, and a run's
//! measurements cannot be shifted by how many guesses ran or what became of
//! them.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::fake::FakeResponse;
use crate::hook::HookRegistry;
use crate::item::{CompletionStatus, Item, ItemPayload};
use crate::model::{ModelBackend, ModelEvent, ToolCallRequest, Usage};
use crate::speculation::canonical_json;
use crate::store::SessionStore;
use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolRegistry, ToolSpec};
use crate::tools::default_profile;
use crate::trace::{TraceRecord, TraceWriter};
use crate::{Harness, HarnessConfig};

use super::task::{BenchTask, ProfileFn, ScriptFn, SuccessFn, Timing};
use super::{create_session, now_ms, run_prompt};

/// The recording's own half of the task: what the fixture and session cannot
/// say. Everything else is loaded from the artifacts beside it.
#[derive(Debug, Serialize, Deserialize)]
struct RecordingMeta {
    name: String,
    prompts: Vec<String>,
}

/// Drives the live agent once and stores the run as a recording at `out`,
/// which must not already exist. Every turn must complete; a recording of an
/// interrupted run would replay an interruption, so anything less fails and
/// the partial directory is left behind for inspection.
pub async fn record_task(
    name: &str,
    fixture: &Path,
    prompts: &[String],
    model: Arc<dyn ModelBackend>,
    out: &Path,
) -> io::Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(out)?; // Refuses to overwrite an existing recording.
    // An output inside the fixture would be swept up by the snapshot below —
    // copying the recording into itself until the disk gives out.
    let out_root = std::fs::canonicalize(out)?;
    let fixture_root = std::fs::canonicalize(fixture)?;
    if out_root.starts_with(&fixture_root) {
        std::fs::remove_dir(out)?;
        return Err(io::Error::other(format!(
            "the recording output {} sits inside the fixture {}",
            out_root.display(),
            fixture_root.display()
        )));
    }
    let workspace = out.join("workspace");
    copy_dir(fixture, &out.join("fixture"))?;
    copy_dir(fixture, &workspace)?;

    let trace = TraceWriter::create(&out.join("trace.jsonl"))?;
    trace.write(&TraceRecord::Run {
        task: name.to_string(),
        trial: 0,
        started_at_ms: now_ms(),
    })?;
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model,
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace: workspace.clone(),
        sessions_dir: out.join("sessions"),
        trace: Some(trace.clone()),
        speculation: None,
    })?;

    let session_id = create_session(&commands, &mut events).await;
    for prompt in prompts {
        let (status, _) = run_prompt(&commands, &mut events, &session_id, prompt).await;
        if status != CompletionStatus::Completed {
            return Err(io::Error::other(format!(
                "recording aborted: a turn ended {status}; the partial recording is at {}",
                out.display()
            )));
        }
    }
    trace.write(&TraceRecord::Outcome { success: true })?;

    copy_dir(&workspace, &out.join("final"))?;
    let meta = RecordingMeta {
        name: name.to_string(),
        prompts: prompts.to_vec(),
    };
    std::fs::write(
        out.join("recording.json"),
        serde_json::to_vec_pretty(&meta).map_err(io::Error::other)?,
    )
}

/// Loads a recording as a benchmark task: the snapshot as the fixture, the
/// session replayed as the script with its recorded generation delays, tools
/// timed by their recorded durations, and success meaning the replayed
/// workspace ends byte-for-byte where the live run ended.
pub fn recorded_task(dir: &Path) -> io::Result<BenchTask> {
    let meta: RecordingMeta = serde_json::from_slice(&std::fs::read(dir.join("recording.json"))?)
        .map_err(io::Error::other)?;
    let fixture = read_files(&dir.join("fixture"))?;

    let store = SessionStore::new(dir.join("sessions"))?;
    let recorded =
        store.list().first().cloned().ok_or_else(|| {
            io::Error::other(format!("no session in recording {}", dir.display()))
        })?;
    let (snapshot, _log) = store.load(&recorded.id)?;

    let usages = recorded_usages(&dir.join("trace.jsonl"))?;
    let template = Arc::new(script_from_items(&snapshot.items, &usages)?);
    let durations = RecordedDurations::of(&snapshot.items);
    let finals = read_files(&dir.join("final"))?;

    Ok(BenchTask {
        name: meta.name,
        fixture,
        profile: Arc::new(move |workspace: PathBuf| {
            replay_durations(default_profile(workspace), durations.clone())
        }) as ProfileFn,
        prompts: meta.prompts,
        script: Arc::new(move || template.iter().cloned().collect()) as ScriptFn,
        success: Arc::new(move |workspace: &Path, _answers: &[String]| {
            matches_final_state(workspace, &finals)
        }) as SuccessFn,
        timing: Timing::Recorded,
    })
}

/// The per-turn usage the live run reported, in turn order. Usage is never
/// persisted as an item, so it is read back from the trace the recorder keeps:
/// each turn's total rides on its `TurnFinished` record.
fn recorded_usages(path: &Path) -> io::Result<Vec<Option<Usage>>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter_map(|line| serde_json::from_str::<TraceRecord>(line).ok())
        .filter_map(|record| match record {
            TraceRecord::TurnFinished { usage, .. } => Some(usage),
            _ => None,
        })
        .collect())
}

/// Rebuilds the model's responses from a recorded session's items.
///
/// The item log preserves stream order, and a round's tool results are the
/// last items it appends, so the boundaries are recoverable: a streamed item
/// after a tool result begins the next response, and a user message begins the
/// next turn. Each response carries its real generation delay — the time from
/// the item that preceded it to its last streamed item, which is what the
/// authoritative model actually spent.
///
/// `usages` is each turn's recorded token total, in turn order. The engine
/// sums usage over a turn's responses, so the whole total rides on the turn's
/// last response and the replayed turn reports what the live one billed.
///
/// The conversion refuses interrupted recordings: a cancelled item, a failed
/// item that is not a tool's own error, or a call whose arguments never
/// finished all mean the recording is not a complete run.
pub fn script_from_items(
    items: &[Item],
    usages: &[Option<Usage>],
) -> io::Result<Vec<FakeResponse>> {
    let mut script = Vec::new();
    let mut events: Vec<ModelEvent> = Vec::new();
    let mut saw_result = false;
    // Which turn the response being built belongs to, indexing `usages`.
    let mut turn = 0usize;
    // `created_at_ms` of the item preceding the open response, and of the
    // response's last streamed item: their distance is the generation delay.
    let mut base_ms: Option<u64> = None;
    let mut last_streamed_ms = 0u64;
    let mut previous_ms = 0u64;

    let flush = |events: &mut Vec<ModelEvent>,
                 base_ms: u64,
                 last_streamed_ms: u64,
                 usage: Option<Usage>,
                 script: &mut Vec<_>| {
        if events.is_empty() {
            return;
        }
        let mut response = std::mem::take(events);
        if let Some(usage) = usage {
            response.push(ModelEvent::Usage { usage });
        }
        response.push(ModelEvent::Completed);
        script.push(FakeResponse::new(response).delayed(last_streamed_ms.saturating_sub(base_ms)));
    };

    for item in items {
        let interrupted = item.status == CompletionStatus::Cancelled
            || (item.status == CompletionStatus::Failed
                && item.payload.kind() != crate::item::ItemKind::ToolResult);
        if interrupted {
            return Err(io::Error::other(format!(
                "the recording holds an interrupted item ({} {}), not a complete run",
                item.payload.kind(),
                item.status
            )));
        }
        match &item.payload {
            ItemPayload::UserMessage { .. } => {
                // A user message closes the previous turn, so the response
                // flushed here is that turn's last and carries its usage.
                if !events.is_empty() {
                    let base = base_ms.unwrap_or(item.created_at_ms);
                    let usage = usages.get(turn).copied().flatten();
                    flush(&mut events, base, last_streamed_ms, usage, &mut script);
                    turn += 1;
                }
                saw_result = false;
                base_ms = Some(item.created_at_ms);
            }
            ItemPayload::ToolResult { .. } => {
                saw_result = true;
            }
            streamed => {
                let Some(base) = base_ms else {
                    return Err(io::Error::other(
                        "the recording does not start with a user message",
                    ));
                };
                if saw_result {
                    // A streamed item after a result: the next response began
                    // mid-turn, so no usage rides on the one being closed.
                    flush(&mut events, base, last_streamed_ms, None, &mut script);
                    saw_result = false;
                    base_ms = Some(previous_ms);
                }
                match streamed {
                    ItemPayload::Reasoning { text, .. } if !text.is_empty() => {
                        events.push(ModelEvent::ReasoningDelta {
                            delta: text.clone(),
                        });
                    }
                    // Reasoning that streamed no text was provider data alone,
                    // which a scripted model cannot replay.
                    ItemPayload::Reasoning { .. } => {}
                    ItemPayload::AssistantMessage { text } if !text.is_empty() => {
                        events.push(ModelEvent::TextDelta {
                            delta: text.clone(),
                        });
                    }
                    ItemPayload::AssistantMessage { .. } => {}
                    ItemPayload::ToolCall {
                        tool,
                        call_id,
                        arguments,
                        ..
                    } => {
                        if arguments.is_null() {
                            return Err(io::Error::other(format!(
                                "the recording holds a call whose arguments never finished: {tool}"
                            )));
                        }
                        events.push(ModelEvent::ToolCallCompleted {
                            call: ToolCallRequest {
                                call_id: call_id.clone(),
                                tool: tool.clone(),
                                arguments: arguments.clone(),
                            },
                        });
                    }
                    _ => unreachable!("user messages and results are matched above"),
                }
                last_streamed_ms = item.created_at_ms;
            }
        }
        previous_ms = item.created_at_ms;
    }
    flush(
        &mut events,
        base_ms.unwrap_or(0),
        last_streamed_ms,
        usages.get(turn).copied().flatten(),
        &mut script,
    );
    Ok(script)
}

/// The recorded duration of every call. The replayed script carries the
/// recording's own call ids, so an authoritative call looks up exactly the
/// duration that call took live; a speculative guess carries an id the
/// recording never issued and falls back to the call's canonical identity —
/// the cache key's tool and canonical arguments — costing what the (first)
/// call it stands for cost live. Deliberately stateless: whatever runs, in
/// whatever order, however it ends — baselines, speculative executions,
/// discarded guesses — every trial replays identical timings.
#[derive(Clone)]
pub struct RecordedDurations {
    by_call: Arc<HashMap<String, u64>>,
    by_identity: Arc<HashMap<String, u64>>,
}

impl RecordedDurations {
    pub fn of(items: &[Item]) -> Self {
        let mut identities: HashMap<&str, String> = HashMap::new();
        for item in items {
            if let ItemPayload::ToolCall {
                tool,
                call_id,
                arguments,
                ..
            } = &item.payload
            {
                identities.insert(call_id, identity(tool, arguments));
            }
        }
        let mut by_call: HashMap<String, u64> = HashMap::new();
        let mut by_identity: HashMap<String, u64> = HashMap::new();
        for item in items {
            if let ItemPayload::ToolResult {
                call_id,
                duration_ms,
                ..
            } = &item.payload
                && let Some(identity) = identities.get(call_id.as_str())
            {
                by_call.insert(call_id.clone(), *duration_ms);
                by_identity.entry(identity.clone()).or_insert(*duration_ms);
            }
        }
        Self {
            by_call: Arc::new(by_call),
            by_identity: Arc::new(by_identity),
        }
    }

    /// The recorded duration for one call: exact for a recorded call id, the
    /// identity's first recorded duration for a guess. A call the recording
    /// never made gets no added time: a wrong guess runs at fixture speed,
    /// and its waste is counted in tokens, not wall time.
    fn recorded_ms(&self, call_id: &str, identity: &str) -> Option<u64> {
        self.by_call
            .get(call_id)
            .or_else(|| self.by_identity.get(identity))
            .copied()
    }
}

fn identity(tool: &str, arguments: &serde_json::Value) -> String {
    format!("{tool} {}", canonical_json(arguments))
}

/// A tool that takes as long as the recording says this call took: it runs
/// the real tool, then sleeps whatever remains of the recorded duration. The
/// recorded time already contains the live execution, so it is a total to
/// reach, not a surcharge — sleeping the whole of it on top of the replayed
/// execution would double-count exactly the calls whose time matters most.
/// The engine measures a call around the whole future, so the replayed time
/// lands in the `tool_result` duration exactly as the live time did.
struct ReplayedTool {
    inner: Arc<dyn Tool>,
    durations: RecordedDurations,
}

impl Tool for ReplayedTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn effect(&self) -> ToolEffect {
        self.inner.effect()
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        let key = identity(&self.inner.spec().name, &invocation.arguments);
        let delay = self.durations.recorded_ms(&invocation.call_id, &key);
        let inner = self.inner.clone();
        Box::pin(async move {
            let started = tokio::time::Instant::now();
            let result = inner.execute(invocation).await;
            if let Some(ms) = delay {
                let recorded = Duration::from_millis(ms);
                let elapsed = started.elapsed();
                if recorded > elapsed {
                    tokio::time::sleep(recorded - elapsed).await;
                }
            }
            result
        })
    }
}

/// Wraps every tool of a profile in its recorded durations, keeping specs and
/// effects untouched — the recorded counterpart of `inject_latency`.
pub fn replay_durations(registry: ToolRegistry, durations: RecordedDurations) -> ToolRegistry {
    let mut replayed = ToolRegistry::new();
    for tool in registry.tools() {
        replayed.register(ReplayedTool {
            inner: tool.clone(),
            durations: durations.clone(),
        });
    }
    replayed
}

/// True when the workspace holds exactly the recorded final files, byte for
/// byte — the strongest claim a replay can make without re-judging anything.
fn matches_final_state(workspace: &Path, finals: &[(String, Vec<u8>)]) -> bool {
    let Ok(replayed) = read_files(workspace) else {
        return false;
    };
    replayed.len() == finals.len()
        && finals.iter().all(|(path, bytes)| {
            replayed
                .iter()
                .any(|(other, contents)| other == path && contents == bytes)
        })
}

/// Every file under `root`, as workspace-relative paths, sorted for
/// deterministic fixtures. Paths and bytes only — file modes are not carried
/// through a replay, so a fixture whose scripts must run invokes them through
/// an interpreter (`sh test.sh`), not an execute bit.
fn read_files(root: &Path) -> io::Result<Vec<(String, Vec<u8>)>> {
    fn walk(root: &Path, dir: &Path, files: &mut Vec<(String, Vec<u8>)>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, files)?;
            } else {
                let relative = path
                    .strip_prefix(root)
                    .map_err(io::Error::other)?
                    .to_string_lossy()
                    .into_owned();
                files.push((relative, std::fs::read(&path)?));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(root, root, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

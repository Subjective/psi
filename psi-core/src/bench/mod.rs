//! Non-speculative baselines (docs/design.md, Milestone 5). A benchmark runs
//! a deterministic task — a fixture workspace, a scripted model, the real
//! five-tool profile with injected latency — writes one trace per trial, and
//! reports the trials by reading those traces back. Nothing measured comes
//! from anywhere but the traces, which is what makes a baseline reproducible
//! from its artifacts and what Milestone 6 will be compared against.
//!
//! This module and its `psi-bench` binary are the dev-side surface: the `psi`
//! binary is a different package, so nothing here ships with it.

mod latency;
mod oracle;
mod report;
mod task;

pub use latency::{Latency, LatencyProfile, LatencyStream, inject_latency};
pub use oracle::ReplayOracle;
pub use report::{SpeculationStats, Stats, TaskReport, ToolStats};
pub use task::{BenchTask, tasks};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;

use crate::fake::FakeModel;
use crate::hook::HookRegistry;
use crate::item::{CompletionStatus, ItemPayload};
use crate::protocol::{Command, Event, EventPayload};
use crate::session::SessionId;
use crate::speculation::{SpeculationConfig, v0_allowlist};
use crate::tools::default_profile;
use crate::trace::{RunTrace, TraceRecord, TraceWriter};
use crate::{Harness, HarnessConfig};

/// How a benchmark run is timed. Latency is the independent variable of the
/// baseline: everything else about a task is fixed.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub trials: u32,
    pub latency: Latency,
    /// Wall time every scripted model response spends before it streams. Real
    /// sessions spend around 82% of a turn's wall time here, and it is the
    /// time Milestone 6 speculates into, so a baseline without it would leave
    /// nothing to measure against. The default puts a task near that split
    /// against the default tool profile.
    pub model_delay_ms: u64,
    /// `Some(execution budget)` drives the run with the replay oracle over the
    /// v0 allowlist — the perfect-prediction ceiling. `None` is the baseline.
    pub speculate: Option<usize>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            trials: 5,
            latency: Latency::measured(),
            model_delay_ms: 2_000,
            speculate: None,
        }
    }
}

/// Runs repeated trials of one task and reports them from their traces.
pub async fn run_task(
    task: &BenchTask,
    config: &BenchConfig,
    dir: &Path,
) -> std::io::Result<TaskReport> {
    let mut traces = Vec::new();
    for trial in 0..config.trials {
        let path = run_trial(task, trial, config, dir).await?;
        traces.push(RunTrace::read(&path)?);
    }
    Ok(TaskReport::of(&traces))
}

/// Runs one trial against a fresh fixture workspace and returns its trace's
/// path. The workspace, the session log, and the trace all live under `dir`,
/// so a benchmark never touches the user's sessions and leaves every artifact
/// of a run in one place.
pub async fn run_trial(
    task: &BenchTask,
    trial: u32,
    config: &BenchConfig,
    dir: &Path,
) -> std::io::Result<PathBuf> {
    let workspace = dir.join(format!("{}-{trial}.workspace", task.name));
    lay_out_fixture(&workspace, task.fixture)?;

    let path = RunTrace::path(dir, task.name, trial);
    let trace = TraceWriter::create(&path)?;
    trace.write(&TraceRecord::Run {
        task: task.name.to_string(),
        trial,
        started_at_ms: now_ms(),
    })?;

    let script: Vec<_> = (task.script)()
        .into_iter()
        .map(|response| response.delayed(config.model_delay_ms))
        .collect();
    // The oracle reads the same script the model plays, so its guesses are
    // exactly the recording's next calls.
    let speculation = config.speculate.map(|execution_budget| SpeculationConfig {
        predictor: Arc::new(ReplayOracle::for_script(&script)),
        allowlist: v0_allowlist(),
        execution_budget,
    });
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(FakeModel::new(script)),
        tools: inject_latency(default_profile(workspace.clone()), &config.latency),
        hooks: HookRegistry::new(),
        workspace: workspace.clone(),
        sessions_dir: dir.join("sessions"),
        trace: Some(trace.clone()),
        speculation,
    })?;

    let session_id = create_session(&commands, &mut events).await;
    let mut completed = true;
    let mut answers = Vec::new();
    for prompt in task.prompts {
        let (status, answer) = run_prompt(&commands, &mut events, &session_id, prompt).await;
        completed &= status == CompletionStatus::Completed;
        answers.push(answer);
    }

    let success = completed && (task.success)(&workspace, &answers);
    trace.write(&TraceRecord::Outcome { success })?;
    Ok(path)
}

/// Writes a task's fixture into an empty directory, replacing whatever an
/// earlier trial left, so every trial starts from the same workspace.
fn lay_out_fixture(workspace: &Path, fixture: &[(&str, &str)]) -> std::io::Result<()> {
    match std::fs::remove_dir_all(workspace) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    std::fs::create_dir_all(workspace)?;
    for (path, contents) in fixture {
        let path = workspace.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
    }
    Ok(())
}

async fn create_session(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
) -> SessionId {
    commands.send(Command::CreateSession).await.expect("engine");
    match events.recv().await.map(|event| event.payload) {
        Some(EventPayload::SessionCreated { meta }) => meta.id,
        other => panic!("expected session_created, got {other:?}"),
    }
}

/// Submits one prompt and drains the turn, returning how it ended and the
/// assistant message it ended with. The trace records of a turn are written
/// before the events that carry them, so the trace already holds the turn by
/// the time this returns.
async fn run_prompt(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
    session_id: &SessionId,
    prompt: &str,
) -> (CompletionStatus, String) {
    commands
        .send(Command::SubmitMessage {
            session_id: session_id.clone(),
            text: prompt.to_string(),
        })
        .await
        .expect("engine");
    let mut answer = String::new();
    while let Some(event) = events.recv().await {
        match event.payload {
            EventPayload::ItemFinished { item } => {
                if let ItemPayload::AssistantMessage { text } = item.payload {
                    answer = text;
                }
            }
            EventPayload::TurnFinished { status, .. } => return (status, answer),
            _ => {}
        }
    }
    panic!("the harness stopped mid-turn");
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis() as u64
}

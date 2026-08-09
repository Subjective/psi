//! Benchmarks (docs/design.md, Milestones 5 to 7). A benchmark runs a
//! deterministic task — a fixture workspace, a scripted model, a real tool
//! profile with injected latency — writes one trace per trial, and reports the
//! trials by reading those traces back. Nothing measured comes from anywhere
//! but the traces, which is what makes a run reproducible from its artifacts.
//!
//! A run either speculates or does not, and a speculating one names its
//! strategy and its two budgets. The authoritative model stays scripted in
//! every case, so a comparison between two runs varies the predictor and
//! nothing else.
//!
//! This module and its `psi-bench` binary are the dev-side surface: the `psi`
//! binary is a different package, so nothing here ships with it.

mod latency;
mod oracle;
mod record;
mod report;
mod task;

pub use latency::{Latency, LatencyProfile, LatencyStream, inject_latency};
pub use oracle::ReplayOracle;
pub use record::{
    RecordedDurations, record_task, recorded_task, replay_durations, script_from_items,
};
pub use report::{Comparison, SpeculationStats, Stats, TaskReport, ToolStats};
pub use task::{BenchTask, Timing, tasks};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;

use crate::fake::{FakeModel, FakeResponse};
use crate::hook::HookRegistry;
use crate::item::{CompletionStatus, ItemPayload};
use crate::model::ModelBackend;
use crate::predictor::{BranchSampling, DirectProposal};
use crate::protocol::{Command, Event, EventPayload};
use crate::session::SessionId;
use crate::speculation::{Predictor, SpeculationConfig, v0_allowlist};
use crate::trace::{RunTrace, TraceRecord, TraceWriter};
use crate::vllm::{VllmBackend, VllmConfig};
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
    /// How the run predicts. `None` is the non-speculative baseline.
    pub speculate: Option<Speculation>,
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

/// How a speculating run guesses. The strategy is per-run configuration, and
/// the two budgets are the research question's independent variables: fixing
/// both is what makes the strategies comparable, because branch sampling
/// spends far more predictor tokens per guess than direct proposal (docs/
/// design.md, "Speculation").
#[derive(Debug, Clone)]
pub struct Speculation {
    pub strategy: Strategy,
    /// Predictor tokens one round may spend guessing.
    pub prediction_budget: u64,
    /// Concurrent speculative calls per round — the fanout.
    pub execution_budget: usize,
}

/// Which predictor drives a run. The two real strategies carry the vLLM target
/// they ask; the oracle asks nothing.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// The replay oracle: always right and free, so its run is the ceiling a
    /// real strategy is measured against.
    Oracle,
    /// One predictor request for the calls the assistant will make next.
    Direct { predictor: VllmConfig },
    /// `samples` temperature-sampled continuations, ranked by agreement.
    Branch {
        predictor: VllmConfig,
        samples: usize,
    },
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
    lay_out_fixture(&workspace, &task.fixture)?;

    let path = RunTrace::path(dir, &task.name, trial);
    let trace = TraceWriter::create(&path)?;
    trace.write(&TraceRecord::Run {
        task: task.name.clone(),
        trial,
        started_at_ms: now_ms(),
    })?;

    // A hand-written task is timed by the run config; a recorded one carries
    // its own generation delays and tool durations (task::Timing).
    let script: Vec<_> = match task.timing {
        task::Timing::Injected => (task.script)()
            .into_iter()
            .map(|response| response.delayed(config.model_delay_ms))
            .collect(),
        task::Timing::Recorded => (task.script)(),
    };
    let speculation = match &config.speculate {
        Some(speculate) => Some(SpeculationConfig {
            predictor: predictor(&speculate.strategy, &script, &workspace)?,
            allowlist: v0_allowlist(),
            prediction_budget: speculate.prediction_budget,
            execution_budget: speculate.execution_budget,
        }),
        None => None,
    };
    let tools = match task.timing {
        task::Timing::Injected => {
            inject_latency((task.profile)(workspace.clone()), &config.latency)
        }
        // A recorded task's profile already wraps its tools in their recorded
        // durations, keyed by call identity rather than drawn in call order.
        task::Timing::Recorded => (task.profile)(workspace.clone()),
    };
    let (commands, mut events) = Harness::spawn(HarnessConfig {
        model: Arc::new(FakeModel::new(script)),
        tools,
        hooks: HookRegistry::new(),
        workspace: workspace.clone(),
        sessions_dir: dir.join("sessions"),
        trace: Some(trace.clone()),
        speculation,
    })?;

    let session_id = create_session(&commands, &mut events).await;
    let mut completed = true;
    let mut answers = Vec::new();
    for prompt in &task.prompts {
        let (status, answer) = run_prompt(&commands, &mut events, &session_id, prompt).await;
        completed &= status == CompletionStatus::Completed;
        answers.push(answer);
    }

    let success = completed && (task.success)(&workspace, &answers);
    trace.write(&TraceRecord::Outcome { success })?;
    Ok(path)
}

/// Builds the run's predictor. The oracle reads the same script the model
/// plays, so its guesses are exactly the recording's next calls; the two real
/// strategies read a vLLM target instead.
fn predictor(
    strategy: &Strategy,
    script: &[FakeResponse],
    workspace: &Path,
) -> std::io::Result<Arc<dyn Predictor>> {
    Ok(match strategy {
        Strategy::Oracle => Arc::new(ReplayOracle::for_script(script)),
        Strategy::Direct { predictor } => {
            Arc::new(DirectProposal::new(backend(predictor, workspace)?))
        }
        Strategy::Branch { predictor, samples } => Arc::new(BranchSampling::new(
            backend(predictor, workspace)?,
            *samples,
        )),
    })
}

/// The predictor's model target. Its instructions name the fixture workspace,
/// because a benchmark's authoritative model is a recording that needs no such
/// help while the predictor has to guess real paths inside a directory whose
/// name changes every trial.
fn backend(config: &VllmConfig, workspace: &Path) -> std::io::Result<Arc<dyn ModelBackend>> {
    let mut config = config.clone();
    config.instructions = format!(
        "{}\n\nThe workspace root is {}.",
        config.instructions,
        workspace.display()
    );
    VllmBackend::new(config)
        .map(|backend| Arc::new(backend) as Arc<dyn ModelBackend>)
        .map_err(std::io::Error::other)
}

/// Writes a task's fixture into an empty directory, replacing whatever an
/// earlier trial left, so every trial starts from the same workspace.
fn lay_out_fixture(workspace: &Path, fixture: &[(String, String)]) -> std::io::Result<()> {
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

pub(crate) async fn create_session(
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
pub(crate) async fn run_prompt(
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

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis() as u64
}

//! Trace export: one JSONL file per measured run, assembled from the event
//! stream as the engine emits it. Every event already carries a sequence
//! number and a timestamp, so this is an assembly step rather than a retrofit
//! (docs/design.md, "Interface protocol").
//!
//! A trace is the only measurement source a benchmark has, which is what makes
//! the numbers reproducible from a stored artifact: it holds the item
//! sequence, turn boundaries and timings, every tool call with its arguments
//! and duration, statuses, and usage. Three consumers read one:
//!
//! - the baseline report, which aggregates repeated trials of one task
//!   (`crate::bench::TaskReport`);
//! - Milestone 6's speculation records — predictions, executions, hits,
//!   misses, wasted work — which join the same file as further `TraceRecord`
//!   variants, stamped from the same sequence counter and the same clock;
//! - Milestone 6's replay oracle, which replays a recorded run's tool calls in
//!   order (`RunTrace::tool_calls`).
//!
//! The two framing records come from whoever runs the harness: `run` opens the
//! file and names the trial, `outcome` closes it with the task's success
//! criterion. Everything between them is the event stream. A trace missing
//! either one is incomplete and is refused rather than measured.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::item::{CompletionStatus, Item, ItemPayload, TurnId, WorkspaceRevision};
use crate::model::Usage;
use crate::protocol::EventPayload;

/// One line of a trace. The sequenced variants carry the `seq` and
/// `timestamp_ms` of the event they were assembled from, so a record type
/// added later — Milestone 6's speculation records — interleaves with them on
/// one clock without the format changing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum TraceRecord {
    /// The header: which task and which trial of it this run is, so a report
    /// can group repeated trials. Written before the harness emits anything,
    /// so it is one of the two records with no sequence number.
    Run {
        task: String,
        trial: u32,
        started_at_ms: u64,
    },
    TurnStarted {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
    },
    /// One durable item, exactly as `item_finished` carried it.
    Item {
        seq: u64,
        timestamp_ms: u64,
        item: Item,
    },
    TurnFinished {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
        status: CompletionStatus,
        error: Option<String>,
        usage: Option<Usage>,
    },
    /// The predictor's guesses for one model response, before filtering, so
    /// prediction quality is measurable apart from what the budget let run.
    Prediction {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
        calls: Vec<PredictedCall>,
        /// Tokens the predictor spent on this round. The report sums these
        /// into predictor cost, the price the hit rate has to earn back; the
        /// replay oracle spends none.
        usage: Usage,
        /// Why the round came back short, when it did. The report prints these
        /// so a run whose predictor is failing does not read as one whose
        /// predictor is merely wrong.
        error: Option<String>,
    },
    /// One guess the runtime started executing: allowlisted, uncached,
    /// unblocked, and within the execution budget.
    SpeculativeExecution {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
        tool: String,
        arguments: serde_json::Value,
        revision: WorkspaceRevision,
    },
    /// How one authoritative call met the cache: a hit adopted the entry
    /// (`finished` says whether it was still in flight), a miss executed
    /// normally. Hit rate is the core metric and this is its record.
    Reconciliation {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
        tool: String,
        arguments: serde_json::Value,
        hit: bool,
        finished: Option<bool>,
    },
    /// A cache entry that never served: wasted work, the cost side of the
    /// research question.
    SpeculativeDiscard {
        seq: u64,
        timestamp_ms: u64,
        turn_id: TurnId,
        tool: String,
        arguments: serde_json::Value,
        finished: bool,
        reason: DiscardReason,
    },
    /// The terminator: whether the run met its task's success criterion.
    Outcome { success: bool },
}

/// One guessed call, as the predictor proposed it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedCall {
    pub tool: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscardReason {
    /// A mutation bumped the revision out from under the entry.
    Invalidated,
    /// The turn ended with the entry still parked.
    Unused,
}

/// The append end of one trace. Two writers share it — the engine, which
/// records the event stream as it is emitted, and the benchmark runner, which
/// frames the run — so it is cloneable and internally locked.
#[derive(Clone)]
pub struct TraceWriter {
    file: Arc<Mutex<File>>,
}

impl TraceWriter {
    /// Creates the trace file, and the directory holding it if it is missing.
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            file: Arc::new(Mutex::new(File::create(path)?)),
        })
    }

    /// Appends one record. The line is built in memory and written with a
    /// single call, so two writers cannot interleave one line.
    pub fn write(&self, record: &TraceRecord) -> io::Result<()> {
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        self.file.lock().expect("trace lock").write_all(&line)
    }

    /// Records one emitted event. Events that carry no measurement are
    /// dropped: session lifecycle, and the streaming deltas whose content the
    /// finished item already holds.
    ///
    /// A failed write is dropped, because nothing in the interface protocol
    /// can report it. The runner writes the closing `outcome` record through
    /// this same file, so a disk that has stopped accepting writes fails the
    /// run rather than quietly shortening its trace.
    pub fn record_event(&self, seq: u64, timestamp_ms: u64, payload: &EventPayload) {
        let record = match payload {
            EventPayload::TurnStarted { turn_id } => TraceRecord::TurnStarted {
                seq,
                timestamp_ms,
                turn_id: *turn_id,
            },
            EventPayload::ItemFinished { item } => TraceRecord::Item {
                seq,
                timestamp_ms,
                item: item.clone(),
            },
            EventPayload::TurnFinished {
                turn_id,
                status,
                error,
                usage,
            } => TraceRecord::TurnFinished {
                seq,
                timestamp_ms,
                turn_id: *turn_id,
                status: *status,
                error: error.clone(),
                usage: *usage,
            },
            _ => return,
        };
        let _ = self.write(&record);
    }
}

/// One run, reconstructed from its trace.
#[derive(Debug, Clone)]
pub struct RunTrace {
    pub task: String,
    pub trial: u32,
    pub started_at_ms: u64,
    /// In the order the run ran them.
    pub turns: Vec<TurnTrace>,
    /// Whether the run met its task's success criterion, so a baseline reports
    /// success alongside latency.
    pub success: bool,
}

/// One turn of a run: its boundaries, what it produced, and what it cost.
#[derive(Debug, Clone)]
pub struct TurnTrace {
    pub turn_id: TurnId,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub status: CompletionStatus,
    /// Why the turn failed, for a trial whose success column reads zero.
    pub error: Option<String>,
    /// Tokens billed across the turn's model responses, which is what a
    /// baseline costs a run by.
    pub usage: Option<Usage>,
    /// Every item of the turn, in append order.
    pub items: Vec<Item>,
    /// The turn's speculation records, in stamp order — empty for a baseline.
    /// `crate::bench::SpeculationStats` aggregates them.
    pub speculation: Vec<TraceRecord>,
}

/// One tool call and what it cost, paired from the trace's `tool_call` and
/// `tool_result` items. The unit of the per-tool latency stats, and the unit
/// Milestone 6's replay oracle predicts.
#[derive(Debug, Clone, PartialEq)]
pub struct TracedCall {
    pub tool: String,
    pub arguments: serde_json::Value,
    pub duration_ms: u64,
}

impl RunTrace {
    /// Reads a complete trace. Anything unparseable, out of order, or missing
    /// is an error rather than a shorter run: a partial trace measured as if
    /// it were whole is a wrong number, which is worse than no number.
    pub fn read(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut run: Option<RunTrace> = None;
        // The turn being assembled: its id, when it started, its items, and
        // its speculation records.
        let mut open: Option<(TurnId, u64, Vec<Item>, Vec<TraceRecord>)> = None;
        let mut closed = false;

        for (index, line) in text.lines().enumerate() {
            let defect = |message: &str| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}:{}: {message}", path.display(), index + 1),
                )
            };
            let record: TraceRecord =
                serde_json::from_str(line).map_err(|err| defect(&err.to_string()))?;
            if closed {
                return Err(defect("a record after the outcome"));
            }
            match (record, run.as_mut()) {
                (
                    TraceRecord::Run {
                        task,
                        trial,
                        started_at_ms,
                    },
                    None,
                ) => {
                    run = Some(RunTrace {
                        task,
                        trial,
                        started_at_ms,
                        turns: Vec::new(),
                        success: false,
                    });
                }
                (
                    TraceRecord::TurnStarted {
                        timestamp_ms,
                        turn_id,
                        ..
                    },
                    Some(_),
                ) if open.is_none() => {
                    open = Some((turn_id, timestamp_ms, Vec::new(), Vec::new()));
                }
                (TraceRecord::Item { item, .. }, Some(_)) => match open.as_mut() {
                    Some((_, _, items, _)) => items.push(item),
                    None => return Err(defect("an item outside a turn")),
                },
                (
                    record @ (TraceRecord::Prediction { .. }
                    | TraceRecord::SpeculativeExecution { .. }
                    | TraceRecord::Reconciliation { .. }
                    | TraceRecord::SpeculativeDiscard { .. }),
                    Some(_),
                ) => match open.as_mut() {
                    Some((_, _, _, speculation)) => speculation.push(record),
                    None => return Err(defect("a speculation record outside a turn")),
                },
                (
                    TraceRecord::TurnFinished {
                        timestamp_ms,
                        turn_id,
                        status,
                        error,
                        usage,
                        ..
                    },
                    Some(run),
                ) => {
                    let Some((open_id, started_at_ms, items, speculation)) = open.take() else {
                        return Err(defect("a turn finished that never started"));
                    };
                    if open_id != turn_id {
                        return Err(defect("a turn finished under another turn's id"));
                    }
                    run.turns.push(TurnTrace {
                        turn_id,
                        started_at_ms,
                        finished_at_ms: timestamp_ms,
                        status,
                        error,
                        usage,
                        items,
                        speculation,
                    });
                }
                (TraceRecord::Outcome { success }, Some(run)) if open.is_none() => {
                    run.success = success;
                    closed = true;
                }
                _ => return Err(defect("a record out of order")),
            }
        }
        match run {
            Some(run) if closed => Ok(run),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: incomplete trace, no outcome", path.display()),
            )),
        }
    }

    /// Every item of the run, in the order it appended them.
    pub fn items(&self) -> impl Iterator<Item = &Item> {
        self.turns.iter().flat_map(|turn| turn.items.iter())
    }

    /// Every tool call of the run in call order. A call whose result never
    /// landed is dropped: it has no duration to measure and no result to
    /// replay.
    pub fn tool_calls(&self) -> Vec<TracedCall> {
        let mut results: HashMap<&str, u64> = HashMap::new();
        for item in self.items() {
            if let ItemPayload::ToolResult {
                call_id,
                duration_ms,
                ..
            } = &item.payload
            {
                results.insert(call_id, *duration_ms);
            }
        }
        self.items()
            .filter_map(|item| match &item.payload {
                ItemPayload::ToolCall {
                    tool,
                    call_id,
                    arguments,
                    ..
                } => Some(TracedCall {
                    tool: tool.clone(),
                    arguments: arguments.clone(),
                    duration_ms: results.get(call_id.as_str()).copied()?,
                }),
                _ => None,
            })
            .collect()
    }

    /// Where a trial's trace is written under `dir`. One flat file per trial,
    /// so repeated trials of one task sit next to each other.
    pub fn path(dir: &Path, task: &str, trial: u32) -> PathBuf {
        dir.join(format!("{task}-{trial}.jsonl"))
    }
}

impl TurnTrace {
    /// Wall time from `turn_started` to `turn_finished`.
    pub fn wall_ms(&self) -> u64 {
        self.finished_at_ms.saturating_sub(self.started_at_ms)
    }

    /// Time inside tools: the engine measures each call around its own future,
    /// and the turn's calls run one at a time.
    pub fn tool_ms(&self) -> u64 {
        self.items
            .iter()
            .map(|item| match &item.payload {
                ItemPayload::ToolResult { duration_ms, .. } => *duration_ms,
                _ => 0,
            })
            .sum()
    }

    /// Wall time not spent in tools, which is the model generating plus the
    /// engine's own bookkeeping. The bookkeeping is microseconds against tool
    /// and model latencies measured in milliseconds, so this is the model half
    /// of the split a baseline reports.
    pub fn model_ms(&self) -> u64 {
        self.wall_ms().saturating_sub(self.tool_ms())
    }
}

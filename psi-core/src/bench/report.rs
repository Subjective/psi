//! The baseline report: repeated trials of one task, aggregated. Every number
//! here is read out of the trials' traces, so a report is a pure function of
//! the artifacts on disk and can be recomputed long after the run — which is
//! the property Milestone 6's comparison against these baselines rests on.

use std::collections::BTreeMap;
use std::fmt;

use crate::model::Usage;
use crate::trace::{RunTrace, TraceRecord};

/// Speculation across every trial, aggregated from the turns' records. The
/// core metrics of the research question: hit rate and wasted work, read
/// beside the latency change against a baseline report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeculationStats {
    /// Calls the predictor proposed, before filtering.
    pub proposed: usize,
    /// Guesses the runtime executed speculatively.
    pub executed: usize,
    /// Authoritative calls served from the cache.
    pub hits: usize,
    /// Authoritative calls that executed normally.
    pub misses: usize,
    /// Executions that never served: invalidated by a mutation or unused at
    /// turn end.
    pub wasted: usize,
    /// What the predictor billed across every round — predictor cost, the
    /// price the hit rate has to earn back. Zero for a replay-oracle run,
    /// which reads a recording instead of a model.
    pub predictor_tokens: Usage,
}

impl SpeculationStats {
    /// `None` when the traces hold no speculation records — a baseline.
    pub fn of(traces: &[RunTrace]) -> Option<Self> {
        let mut stats = Self::default();
        let mut any = false;
        let records = traces
            .iter()
            .flat_map(|trace| &trace.turns)
            .flat_map(|turn| &turn.speculation);
        for record in records {
            any = true;
            match record {
                TraceRecord::Prediction { calls, usage, .. } => {
                    stats.proposed += calls.len();
                    stats.predictor_tokens.add(*usage);
                }
                TraceRecord::SpeculativeExecution { .. } => stats.executed += 1,
                TraceRecord::Reconciliation { hit, .. } => {
                    if *hit {
                        stats.hits += 1;
                    } else {
                        stats.misses += 1;
                    }
                }
                TraceRecord::SpeculativeDiscard { .. } => stats.wasted += 1,
                _ => {}
            }
        }
        any.then_some(stats)
    }

    /// The fraction of authoritative calls served from the cache.
    pub fn hit_rate(&self) -> f64 {
        match self.hits + self.misses {
            0 => 0.0,
            calls => self.hits as f64 / calls as f64,
        }
    }
}

/// A measured distribution. Percentiles are nearest-rank over the sorted
/// samples with no interpolation, so every reported number is a number that
/// was actually measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub mean_ms: f64,
    pub median_ms: u64,
    pub p95_ms: u64,
}

impl Stats {
    pub fn of(samples: &[u64]) -> Self {
        if samples.is_empty() {
            return Self {
                count: 0,
                mean_ms: 0.0,
                median_ms: 0,
                p95_ms: 0,
            };
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let total: u64 = sorted.iter().sum();
        Self {
            count: sorted.len(),
            mean_ms: total as f64 / sorted.len() as f64,
            median_ms: percentile(&sorted, 1, 2),
            p95_ms: percentile(&sorted, 95, 100),
        }
    }
}

/// Rank arithmetic in integers, so the same samples always summarise to the
/// same numbers.
fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = (sorted.len() * numerator).div_ceil(denominator).max(1);
    sorted[rank - 1]
}

/// One tool's calls across every trial.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolStats {
    pub tool: String,
    pub calls: usize,
    pub latency: Stats,
}

/// Repeated trials of one task.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskReport {
    pub task: String,
    pub trials: usize,
    pub successes: usize,
    /// Per turn, across every trial.
    pub turn_ms: Stats,
    /// The half of each turn's wall time that was not spent in a tool.
    pub model_ms: Stats,
    /// The half that was.
    pub tool_ms: Stats,
    /// By tool name, so the report reads the same however the calls were
    /// ordered.
    pub tools: Vec<ToolStats>,
    /// Tokens billed across every turn of every trial: what the trials cost.
    pub tokens: Usage,
    /// Why turns failed, when any did. Empty for a clean baseline.
    pub errors: Vec<String>,
    /// Why prediction rounds came back short, when any did. A predictor never
    /// fails a turn, so without these a misconfigured predictor reads as one
    /// that merely guesses badly.
    pub predictor_errors: Vec<String>,
    /// Present when the trials speculated.
    pub speculation: Option<SpeculationStats>,
}

impl TaskReport {
    /// Aggregates the traces of repeated trials of one task.
    pub fn of(traces: &[RunTrace]) -> Self {
        let mut turn_ms = Vec::new();
        let mut model_ms = Vec::new();
        let mut tool_ms = Vec::new();
        let mut by_tool: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut tokens = Usage::default();
        let mut errors = Vec::new();
        let mut predictor_errors = Vec::new();

        for trace in traces {
            for turn in &trace.turns {
                turn_ms.push(turn.wall_ms());
                model_ms.push(turn.model_ms());
                tool_ms.push(turn.tool_ms());
                if let Some(usage) = turn.usage {
                    tokens.add(usage);
                }
                if let Some(error) = &turn.error {
                    errors.push(error.clone());
                }
                for record in &turn.speculation {
                    if let TraceRecord::Prediction {
                        error: Some(error), ..
                    } = record
                    {
                        predictor_errors.push(error.clone());
                    }
                }
            }
            for call in trace.tool_calls() {
                by_tool.entry(call.tool).or_default().push(call.duration_ms);
            }
        }

        Self {
            task: traces.first().map(|t| t.task.clone()).unwrap_or_default(),
            trials: traces.len(),
            successes: traces.iter().filter(|trace| trace.success).count(),
            turn_ms: Stats::of(&turn_ms),
            model_ms: Stats::of(&model_ms),
            tool_ms: Stats::of(&tool_ms),
            tools: by_tool
                .into_iter()
                .map(|(tool, durations)| ToolStats {
                    tool,
                    calls: durations.len(),
                    latency: Stats::of(&durations),
                })
                .collect(),
            tokens,
            errors,
            predictor_errors,
            speculation: SpeculationStats::of(traces),
        }
    }
}

impl fmt::Display for TaskReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {}/{} succeeded, {} turns, {} in + {} out tokens",
            self.task,
            self.successes,
            self.trials,
            self.turn_ms.count,
            self.tokens.input_tokens,
            self.tokens.output_tokens,
        )?;
        writeln!(
            f,
            "  {:<14} {:>5} {:>9} {:>9} {:>9}",
            "", "n", "median", "p95", "mean"
        )?;
        for (label, stats) in [
            ("turn wall", self.turn_ms),
            ("model time", self.model_ms),
            ("tool time", self.tool_ms),
        ] {
            row(f, label, stats)?;
        }
        for tool in &self.tools {
            row(f, &tool.tool, tool.latency)?;
        }
        if let Some(spec) = &self.speculation {
            writeln!(
                f,
                "  speculation: {} proposed, {} executed, {}/{} hits ({:.0}%), {} wasted",
                spec.proposed,
                spec.executed,
                spec.hits,
                spec.hits + spec.misses,
                spec.hit_rate() * 100.0,
                spec.wasted,
            )?;
            writeln!(
                f,
                "  predictor cost: {} in + {} out tokens",
                spec.predictor_tokens.input_tokens, spec.predictor_tokens.output_tokens,
            )?;
        }
        for error in &self.errors {
            writeln!(f, "  error: {error}")?;
        }
        for error in &self.predictor_errors {
            writeln!(f, "  predictor error: {error}")?;
        }
        Ok(())
    }
}

/// One strategy against the baseline it has to beat. Net latency change needs
/// both runs, so it lives here rather than on either report; everything else
/// the research question asks for — hit rate, predictor cost, wasted work —
/// is already on the speculated report.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub baseline: TaskReport,
    pub speculated: TaskReport,
}

impl Comparison {
    /// Median turn wall time saved, negative when speculation cost time.
    /// Median rather than mean because one long trial should not decide
    /// whether a strategy paid.
    pub fn latency_change_ms(&self) -> i64 {
        self.speculated.turn_ms.median_ms as i64 - self.baseline.turn_ms.median_ms as i64
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.baseline, self.speculated)?;
        writeln!(
            f,
            "  net latency change: median turn {}ms -> {}ms ({:+}ms)",
            self.baseline.turn_ms.median_ms,
            self.speculated.turn_ms.median_ms,
            self.latency_change_ms(),
        )
    }
}

fn row(f: &mut fmt::Formatter<'_>, label: &str, stats: Stats) -> fmt::Result {
    writeln!(
        f,
        "  {:<14} {:>5} {:>7}ms {:>7}ms {:>7.1}ms",
        label, stats.count, stats.median_ms, stats.p95_ms, stats.mean_ms
    )
}

//! Speculative tool execution (docs/design.md, "Speculation"). While the
//! authoritative model generates its response, a predictor guesses the tool
//! calls that response will make; the runtime executes the allowlisted guesses
//! concurrently and parks each result in a per-turn cache; when the real call
//! arrives, reconciliation is an exact cache-key lookup that adopts the
//! finished result or the in-flight future.
//!
//! Speculation adds no interface events and its executions never enter session
//! history. It is recorded through trace records stamped from the event
//! stream's clock and sequence space (`crate::trace`), and the engine consults
//! it only at its own loop points, so the cache needs no locking: speculative
//! work runs on spawned tasks, but every insertion, adoption, and discard
//! happens on the engine task.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::item::WorkspaceRevision;
use crate::model::{ToolCallRequest, TurnRequest, Usage};
use crate::tool::ToolOutput;

/// The v0 speculative allowlist: the read-only structured tools (docs/
/// design.md, "Five tools, one profile"). Writes remain authoritative.
pub fn v0_allowlist() -> Vec<String> {
    ["read_file", "list_directory", "search"]
        .map(String::from)
        .to_vec()
}

/// A prediction is just a guessed tool call, and a strategy is whatever hands
/// the runtime an ordered, deduplicated list of them for the model response
/// now being generated. The request is exactly the authoritative request —
/// agent and predictor share the tool profile, so their calls are comparable.
///
/// `budget` is the prediction budget: the predictor tokens this round may
/// spend guessing, the first of the research question's two independent
/// variables. A strategy caps its own requests by it, which is what lets two
/// strategies be compared under one number; the replay oracle spends none.
pub trait Predictor: Send + Sync {
    fn predict(&self, request: &TurnRequest, budget: u64) -> PredictionFuture;
}

pub type PredictionFuture = Pin<Box<dyn Future<Output = Prediction> + Send>>;

/// One round of guessing.
#[derive(Debug, Clone, Default)]
pub struct Prediction {
    /// Ordered and deduplicated; the runtime executes the first few that pass
    /// selection.
    pub calls: Vec<ToolCallRequest>,
    /// What the predictor's own requests billed, which the report sums into
    /// predictor cost (`crate::bench::SpeculationStats`).
    pub usage: Usage,
    /// Why a round came back short, when it did. A predictor that fails, times
    /// out, or answers with nothing usable yields an empty prediction rather
    /// than a failed turn, so without this a misconfigured predictor is
    /// indistinguishable from one that simply guesses badly.
    pub error: Option<String>,
}

/// Speculation switched on: who guesses, what may run, and how wide.
pub struct SpeculationConfig {
    pub predictor: Arc<dyn Predictor>,
    /// Tools speculation may execute. v0 is the read-only structured tools;
    /// writes remain authoritative.
    pub allowlist: Vec<String>,
    /// Predictor tokens one round may spend guessing — the first of the
    /// research question's two independent variables, handed to the predictor
    /// on every round.
    pub prediction_budget: u64,
    /// Concurrent speculative executions per round — the fanout, the second
    /// variable. The runtime executes the first `execution_budget` guesses that
    /// pass selection.
    pub execution_budget: usize,
}

/// The exact identity of a call against a workspace state: same tool, same
/// canonical arguments, same working directory, same revision. Anything less
/// exact could adopt a stale or wrong result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    tool: String,
    arguments: String,
    cwd: PathBuf,
    revision: WorkspaceRevision,
}

impl CacheKey {
    pub fn new(
        tool: &str,
        arguments: &serde_json::Value,
        cwd: &Path,
        revision: WorkspaceRevision,
    ) -> Self {
        Self {
            tool: tool.to_string(),
            arguments: canonical_json(arguments),
            cwd: cwd.to_path_buf(),
            revision,
        }
    }
}

/// One serialization per value: objects render with their keys sorted at every
/// depth, so two calls whose arguments differ only in key order are the same
/// call.
pub fn canonical_json(value: &serde_json::Value) -> String {
    fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by_key(|(key, _)| key.as_str());
                serde_json::Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key.clone(), canonicalize(value)))
                        .collect(),
                )
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(canonicalize).collect())
            }
            other => other.clone(),
        }
    }
    canonicalize(value).to_string()
}

/// One speculative execution, in flight or finished.
pub struct CacheEntry {
    pub handle: JoinHandle<ToolOutput>,
    /// Kept alongside the key's canonical form so discard records can carry
    /// the arguments as the model would have sent them.
    pub tool: String,
    pub arguments: serde_json::Value,
}

/// A cache entry that never served an authoritative call.
pub struct Discarded {
    pub tool: String,
    pub arguments: serde_json::Value,
    /// Whether the execution had finished when it was discarded; an unfinished
    /// one was aborted mid-flight.
    pub finished: bool,
}

/// The per-turn cache plus the configuration that fills it. Owned by the
/// engine; entries hold spawned executions.
pub struct SpeculationRuntime {
    config: SpeculationConfig,
    cache: HashMap<CacheKey, CacheEntry>,
}

impl SpeculationRuntime {
    pub fn new(config: SpeculationConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    pub fn predictor(&self) -> &Arc<dyn Predictor> {
        &self.config.predictor
    }

    pub fn prediction_budget(&self) -> u64 {
        self.config.prediction_budget
    }

    pub fn execution_budget(&self) -> usize {
        self.config.execution_budget
    }

    pub fn allowlisted(&self, tool: &str) -> bool {
        self.config.allowlist.iter().any(|name| name == tool)
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.cache.contains_key(key)
    }

    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) {
        self.cache.insert(key, entry);
    }

    /// Adopts the entry for one authoritative call, removing it: an entry
    /// serves at most once.
    pub fn take(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        self.cache.remove(key)
    }

    /// Entries still executing. They count against the next round's fanout:
    /// the execution budget caps concurrency, not starts per round.
    pub fn in_flight(&self) -> usize {
        self.cache
            .values()
            .filter(|entry| !entry.handle.is_finished())
            .count()
    }

    /// Drops every entry made against an older revision. Called after a
    /// mutation bumps the revision: a stale result must never be adopted, so
    /// in-flight executions are aborted rather than left to finish.
    pub fn invalidate(&mut self, current: WorkspaceRevision) -> Vec<Discarded> {
        let stale: Vec<CacheKey> = self
            .cache
            .keys()
            .filter(|key| key.revision != current)
            .cloned()
            .collect();
        stale
            .into_iter()
            .map(|key| discard(self.cache.remove(&key).expect("key just listed")))
            .collect()
    }

    /// Drops everything at turn end: the cache never outlives a turn.
    pub fn drain(&mut self) -> Vec<Discarded> {
        self.cache
            .drain()
            .map(|(_, entry)| discard(entry))
            .collect()
    }
}

fn discard(entry: CacheEntry) -> Discarded {
    let finished = entry.handle.is_finished();
    entry.handle.abort();
    Discarded {
        tool: entry.tool,
        arguments: entry.arguments,
        finished,
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_json;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_at_every_depth() {
        let a = json!({ "b": { "d": 1, "c": [ { "f": 2, "e": 3 } ] }, "a": 4 });
        let b = json!({ "a": 4, "b": { "c": [ { "e": 3, "f": 2 } ], "d": 1 } });
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(
            canonical_json(&a),
            r#"{"a":4,"b":{"c":[{"e":3,"f":2}],"d":1}}"#
        );
    }

    #[test]
    fn canonical_json_preserves_array_order() {
        assert_ne!(
            canonical_json(&json!({ "paths": ["a", "b"] })),
            canonical_json(&json!({ "paths": ["b", "a"] })),
        );
    }
}

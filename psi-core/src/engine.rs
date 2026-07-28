//! The turn engine: executes commands against sessions and emits the event
//! stream. Runs as one task; a turn runs inline in the command loop, which
//! keeps selecting on the command channel so `cancel_turn` lands mid-turn
//! while every other command waits for the turn to end.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;

use crate::hook::{HookDecision, HookRegistry};
use crate::item::{CompletionStatus, ItemId, ItemKind, ItemPayload, TurnId, WorkspaceRevision};
use crate::model::{ModelBackend, ModelEvent, Sampling, ToolCallRequest, TurnRequest, Usage};
use crate::protocol::{Command, Event, EventPayload};
use crate::session::{Session, SessionId};
use crate::speculation::{
    CacheEntry, CacheKey, Prediction, PredictionFuture, SpeculationConfig, SpeculationRuntime,
};
use crate::store::SessionStore;
use crate::tool::{ToolEffect, ToolFuture, ToolInvocation, ToolOutput, ToolRegistry};
use crate::trace::{DiscardReason, PredictedCall, TraceRecord, TraceWriter};

/// Everything the engine needs to run. A struct rather than arguments because
/// `workspace` and `sessions_dir` are two paths that must not be swapped.
pub struct HarnessConfig {
    pub model: Arc<dyn ModelBackend>,
    pub tools: ToolRegistry,
    /// Registered here and nowhere else.
    pub hooks: HookRegistry,
    pub workspace: PathBuf,
    /// Where session logs live; created if missing.
    pub sessions_dir: PathBuf,
    /// Where this run's trace is written. `Some` only for a measured run: the
    /// Milestone 5 baselines are the consumer, and Milestone 6's speculation
    /// records are stamped from the same sequence counter into the same file.
    /// Interactive Psi passes `None`.
    pub trace: Option<TraceWriter>,
    /// Speculative tool execution. `None` runs the baseline agent loop
    /// untouched: speculation is optional middleware (docs/design.md).
    pub speculation: Option<SpeculationConfig>,
}

pub struct Harness;

impl Harness {
    /// Spawns the engine task and returns the command/event channel pair —
    /// the in-process form of the interface protocol. Fails when the sessions
    /// directory cannot be opened, before any client can be waiting on an
    /// event that would never come.
    pub fn spawn(
        config: HarnessConfig,
    ) -> std::io::Result<(mpsc::Sender<Command>, mpsc::Receiver<Event>)> {
        let store = SessionStore::new(config.sessions_dir)?;
        let (command_tx, command_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(256);
        let engine = Engine {
            model: config.model,
            tools: config.tools,
            hooks: config.hooks,
            workspace: config.workspace,
            store,
            revision: WorkspaceRevision(0),
            sessions: HashMap::new(),
            commands: command_rx,
            deferred: VecDeque::new(),
            events: EventSink {
                tx: event_tx,
                next_seq: 0,
                trace: config.trace,
            },
            speculation: config.speculation.map(SpeculationRuntime::new),
        };
        tokio::spawn(engine.run());
        Ok((command_tx, event_rx))
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis() as u64
}

struct EventSink {
    tx: mpsc::Sender<Event>,
    next_seq: u64,
    trace: Option<TraceWriter>,
}

impl EventSink {
    async fn emit(&mut self, session_id: Option<SessionId>, payload: EventPayload) {
        let seq = self.next_seq;
        let timestamp_ms = now_ms();
        self.next_seq += 1;
        // The trace record is written before the event goes out, so a client
        // that has seen `turn_finished` can read a trace that already holds it.
        if let Some(trace) = &self.trace {
            trace.record_event(seq, timestamp_ms, &payload);
        }
        let event = Event {
            seq,
            timestamp_ms,
            session_id,
            payload,
        };
        // A dropped receiver means no client is listening; the engine keeps going.
        let _ = self.tx.send(event).await;
    }

    /// Stamps a speculation record from the same clock and sequence space as
    /// the events without emitting any event: speculation adds no interface
    /// events (docs/design.md, "Speculation"). Without a trace nothing is
    /// recorded and no sequence number is consumed, so an untraced run's event
    /// stream is identical with speculation on or off.
    fn record_speculation(&mut self, make: impl FnOnce(u64, u64) -> TraceRecord) {
        let Some(trace) = &self.trace else { return };
        let seq = self.next_seq;
        self.next_seq += 1;
        let _ = trace.write(&make(seq, now_ms()));
    }
}

/// What ended one model response round.
enum RoundEnd {
    Completed,
    Cancelled,
    Failed(String),
}

/// A streamed item under assembly: `item_started` and deltas already emitted
/// under a reserved id, appended as a complete item when it closes.
struct OpenItem {
    id: ItemId,
    kind: StreamedKind,
    buffer: String,
    /// Set from `ReasoningCompleted` just before the item closes.
    provider_data: Option<serde_json::Value>,
}

#[derive(Clone, Copy, PartialEq)]
enum StreamedKind {
    AssistantMessage,
    Reasoning,
}

/// A tool call whose arguments are still streaming: `item_started` and
/// argument deltas already emitted under a reserved id, appended when the call
/// completes.
struct OpenCall {
    id: ItemId,
    call_id: String,
    tool: String,
}

/// How a turn ended, as reported on `turn_finished`.
struct TurnOutcome {
    status: CompletionStatus,
    error: Option<String>,
    usage: Option<Usage>,
}

struct Engine {
    model: Arc<dyn ModelBackend>,
    tools: ToolRegistry,
    hooks: HookRegistry,
    workspace: PathBuf,
    store: SessionStore,
    /// One harness serves one workspace, so one revision counter is shared by
    /// every session.
    revision: WorkspaceRevision,
    /// Sessions loaded into memory. Every one of them is also on disk; this is
    /// the working set, not the record.
    sessions: HashMap<SessionId, Session>,
    commands: mpsc::Receiver<Command>,
    /// Commands that arrived mid-turn; replayed in order once the turn ends.
    deferred: VecDeque<Command>,
    events: EventSink,
    /// The speculation cache and its configuration; `None` is the baseline
    /// loop. Only the engine task touches it — speculative executions run on
    /// spawned tasks, but their handles live here.
    speculation: Option<SpeculationRuntime>,
}

impl Engine {
    async fn run(mut self) {
        loop {
            let command = match self.deferred.pop_front() {
                Some(command) => command,
                None => match self.commands.recv().await {
                    Some(command) => command,
                    None => return,
                },
            };
            if !self.handle(command).await {
                return;
            }
        }
    }

    /// Returns false when the engine must stop.
    async fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::CreateSession => {
                // A store that cannot start a session cannot serve any, and no
                // event would ever answer this command. Stopping closes the
                // event stream, which clients already read as the harness
                // going away.
                let Ok((meta, log)) = self.store.create(now_ms()) else {
                    return false;
                };
                let id = meta.id.clone();
                self.sessions
                    .insert(id.clone(), Session::new(meta.clone(), log));
                self.emit(&id, EventPayload::SessionCreated { meta }).await;
            }
            Command::LoadSession { session_id } => {
                // An id that names nothing on disk is a client mistake, not a
                // broken store: the command is dropped.
                if !self.sessions.contains_key(&session_id) {
                    let Ok((snapshot, log)) = self.store.load(&session_id) else {
                        return true;
                    };
                    self.sessions
                        .insert(session_id.clone(), Session::restore(snapshot, log));
                }
                let snapshot = self.sessions[&session_id].snapshot();
                self.emit(&session_id, EventPayload::SessionLoaded { snapshot })
                    .await;
            }
            Command::ListSessions => {
                let sessions = self.store.list();
                self.events
                    .emit(None, EventPayload::SessionsListed { sessions })
                    .await;
            }
            Command::SetHead {
                session_id,
                item_id,
            } => {
                if let Some(session) = self.sessions.get_mut(&session_id) {
                    let _ = session.set_head(item_id);
                }
            }
            Command::CancelTurn { .. } => {} // No turn is running.
            Command::SubmitMessage { session_id, text } => {
                self.run_turn(session_id, text).await;
            }
        }
        true
    }

    async fn run_turn(&mut self, session_id: SessionId, text: String) {
        // The session leaves the map for the duration of the turn so the
        // engine can borrow it mutably alongside its own channels.
        let Some(mut session) = self.sessions.remove(&session_id) else {
            return;
        };
        let turn_id = session.begin_turn();
        self.emit(&session_id, EventPayload::TurnStarted { turn_id })
            .await;

        let user_id = session.reserve_item_id();
        self.emit(
            &session_id,
            EventPayload::ItemStarted {
                item_id: user_id,
                kind: ItemKind::UserMessage,
            },
        )
        .await;
        let item = session
            .append(
                user_id,
                turn_id,
                ItemPayload::UserMessage { text },
                CompletionStatus::Completed,
                None,
                now_ms(),
            )
            .clone();
        self.emit(&session_id, EventPayload::ItemFinished { item })
            .await;

        let mut outcome = self.turn_loop(&session_id, &mut session, turn_id).await;
        // The cache never outlives a turn: whatever is still parked is wasted
        // work, recorded inside the turn before turn_finished closes it.
        let unused = self
            .speculation
            .as_mut()
            .map(|spec| spec.drain())
            .unwrap_or_default();
        self.record_discards(turn_id, unused, DiscardReason::Unused);
        // A turn whose items did not reach disk did not really complete. A
        // turn that already failed keeps its own error and leaves this one
        // pending, so the report is never lost, only delayed.
        if outcome.status == CompletionStatus::Completed
            && let Some(error) = session.take_log_error()
        {
            outcome.status = CompletionStatus::Failed;
            outcome.error = Some(error);
        }
        self.emit(
            &session_id,
            EventPayload::TurnFinished {
                turn_id,
                status: outcome.status,
                error: outcome.error,
                usage: outcome.usage,
            },
        )
        .await;
        self.sessions.insert(session_id, session);
    }

    /// Alternates model responses and tool execution until a response carries
    /// no tool calls, the turn is cancelled, or something fails.
    async fn turn_loop(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
    ) -> TurnOutcome {
        let mut usage: Option<Usage> = None;
        loop {
            let request = TurnRequest {
                session_id: session_id.clone(),
                items: session.path_to_head().into_iter().cloned().collect(),
                tools: self.tools.specs(),
                // The authoritative turn takes the target's own sampling
                // defaults; only the predictor overrides them.
                sampling: Sampling::default(),
            };
            // The predictor starts guessing as the authoritative request goes
            // out: the model's generation time is the window speculation uses.
            // Dropping the future at round end cancels an unfinished predictor.
            let mut prediction: Option<PredictionFuture> = self
                .speculation
                .as_ref()
                .map(|spec| spec.predictor().predict(&request, spec.prediction_budget()));
            let mut stream = self.model.stream_response(request);

            let mut open: Option<OpenItem> = None;
            let mut open_call: Option<OpenCall> = None;
            let mut calls: Vec<ToolCallRequest> = Vec::new();

            let end = loop {
                enum Wake {
                    Model(Option<ModelEvent>),
                    Command(Option<Command>),
                    Predicted(Prediction),
                }
                let wake = tokio::select! {
                    event = stream.recv() => Wake::Model(event),
                    command = self.commands.recv() => Wake::Command(command),
                    guesses = async {
                        prediction.as_mut().expect("branch enabled only when set").await
                    }, if prediction.is_some() => Wake::Predicted(guesses),
                };
                match wake {
                    Wake::Predicted(guesses) => {
                        prediction = None;
                        self.speculate(turn_id, guesses);
                    }
                    Wake::Model(Some(ModelEvent::TextDelta { delta })) => {
                        self.stream_delta(
                            session_id,
                            session,
                            turn_id,
                            &mut open,
                            StreamedKind::AssistantMessage,
                            delta,
                        )
                        .await;
                    }
                    Wake::Model(Some(ModelEvent::ReasoningDelta { delta })) => {
                        self.stream_delta(
                            session_id,
                            session,
                            turn_id,
                            &mut open,
                            StreamedKind::Reasoning,
                            delta,
                        )
                        .await;
                    }
                    Wake::Model(Some(ModelEvent::ReasoningCompleted { provider_data })) => {
                        // Reasoning that streams no text still has to become an
                        // item: the provider data is what makes the turn
                        // replayable.
                        if !matches!(&open, Some(item) if item.kind == StreamedKind::Reasoning) {
                            self.close_open(
                                session_id,
                                session,
                                turn_id,
                                open.take(),
                                CompletionStatus::Completed,
                                None,
                            )
                            .await;
                            let id = session.reserve_item_id();
                            self.emit(
                                session_id,
                                EventPayload::ItemStarted {
                                    item_id: id,
                                    kind: ItemKind::Reasoning,
                                },
                            )
                            .await;
                            open = Some(OpenItem {
                                id,
                                kind: StreamedKind::Reasoning,
                                buffer: String::new(),
                                provider_data: None,
                            });
                        }
                        if let Some(item) = open.as_mut() {
                            item.provider_data = Some(provider_data);
                        }
                        self.close_open(
                            session_id,
                            session,
                            turn_id,
                            open.take(),
                            CompletionStatus::Completed,
                            None,
                        )
                        .await;
                    }
                    Wake::Model(Some(ModelEvent::ToolCallArgumentsDelta {
                        call_id,
                        tool,
                        delta,
                    })) => {
                        let item_id = match &open_call {
                            Some(open) if open.call_id == call_id => open.id,
                            _ => {
                                self.close_open(
                                    session_id,
                                    session,
                                    turn_id,
                                    open.take(),
                                    CompletionStatus::Completed,
                                    None,
                                )
                                .await;
                                self.close_open_call(
                                    session_id,
                                    session,
                                    turn_id,
                                    open_call.take(),
                                    CompletionStatus::Failed,
                                    Some("the model started another call first".to_string()),
                                )
                                .await;
                                let id = session.reserve_item_id();
                                self.emit(
                                    session_id,
                                    EventPayload::ItemStarted {
                                        item_id: id,
                                        kind: ItemKind::ToolCall,
                                    },
                                )
                                .await;
                                open_call = Some(OpenCall { id, call_id, tool });
                                id
                            }
                        };
                        self.emit(session_id, EventPayload::ItemDelta { item_id, delta })
                            .await;
                    }
                    Wake::Model(Some(ModelEvent::Usage { usage: reported })) => {
                        usage.get_or_insert_default().add(reported);
                    }
                    Wake::Model(Some(ModelEvent::ToolCallCompleted { call })) => {
                        self.close_open(
                            session_id,
                            session,
                            turn_id,
                            open.take(),
                            CompletionStatus::Completed,
                            None,
                        )
                        .await;
                        // The item already exists when the arguments streamed.
                        let id = match open_call.take() {
                            Some(open) if open.call_id == call.call_id => open.id,
                            stale => {
                                self.close_open_call(
                                    session_id,
                                    session,
                                    turn_id,
                                    stale,
                                    CompletionStatus::Failed,
                                    Some("the model completed another call first".to_string()),
                                )
                                .await;
                                let id = session.reserve_item_id();
                                self.emit(
                                    session_id,
                                    EventPayload::ItemStarted {
                                        item_id: id,
                                        kind: ItemKind::ToolCall,
                                    },
                                )
                                .await;
                                id
                            }
                        };
                        let payload = ItemPayload::ToolCall {
                            tool: call.tool.clone(),
                            call_id: call.call_id.clone(),
                            arguments: call.arguments.clone(),
                            cwd: self.workspace.clone(),
                            revision: self.revision,
                        };
                        let item = session
                            .append(
                                id,
                                turn_id,
                                payload,
                                CompletionStatus::Completed,
                                None,
                                now_ms(),
                            )
                            .clone();
                        self.emit(session_id, EventPayload::ItemFinished { item })
                            .await;
                        calls.push(call);
                    }
                    Wake::Model(Some(ModelEvent::Completed)) => break RoundEnd::Completed,
                    Wake::Model(Some(ModelEvent::Error { message })) => {
                        break RoundEnd::Failed(message);
                    }
                    Wake::Model(None) => {
                        // The backend hung up without Completed or Error;
                        // silence is never success.
                        break RoundEnd::Failed(
                            "model stream ended without completing".to_string(),
                        );
                    }
                    Wake::Command(Some(Command::CancelTurn { session_id: target }))
                        if target == *session_id =>
                    {
                        break RoundEnd::Cancelled;
                    }
                    Wake::Command(Some(other)) => self.deferred.push_back(other),
                    // Every client hung up; treat as cancellation.
                    Wake::Command(None) => break RoundEnd::Cancelled,
                }
            };

            let (status, error) = match &end {
                RoundEnd::Completed => (CompletionStatus::Completed, None),
                RoundEnd::Cancelled => (CompletionStatus::Cancelled, None),
                RoundEnd::Failed(message) => (CompletionStatus::Failed, Some(message.clone())),
            };
            self.close_open(session_id, session, turn_id, open.take(), status, error)
                .await;
            // A response that ends mid-arguments leaves a call that can never
            // run: it is recorded so its `item_started` closes, and dropped
            // from the request the codec builds next.
            let (status, error) = match &end {
                RoundEnd::Cancelled => (CompletionStatus::Cancelled, None),
                RoundEnd::Failed(message) => (CompletionStatus::Failed, Some(message.clone())),
                RoundEnd::Completed => (
                    CompletionStatus::Failed,
                    Some("the response completed before the arguments did".to_string()),
                ),
            };
            self.close_open_call(
                session_id,
                session,
                turn_id,
                open_call.take(),
                status,
                error,
            )
            .await;

            match end {
                RoundEnd::Failed(message) => {
                    self.settle_unexecuted_calls(session_id, session, turn_id, &calls)
                        .await;
                    return TurnOutcome {
                        status: CompletionStatus::Failed,
                        error: Some(message),
                        usage,
                    };
                }
                RoundEnd::Cancelled => {
                    self.settle_unexecuted_calls(session_id, session, turn_id, &calls)
                        .await;
                    return TurnOutcome {
                        status: CompletionStatus::Cancelled,
                        error: None,
                        usage,
                    };
                }
                RoundEnd::Completed => {
                    if calls.is_empty() {
                        return TurnOutcome {
                            status: CompletionStatus::Completed,
                            error: None,
                            usage,
                        };
                    }
                    let mut remaining = calls.into_iter();
                    while let Some(call) = remaining.next() {
                        if !self.execute_call(session_id, session, turn_id, &call).await {
                            // Cancelled mid-execution: settle the rest and end.
                            let rest: Vec<_> = remaining.collect();
                            self.settle_unexecuted_calls(session_id, session, turn_id, &rest)
                                .await;
                            return TurnOutcome {
                                status: CompletionStatus::Cancelled,
                                error: None,
                                usage,
                            };
                        }
                    }
                    // All calls executed; ask the model for its next response.
                }
            }
        }
    }

    /// Executes the predictor's guesses, in their order, up to the execution
    /// budget: each one allowlisted, not already cached, and passed by the
    /// before-hooks — a call a hook would block is never executed
    /// speculatively. Results park in the cache; nothing here touches the
    /// session or emits events.
    fn speculate(&mut self, turn_id: TurnId, prediction: Prediction) {
        let Some(spec) = self.speculation.as_mut() else {
            return;
        };
        let budget = spec.execution_budget();
        let Prediction {
            calls: guesses,
            usage,
            error,
        } = prediction;
        self.events
            .record_speculation(|seq, timestamp_ms| TraceRecord::Prediction {
                seq,
                timestamp_ms,
                turn_id,
                calls: guesses
                    .iter()
                    .map(|call| PredictedCall {
                        tool: call.tool.clone(),
                        arguments: call.arguments.clone(),
                    })
                    .collect(),
                usage,
                error,
            });

        let mut started = self
            .speculation
            .as_ref()
            .map(|spec| spec.in_flight())
            .unwrap_or(0);
        for call in guesses {
            if started >= budget {
                break;
            }
            let spec = self.speculation.as_mut().expect("checked above");
            if !spec.allowlisted(&call.tool) {
                continue;
            }
            let key = CacheKey::new(&call.tool, &call.arguments, &self.workspace, self.revision);
            if spec.contains(&key) {
                continue;
            }
            let invocation = ToolInvocation {
                call_id: call.call_id.clone(),
                arguments: call.arguments.clone(),
                cwd: self.workspace.clone(),
            };
            if let HookDecision::Block { .. } = self.hooks.before(&call.tool, &invocation) {
                continue;
            }
            let Some(tool) = self.tools.get(&call.tool) else {
                continue;
            };
            let handle = tokio::spawn(tool.execute(invocation));
            let revision = self.revision;
            self.speculation.as_mut().expect("checked above").insert(
                key,
                CacheEntry {
                    handle,
                    tool: call.tool.clone(),
                    arguments: call.arguments.clone(),
                },
            );
            self.events
                .record_speculation(|seq, timestamp_ms| TraceRecord::SpeculativeExecution {
                    seq,
                    timestamp_ms,
                    turn_id,
                    tool: call.tool,
                    arguments: call.arguments,
                    revision,
                });
            started += 1;
        }
    }

    /// Bumps the workspace revision and drops every cache entry made against
    /// the old one, aborting executions still in flight: a stale result must
    /// never be adopted.
    fn bump_revision(&mut self, turn_id: TurnId) {
        self.revision = WorkspaceRevision(self.revision.0 + 1);
        let stale = self
            .speculation
            .as_mut()
            .map(|spec| spec.invalidate(self.revision))
            .unwrap_or_default();
        self.record_discards(turn_id, stale, DiscardReason::Invalidated);
    }

    fn record_discards(
        &mut self,
        turn_id: TurnId,
        discarded: Vec<crate::speculation::Discarded>,
        reason: DiscardReason,
    ) {
        for entry in discarded {
            self.events
                .record_speculation(|seq, timestamp_ms| TraceRecord::SpeculativeDiscard {
                    seq,
                    timestamp_ms,
                    turn_id,
                    tool: entry.tool,
                    arguments: entry.arguments,
                    finished: entry.finished,
                    reason,
                });
        }
    }

    /// Runs one authoritative tool call through to its tool_result item.
    /// Returns false if the turn was cancelled while the call ran.
    async fn execute_call(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
        call: &ToolCallRequest,
    ) -> bool {
        let result_id = session.reserve_item_id();
        self.emit(
            session_id,
            EventPayload::ItemStarted {
                item_id: result_id,
                kind: ItemKind::ToolResult,
            },
        )
        .await;
        let started = Instant::now();

        let invocation = ToolInvocation {
            call_id: call.call_id.clone(),
            arguments: call.arguments.clone(),
            cwd: self.workspace.clone(),
        };
        // A blocked call and an unknown tool both answer the model without
        // running anything, so neither can have mutated: they declare
        // themselves read-only and the revision stands. `ran` marks the case
        // where a tool really executed, which is what after-hooks observe.
        let refusal = match self.hooks.before(&call.tool, &invocation) {
            HookDecision::Block { reason } => Some(format!("{} refused: {reason}", call.tool)),
            HookDecision::Continue if self.tools.get(&call.tool).is_none() => {
                Some(format!("unknown tool: {}", call.tool))
            }
            HookDecision::Continue => None,
        };
        let ran = refusal.is_none();
        // Reconciliation: an exact cache-key lookup. A refused call is not
        // reconciled — it never runs, so it neither hits nor misses.
        let adopted = if ran && self.speculation.is_some() {
            let key = CacheKey::new(&call.tool, &call.arguments, &self.workspace, self.revision);
            let entry = self.speculation.as_mut().and_then(|spec| spec.take(&key));
            let (hit, finished) = match &entry {
                Some(entry) => (true, Some(entry.handle.is_finished())),
                None => (false, None),
            };
            self.events
                .record_speculation(|seq, timestamp_ms| TraceRecord::Reconciliation {
                    seq,
                    timestamp_ms,
                    turn_id,
                    tool: call.tool.clone(),
                    arguments: call.arguments.clone(),
                    hit,
                    finished,
                });
            entry
        } else {
            None
        };
        let (effect, mut future): (ToolEffect, ToolFuture) = match (refusal, adopted) {
            (Some(message), _) => {
                let output = ToolOutput {
                    content: message.clone(),
                    error: Some(message),
                    truncated: false,
                };
                (ToolEffect::ReadOnly, Box::pin(async move { output }))
            }
            (None, Some(entry)) => {
                // The adopted future finishes immediately or is awaited in
                // flight; either way the item's duration is the time this call
                // actually waited, which is the latency speculation saved.
                let tool = self.tools.get(&call.tool).expect("checked above");
                let handle = entry.handle;
                let future: ToolFuture = Box::pin(async move {
                    match handle.await {
                        Ok(output) => output,
                        // A panicked speculative task is a tool bug either
                        // way; surface it rather than quietly re-running.
                        Err(err) => ToolOutput {
                            content: format!("speculative execution failed: {err}"),
                            error: Some(format!("speculative execution failed: {err}")),
                            truncated: false,
                        },
                    }
                });
                (tool.effect(), future)
            }
            (None, None) => {
                let tool = self.tools.get(&call.tool).expect("checked above");
                (tool.effect(), tool.execute(invocation.clone()))
            }
        };

        let output = loop {
            enum Wake {
                Tool(ToolOutput),
                Command(Option<Command>),
            }
            let wake = tokio::select! {
                output = &mut future => Wake::Tool(output),
                command = self.commands.recv() => Wake::Command(command),
            };
            match wake {
                Wake::Tool(output) => break Some(output),
                Wake::Command(Some(Command::CancelTurn { session_id: target }))
                    if target == *session_id =>
                {
                    break None;
                }
                Wake::Command(Some(other)) => self.deferred.push_back(other),
                Wake::Command(None) => break None,
            }
        };
        let duration_ms = started.elapsed().as_millis() as u64;

        match output {
            Some(output) => {
                if ran {
                    self.hooks.after(&call.tool, &invocation, &output);
                }
                let status = if output.error.is_none() {
                    CompletionStatus::Completed
                } else {
                    CompletionStatus::Failed
                };
                let bump = match effect {
                    ToolEffect::ReadOnly => false,
                    ToolEffect::Mutating => status == CompletionStatus::Completed,
                    ToolEffect::Unknown => true,
                };
                if bump {
                    self.bump_revision(turn_id);
                }
                let payload = ItemPayload::ToolResult {
                    call_id: call.call_id.clone(),
                    content: output.content,
                    duration_ms,
                    truncated: output.truncated,
                };
                let item = session
                    .append(result_id, turn_id, payload, status, output.error, now_ms())
                    .clone();
                self.emit(session_id, EventPayload::ItemFinished { item })
                    .await;
                true
            }
            None => {
                // Cancelled while the tool ran. A killed call may already have
                // mutated, so anything not read-only still bumps the revision.
                if effect != ToolEffect::ReadOnly {
                    self.bump_revision(turn_id);
                }
                let payload = ItemPayload::ToolResult {
                    call_id: call.call_id.clone(),
                    content: String::new(),
                    duration_ms,
                    truncated: false,
                };
                let item = session
                    .append(
                        result_id,
                        turn_id,
                        payload,
                        CompletionStatus::Cancelled,
                        None,
                        now_ms(),
                    )
                    .clone();
                self.emit(session_id, EventPayload::ItemFinished { item })
                    .await;
                false
            }
        }
    }

    /// Records a cancelled tool_result for every collected call that never
    /// ran, so no tool_call is left dangling — providers reject replayed
    /// histories whose calls have no output.
    async fn settle_unexecuted_calls(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
        calls: &[ToolCallRequest],
    ) {
        for call in calls {
            let id = session.reserve_item_id();
            self.emit(
                session_id,
                EventPayload::ItemStarted {
                    item_id: id,
                    kind: ItemKind::ToolResult,
                },
            )
            .await;
            let payload = ItemPayload::ToolResult {
                call_id: call.call_id.clone(),
                content: String::new(),
                duration_ms: 0,
                truncated: false,
            };
            let item = session
                .append(
                    id,
                    turn_id,
                    payload,
                    CompletionStatus::Cancelled,
                    None,
                    now_ms(),
                )
                .clone();
            self.emit(session_id, EventPayload::ItemFinished { item })
                .await;
        }
    }

    /// Extends the open streamed item, or closes it and starts the next one
    /// when the delta kind switches.
    async fn stream_delta(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
        open: &mut Option<OpenItem>,
        kind: StreamedKind,
        delta: String,
    ) {
        match open {
            Some(item) if item.kind == kind => {
                item.buffer.push_str(&delta);
                let item_id = item.id;
                self.emit(session_id, EventPayload::ItemDelta { item_id, delta })
                    .await;
            }
            _ => {
                self.close_open(
                    session_id,
                    session,
                    turn_id,
                    open.take(),
                    CompletionStatus::Completed,
                    None,
                )
                .await;
                let id = session.reserve_item_id();
                let item_kind = match kind {
                    StreamedKind::AssistantMessage => ItemKind::AssistantMessage,
                    StreamedKind::Reasoning => ItemKind::Reasoning,
                };
                self.emit(
                    session_id,
                    EventPayload::ItemStarted {
                        item_id: id,
                        kind: item_kind,
                    },
                )
                .await;
                self.emit(
                    session_id,
                    EventPayload::ItemDelta {
                        item_id: id,
                        delta: delta.clone(),
                    },
                )
                .await;
                *open = Some(OpenItem {
                    id,
                    kind,
                    buffer: delta,
                    provider_data: None,
                });
            }
        }
    }

    /// Appends the open streamed item as a complete record. Partial content is
    /// kept: a cancelled assistant message persists what was streamed.
    async fn close_open(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
        open: Option<OpenItem>,
        status: CompletionStatus,
        error: Option<String>,
    ) {
        let Some(open) = open else { return };
        let payload = match open.kind {
            StreamedKind::AssistantMessage => ItemPayload::AssistantMessage { text: open.buffer },
            StreamedKind::Reasoning => ItemPayload::Reasoning {
                text: open.buffer,
                provider_data: open.provider_data,
            },
        };
        let item = session
            .append(open.id, turn_id, payload, status, error, now_ms())
            .clone();
        self.emit(session_id, EventPayload::ItemFinished { item })
            .await;
    }

    /// Appends a tool call whose arguments never finished streaming, so its
    /// `item_started` closes. The call never runs, and the codec drops calls
    /// with no result when it replays history.
    async fn close_open_call(
        &mut self,
        session_id: &SessionId,
        session: &mut Session,
        turn_id: TurnId,
        open_call: Option<OpenCall>,
        status: CompletionStatus,
        error: Option<String>,
    ) {
        let Some(open_call) = open_call else { return };
        let payload = ItemPayload::ToolCall {
            tool: open_call.tool,
            call_id: open_call.call_id,
            arguments: serde_json::Value::Null,
            cwd: self.workspace.clone(),
            revision: self.revision,
        };
        let item = session
            .append(open_call.id, turn_id, payload, status, error, now_ms())
            .clone();
        self.emit(session_id, EventPayload::ItemFinished { item })
            .await;
    }

    async fn emit(&mut self, session_id: &SessionId, payload: EventPayload) {
        self.events.emit(Some(session_id.clone()), payload).await;
    }
}

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
use crate::model::{ModelBackend, ModelEvent, ToolCallRequest, TurnRequest, Usage};
use crate::protocol::{Command, Event, EventPayload};
use crate::session::{Session, SessionId};
use crate::tool::{ToolEffect, ToolFuture, ToolInvocation, ToolOutput, ToolRegistry};

pub struct Harness;

impl Harness {
    /// Spawns the engine task and returns the command/event channel pair —
    /// the in-process form of the interface protocol. Hooks are registered
    /// here and nowhere else.
    pub fn spawn(
        model: Arc<dyn ModelBackend>,
        tools: ToolRegistry,
        hooks: HookRegistry,
        workspace: PathBuf,
    ) -> (mpsc::Sender<Command>, mpsc::Receiver<Event>) {
        let (command_tx, command_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(256);
        let engine = Engine {
            model,
            tools,
            hooks,
            workspace,
            revision: WorkspaceRevision(0),
            sessions: HashMap::new(),
            commands: command_rx,
            deferred: VecDeque::new(),
            events: EventSink {
                tx: event_tx,
                next_seq: 0,
            },
            created_sessions: 0,
        };
        tokio::spawn(engine.run());
        (command_tx, event_rx)
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
}

impl EventSink {
    async fn emit(&mut self, session_id: Option<SessionId>, payload: EventPayload) {
        let event = Event {
            seq: self.next_seq,
            timestamp_ms: now_ms(),
            session_id,
            payload,
        };
        self.next_seq += 1;
        // A dropped receiver means no client is listening; the engine keeps going.
        let _ = self.tx.send(event).await;
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
    /// One harness serves one workspace, so one revision counter is shared by
    /// every session.
    revision: WorkspaceRevision,
    sessions: HashMap<SessionId, Session>,
    commands: mpsc::Receiver<Command>,
    /// Commands that arrived mid-turn; replayed in order once the turn ends.
    deferred: VecDeque<Command>,
    events: EventSink,
    created_sessions: u64,
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
            self.handle(command).await;
        }
    }

    async fn handle(&mut self, command: Command) {
        match command {
            Command::CreateSession => {
                let id = SessionId(format!("s{}-{}", self.created_sessions, now_ms()));
                self.created_sessions += 1;
                let session = Session::new(id.clone(), now_ms());
                let meta = session.meta.clone();
                self.sessions.insert(id.clone(), session);
                self.emit(&id, EventPayload::SessionCreated { meta }).await;
            }
            Command::LoadSession { session_id } => {
                // Unknown ids are dropped; sessions live only in memory until
                // persistence lands (Milestone 3).
                if let Some(session) = self.sessions.get(&session_id) {
                    let snapshot = session.snapshot();
                    self.emit(&session_id, EventPayload::SessionLoaded { snapshot })
                        .await;
                }
            }
            Command::ListSessions => {
                let sessions = self.sessions.values().map(|s| s.meta.clone()).collect();
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

        let outcome = self.turn_loop(&session_id, &mut session, turn_id).await;
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
            };
            let mut stream = self.model.stream_response(request);

            let mut open: Option<OpenItem> = None;
            let mut open_call: Option<OpenCall> = None;
            let mut calls: Vec<ToolCallRequest> = Vec::new();

            let end = loop {
                enum Wake {
                    Model(Option<ModelEvent>),
                    Command(Option<Command>),
                }
                let wake = tokio::select! {
                    event = stream.recv() => Wake::Model(event),
                    command = self.commands.recv() => Wake::Command(command),
                };
                match wake {
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
        let (effect, mut future): (ToolEffect, ToolFuture) = match refusal {
            Some(message) => {
                let output = ToolOutput {
                    content: message.clone(),
                    error: Some(message),
                    truncated: false,
                };
                (ToolEffect::ReadOnly, Box::pin(async move { output }))
            }
            None => {
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
                    self.revision = WorkspaceRevision(self.revision.0 + 1);
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
                    self.revision = WorkspaceRevision(self.revision.0 + 1);
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

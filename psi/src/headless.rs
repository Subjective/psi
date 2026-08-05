//! Headless Psi: one prompt, one turn, printed as it streams. It is the same
//! client of the same protocol the TUI is, with no terminal state to restore
//! and no keys to read.

use std::process::ExitCode;

use psi_core::item::{ItemKind, ItemPayload};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::SessionId;
use tokio::sync::mpsc;

pub async fn run(
    commands: mpsc::Sender<Command>,
    mut events: mpsc::Receiver<Event>,
    resume: bool,
    prompt: String,
) -> ExitCode {
    let mut session_id = None;
    if resume {
        session_id = continue_session(&commands, &mut events).await;
    }
    let session_id = match session_id {
        Some(session_id) => session_id,
        // Nothing to continue is not an error: start fresh.
        None => match create_session(&commands, &mut events).await {
            Some(session_id) => session_id,
            None => return ExitCode::FAILURE,
        },
    };
    commands
        .send(Command::SubmitMessage {
            session_id,
            text: prompt,
        })
        .await
        .expect("engine");

    render(&mut events).await
}

async fn create_session(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
) -> Option<SessionId> {
    commands.send(Command::CreateSession).await.expect("engine");
    match events.recv().await.map(|event| event.payload) {
        Some(EventPayload::SessionCreated { meta }) => Some(meta.id),
        other => {
            eprintln!("psi: could not start a session: {other:?}");
            None
        }
    }
}

/// Loads the most recent session on disk. `None` when there is none to
/// continue, or when its log no longer loads; either way a new session is the
/// answer.
async fn continue_session(
    commands: &mpsc::Sender<Command>,
    events: &mut mpsc::Receiver<Event>,
) -> Option<SessionId> {
    commands.send(Command::ListSessions).await.expect("engine");
    let newest = match events.recv().await.map(|event| event.payload) {
        // `sessions_listed` is newest first.
        Some(EventPayload::SessionsListed { sessions }) => sessions.into_iter().next()?,
        other => {
            eprintln!("psi: unexpected event: {other:?}");
            return None;
        }
    };
    commands
        .send(Command::LoadSession {
            session_id: newest.id.clone(),
        })
        .await
        .expect("engine");
    match events.recv().await.map(|event| event.payload) {
        Some(EventPayload::SessionLoaded { snapshot }) => {
            eprintln!(
                "psi: continuing {} ({} items)",
                snapshot.meta.id.0,
                snapshot.items.len()
            );
            Some(snapshot.meta.id)
        }
        other => {
            eprintln!("psi: could not load {}: {other:?}", newest.id.0);
            None
        }
    }
}

async fn render(events: &mut mpsc::Receiver<Event>) -> ExitCode {
    // Only one item streams at a time, so the last one started is the one the
    // deltas belong to. Tool-call arguments stream too; they are printed whole
    // when the call closes instead.
    let mut streaming = None;
    while let Some(event) = events.recv().await {
        match event.payload {
            EventPayload::ItemStarted { kind, .. } => {
                streaming = Some(kind);
                println!();
            }
            EventPayload::ItemDelta { delta, .. } => {
                if matches!(
                    streaming,
                    Some(ItemKind::AssistantMessage) | Some(ItemKind::Reasoning)
                ) {
                    // Deltas rarely end in a newline; flush or nothing streams.
                    print!("{delta}");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            }
            EventPayload::ItemFinished { item } => match item.payload {
                ItemPayload::ToolCall {
                    tool, arguments, ..
                } => println!("{tool} {arguments}"),
                ItemPayload::ToolResult { content, .. } => println!("{content}"),
                _ => println!(),
            },
            EventPayload::TurnFinished {
                status,
                error,
                usage,
                ..
            } => {
                println!("\n[turn {status}, usage {usage:?}]");
                return match error {
                    Some(error) => {
                        eprintln!("psi: {error}");
                        ExitCode::FAILURE
                    }
                    None => ExitCode::SUCCESS,
                };
            }
            _ => {}
        }
    }
    ExitCode::FAILURE
}

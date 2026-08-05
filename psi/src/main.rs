//! Headless Psi: one prompt, one turn against the workspace in the current
//! directory, printed as it streams. `--continue` runs it against the most
//! recent session instead of a new one, which is what makes persistence
//! visible from the command line. The TUI arrives in Milestone 4.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use psi_core::hook::HookRegistry;
use psi_core::item::{ItemKind, ItemPayload};
use psi_core::openai::{OpenAiBackend, OpenAiConfig};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::SessionId;
use psi_core::tools::default_profile;
use psi_core::{Harness, HarnessConfig};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (resume, prompt) = match args.split_first() {
        Some((flag, rest)) if flag == "--continue" => (true, rest.join(" ")),
        _ => (false, args.join(" ")),
    };
    let sessions_dir = match sessions_dir() {
        Some(dir) => dir,
        None => {
            eprintln!("psi: set HOME or PSI_SESSIONS_DIR");
            return ExitCode::FAILURE;
        }
    };
    if prompt.is_empty() {
        eprintln!("usage: psi [--continue] <prompt>");
        eprintln!("environment: OPENAI_API_KEY (required), PSI_MODEL, PSI_BASE_URL");
        eprintln!("sessions: {}", sessions_dir.display());
        return ExitCode::FAILURE;
    }

    let mut config = OpenAiConfig::default();
    if let Ok(model) = std::env::var("PSI_MODEL") {
        config.model = model;
    }
    if let Ok(base_url) = std::env::var("PSI_BASE_URL") {
        config.base_url = base_url;
    }
    let workspace = match std::env::current_dir() {
        Ok(workspace) => workspace,
        Err(err) => {
            eprintln!("psi: {err}");
            return ExitCode::FAILURE;
        }
    };
    config.instructions = format!(
        "{}\n\nThe workspace root is {}.",
        config.instructions,
        workspace.display()
    );

    let model = match OpenAiBackend::new(config) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("psi: {err}");
            return ExitCode::FAILURE;
        }
    };

    let (commands, mut events) = match Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir,
    }) {
        Ok(harness) => harness,
        Err(err) => {
            eprintln!("psi: sessions directory: {err}");
            return ExitCode::FAILURE;
        }
    };

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

/// Session logs live in one directory rather than one per workspace, so
/// `--continue` and Milestone 4's branch views have a single place to look.
/// `PSI_SESSIONS_DIR` overrides it, which is how tests and benchmark runs keep
/// their sessions out of the user's.
fn sessions_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PSI_SESSIONS_DIR") {
        return Some(PathBuf::from(dir));
    }
    Some(PathBuf::from(std::env::var("HOME").ok()?).join(".psi/sessions"))
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

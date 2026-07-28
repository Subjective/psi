//! Headless Psi: one prompt, one turn against the workspace in the current
//! directory, printed as it streams. The TUI arrives in Milestone 4.

use std::process::ExitCode;
use std::sync::Arc;

use psi_core::Harness;
use psi_core::hook::HookRegistry;
use psi_core::item::{ItemKind, ItemPayload};
use psi_core::openai::{OpenAiBackend, OpenAiConfig};
use psi_core::protocol::{Command, EventPayload};
use psi_core::tools::default_profile;

#[tokio::main]
async fn main() -> ExitCode {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.is_empty() {
        eprintln!("usage: psi <prompt>");
        eprintln!("environment: OPENAI_API_KEY (required), PSI_MODEL, PSI_BASE_URL");
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

    let (commands, mut events) = Harness::spawn(
        Arc::new(model),
        default_profile(workspace.clone()),
        HookRegistry::new(),
        workspace,
    );

    commands.send(Command::CreateSession).await.expect("engine");
    let session_id = match events.recv().await.map(|event| event.payload) {
        Some(EventPayload::SessionCreated { meta }) => meta.id,
        other => {
            eprintln!("psi: unexpected first event: {other:?}");
            return ExitCode::FAILURE;
        }
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

async fn render(events: &mut tokio::sync::mpsc::Receiver<psi_core::protocol::Event>) -> ExitCode {
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
                    print!("{delta}");
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

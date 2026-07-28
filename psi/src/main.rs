//! The Psi binary. With a prompt it runs one turn headless and exits; with no
//! prompt it opens the TUI. `--continue` resumes the most recent session
//! either way. Both are clients of the same harness over the same protocol.

mod headless;
mod tui;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use psi_core::hook::HookRegistry;
use psi_core::openai::{OpenAiBackend, OpenAiConfig};
use psi_core::protocol::{Command, Event};
use psi_core::tools::default_profile;
use psi_core::{Harness, HarnessConfig};
use tokio::sync::mpsc;

const USAGE: &str = "\
usage: psi [--continue] [prompt]
       with a prompt, one turn runs headless; with none, the TUI opens.

environment: OPENAI_API_KEY (required), PSI_MODEL, PSI_BASE_URL, PSI_SESSIONS_DIR

keys, composing:
  Enter          send the message
  Ctrl-J         newline (Alt-Enter also works)
  Esc            leave insert mode; in normal mode, cancel the running turn
  Ctrl-C         cancel the running turn, or quit at the prompt
  Ctrl-P/Ctrl-N  earlier and later prompts
  Ctrl-B         branch mode
  normal mode    i a o O x, h j k l, w b e, 0 $, dd, and counts on any of them

keys, branch mode:
  k / j          select an older or newer message on this branch
  Enter          edit the selected message: forks when you send it
  Tab / BackTab  switch branch
  Esc            back to composing";

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let (resume, prompt) = match args.split_first() {
        Some((flag, rest)) if flag == "--continue" => (true, rest.join(" ")),
        _ => (false, args.join(" ")),
    };

    let (commands, events) = match harness() {
        Ok(harness) => harness,
        Err(message) => {
            eprintln!("psi: {message}");
            return ExitCode::FAILURE;
        }
    };

    if prompt.is_empty() {
        return match tui::run(commands, events, resume).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("psi: {err}");
                ExitCode::FAILURE
            }
        };
    }
    headless::run(commands, events, resume, prompt).await
}

/// Builds the harness both modes talk to.
fn harness() -> Result<(mpsc::Sender<Command>, mpsc::Receiver<Event>), String> {
    let sessions_dir = sessions_dir().ok_or("set HOME or PSI_SESSIONS_DIR")?;
    let workspace = std::env::current_dir().map_err(|err| err.to_string())?;

    let mut config = OpenAiConfig::default();
    if let Ok(model) = std::env::var("PSI_MODEL") {
        config.model = model;
    }
    if let Ok(base_url) = std::env::var("PSI_BASE_URL") {
        config.base_url = base_url;
    }
    config.instructions = format!(
        "{}\n\nThe workspace root is {}.",
        config.instructions,
        workspace.display()
    );
    let model = OpenAiBackend::new(config).map_err(|err| err.to_string())?;

    Harness::spawn(HarnessConfig {
        model: Arc::new(model),
        tools: default_profile(workspace.clone()),
        hooks: HookRegistry::new(),
        workspace,
        sessions_dir,
    })
    .map_err(|err| format!("sessions directory: {err}"))
}

/// Session logs live in one directory rather than one per workspace, so
/// `--continue` and the TUI's branch view have a single place to look.
/// `PSI_SESSIONS_DIR` overrides it, which is how tests and benchmark runs keep
/// their sessions out of the user's.
fn sessions_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PSI_SESSIONS_DIR") {
        return Some(PathBuf::from(dir));
    }
    Some(PathBuf::from(std::env::var("HOME").ok()?).join(".psi/sessions"))
}

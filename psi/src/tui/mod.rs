//! The TUI: a client of the harness's command and event channels and nothing
//! more (docs/design.md, "The harness is the source of truth").
//!
//! Three things run at once. The harness runs in its own task, key presses are
//! read on a thread of their own, and this loop selects between them, so a
//! slow model never delays a keystroke and a burst of keystrokes never delays
//! the stream.

mod app;
mod composer;
mod draw;
mod files;
mod history;
mod view;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crossterm::event::{self, Event as TerminalEvent};
use psi_core::protocol::{Command, Event};
use tokio::sync::mpsc;

use app::App;
use history::History;

pub use app::HELP;

/// How long the reader thread waits for a key before checking whether Psi is
/// still there. Blocking on `read` forever would leave a thread that eats the
/// first keystroke the shell should have had.
const POLL: Duration = Duration::from_millis(100);

/// Runs until the user quits or the harness goes away. The terminal is restored
/// on every path out, including a panic.
///
/// `workspace` is what the `@` picker walks and `sessions_dir` is where the
/// prompt history is kept; both are the process's own configuration, which is
/// why they arrive as arguments rather than over the protocol.
pub async fn run(
    commands: mpsc::Sender<Command>,
    mut events: mpsc::Receiver<Event>,
    resume: bool,
    workspace: PathBuf,
    sessions_dir: PathBuf,
) -> io::Result<()> {
    let (history, prompts) = History::open(&sessions_dir);
    let mut terminal = draw::enter()?;
    let mut keys = spawn_reader();
    let mut app = App::new(workspace, history, prompts);
    app.start(resume);

    let outcome = 'session: loop {
        for command in app.take_commands() {
            // A harness that has stopped taking commands has stopped answering
            // too; there is nothing left to draw.
            if commands.send(command).await.is_err() {
                break 'session Ok(());
            }
        }
        if let Err(err) = draw::scrollback(&mut terminal, app.take_scrollback()) {
            break Err(err);
        }
        if let Err(err) = draw::frame(&mut terminal, &app) {
            break Err(err);
        }
        if app.should_quit() {
            break Ok(());
        }

        tokio::select! {
            key = keys.recv() => match key {
                Some(key) => app.on_terminal_event(key),
                // The reader thread only stops when the terminal does.
                None => break Ok(()),
            },
            event = events.recv() => match event {
                Some(event) => {
                    app.on_event(event);
                    // Deltas arrive far faster than a terminal can usefully be
                    // repainted; take everything waiting before drawing again.
                    while let Ok(event) = events.try_recv() {
                        app.on_event(event);
                    }
                }
                None => break Ok(()),
            },
        }
    };

    // Ordering matters: the viewport is cleared while the terminal still knows
    // where it is, and only then is raw mode given back.
    let left = draw::leave(&mut terminal);
    let restored = draw::restore();
    outcome.and(left).and(restored)
}

/// Reads key presses on a thread, because crossterm's reader blocks. Stops when
/// the receiver goes away, which is how it lets go of the terminal on exit.
fn spawn_reader() -> mpsc::Receiver<TerminalEvent> {
    let (tx, rx) = mpsc::channel(64);
    std::thread::spawn(move || {
        loop {
            match event::poll(POLL) {
                Ok(true) => match event::read() {
                    Ok(event) => {
                        if tx.blocking_send(event).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                },
                Ok(false) if tx.is_closed() => return,
                Ok(false) => {}
                Err(_) => return,
            }
        }
    });
    rx
}

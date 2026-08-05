//! The TUI's state machine: keys and protocol events in, protocol commands
//! out. It speaks only `Command` and `Event`, so branching, cancellation and
//! persistence stay harness behaviour that the TUI merely asks for.
//!
//! Keeping the terminal out of this file is what lets the milestone's
//! interactive slice be tested: a test can press keys and feed the real
//! harness's events without a terminal at all.
//!
//! Keys, in compose mode:
//!
//! ```text
//! Enter          submit the buffer
//! Ctrl-J         insert a newline (Alt-Enter also works)
//! Esc            insert mode -> normal mode; in normal mode, cancel the turn
//! Ctrl-C         cancel the running turn, or quit at the prompt
//! Ctrl-P/Ctrl-N  walk back and forward through submitted prompts
//! Ctrl-B         branch mode
//! ```
//!
//! and in branch mode:
//!
//! ```text
//! k / j          select an older or newer message on this branch
//! Enter          edit the selected message: fork here and load it to compose
//! Tab / BackTab  switch to the next or previous branch of the tree
//! Esc / Ctrl-B   back to composing
//! ```

use crossterm::event::{Event as TerminalEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use psi_core::item::{ItemId, ItemPayload};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::SessionId;

use super::composer::{Composer, Mode, Outcome};
use super::view::{DisplayLine, View};

/// Where the branch picker is pointing: which past message of the active path,
/// and which of the tree's branches that path is.
struct Branch {
    selected: usize,
    leaf: usize,
}

enum AppMode {
    Compose,
    Branch(Branch),
}

pub struct App {
    session: Option<SessionId>,
    view: View,
    composer: Composer,
    mode: AppMode,
    /// Commands the loop has yet to send. The app never awaits the harness, so
    /// a key press can never be delayed by one.
    outbox: Vec<Command>,
    /// Set once cancellation is asked for, cleared when the turn reports how it
    /// ended, so the status line can say the difference.
    cancelling: bool,
    quit: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            session: None,
            view: View::new(),
            composer: Composer::new(),
            mode: AppMode::Compose,
            outbox: Vec::new(),
            cancelling: false,
            quit: false,
        }
    }

    /// Asks for the session to work in. Resuming is a listing followed by a
    /// load, the same two commands `--continue` uses headless.
    pub fn start(&mut self, resume: bool) {
        self.outbox.push(if resume {
            Command::ListSessions
        } else {
            Command::CreateSession
        });
    }

    pub fn take_commands(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.outbox)
    }

    pub fn take_scrollback(&mut self) -> Vec<DisplayLine> {
        self.view.take_scrollback()
    }

    pub fn should_quit(&self) -> bool {
        self.quit
    }

    pub fn composer(&self) -> &Composer {
        &self.composer
    }

    /// What the viewport shows above the composer: the branch picker while it
    /// is open, otherwise whatever is still streaming.
    pub fn live(&self) -> Vec<DisplayLine> {
        match &self.mode {
            AppMode::Compose => self.view.live(),
            AppMode::Branch(branch) => self.view.branch_lines(
                branch.selected,
                branch.leaf,
                self.view.leaves().len().max(1),
            ),
        }
    }

    pub fn status(&self) -> String {
        let session = match &self.session {
            Some(id) => id.0.as_str(),
            None => "starting",
        };
        let state = if self.cancelling {
            "cancelling"
        } else if self.view.running() {
            "running"
        } else {
            "idle"
        };
        let usage = match self.view.usage() {
            Some(usage) => format!("  {}in {}out", usage.input_tokens, usage.output_tokens),
            None => String::new(),
        };
        let (mode, keys) = match self.mode {
            AppMode::Branch(_) => ("BRANCH", "k/j select  Enter edit  Tab branch  Esc back"),
            AppMode::Compose => match self.composer.mode() {
                Mode::Insert => ("INSERT", "Enter send  ^J newline  ^B branch  ^C quit"),
                Mode::Normal => ("NORMAL", "Enter send  Esc cancel  ^B branch  ^C quit"),
            },
        };
        format!("{mode}  {session}  {state}{usage}  —  {keys}")
    }

    pub fn on_event(&mut self, event: Event) {
        match &event.payload {
            EventPayload::SessionCreated { meta } => self.session = Some(meta.id.clone()),
            EventPayload::SessionLoaded { snapshot } => {
                self.session = Some(snapshot.meta.id.clone())
            }
            // Nothing to continue is not an error: start fresh.
            EventPayload::SessionsListed { sessions } => {
                self.outbox.push(match sessions.first() {
                    Some(meta) => Command::LoadSession {
                        session_id: meta.id.clone(),
                    },
                    None => Command::CreateSession,
                });
            }
            EventPayload::TurnFinished { .. } => self.cancelling = false,
            _ => {}
        }
        self.view.apply(&event.payload);
    }

    pub fn on_terminal_event(&mut self, event: TerminalEvent) {
        match event {
            // Release and repeat arrive from terminals that report them; a
            // composer that acted on both would type every character twice.
            TerminalEvent::Key(key) if key.kind == KeyEventKind::Press => self.on_key(key),
            TerminalEvent::Paste(text) => self.composer.paste(&text),
            _ => {}
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match (key.code, control) {
            (KeyCode::Char('c'), true) => {
                if self.view.running() {
                    self.cancel();
                } else {
                    self.quit = true;
                }
                return;
            }
            (KeyCode::Char('b'), true) => {
                self.toggle_branch_mode();
                return;
            }
            (KeyCode::Char('p'), true) => {
                self.composer.recall_previous();
                return;
            }
            (KeyCode::Char('n'), true) => {
                self.composer.recall_next();
                return;
            }
            _ => {}
        }
        match self.mode {
            AppMode::Compose => self.compose_key(key),
            AppMode::Branch(_) => self.branch_key(key),
        }
    }

    fn compose_key(&mut self, key: KeyEvent) {
        // Esc means "back out of what I am doing": out of insert mode first,
        // and only then out of a running turn.
        if key.code == KeyCode::Esc && self.composer.mode() == Mode::Normal {
            self.cancel();
            return;
        }
        if self.composer.key(key) == Outcome::Submit {
            self.submit();
        }
    }

    fn branch_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('k') | KeyCode::Up => self.move_selection(-1),
            KeyCode::Char('j') | KeyCode::Down => self.move_selection(1),
            KeyCode::Tab => self.switch_branch(1),
            KeyCode::BackTab => self.switch_branch(-1),
            KeyCode::Enter => self.fork(),
            KeyCode::Esc => self.mode = AppMode::Compose,
            _ => {}
        }
    }

    fn move_selection(&mut self, step: isize) {
        let last = self.view.user_messages().len().saturating_sub(1) as isize;
        let AppMode::Branch(branch) = &mut self.mode else {
            return;
        };
        branch.selected = (branch.selected as isize + step).clamp(0, last) as usize;
    }

    fn toggle_branch_mode(&mut self) {
        if matches!(self.mode, AppMode::Branch(_)) {
            self.mode = AppMode::Compose;
            return;
        }
        // A `set_head` sent mid-turn would be held until the turn ended and
        // then move the head under the user; branch mode waits instead.
        if self.view.running() || self.view.user_messages().is_empty() {
            return;
        }
        let leaves = self.view.leaves();
        let leaf = leaves
            .iter()
            .position(|id| Some(*id) == self.view.head())
            .unwrap_or(0);
        self.mode = AppMode::Branch(Branch {
            selected: self.view.user_messages().len() - 1,
            leaf,
        });
    }

    /// Moves the head to another branch tip. The whole branch is reprinted,
    /// because terminal scrollback only ever grows.
    fn switch_branch(&mut self, step: isize) {
        let leaves = self.view.leaves();
        if leaves.len() < 2 {
            return;
        }
        let AppMode::Branch(branch) = &mut self.mode else {
            return;
        };
        let count = leaves.len() as isize;
        branch.leaf = ((branch.leaf as isize + step).rem_euclid(count)) as usize;
        let leaf = leaves[branch.leaf];
        self.set_head(Some(leaf));
        let selected = self.view.user_messages().len().saturating_sub(1);
        if let AppMode::Branch(branch) = &mut self.mode {
            branch.selected = selected;
        }
    }

    /// Editing a past message is `set_head` to the item before it followed by a
    /// submit (docs/design.md, "Data Model"). This half moves the head and puts
    /// the old text back in the composer; submitting it is what forks.
    fn fork(&mut self) {
        let AppMode::Branch(branch) = &self.mode else {
            return;
        };
        let messages = self.view.user_messages();
        let Some(id) = messages.get(branch.selected).copied() else {
            return;
        };
        let Some(item) = self.view.item(id) else {
            return;
        };
        let parent = item.parent_id;
        let text = match &item.payload {
            ItemPayload::UserMessage { text } => text.clone(),
            _ => return,
        };
        self.set_head(parent);
        self.composer.load(&text);
        self.mode = AppMode::Compose;
    }

    fn set_head(&mut self, item_id: Option<ItemId>) {
        let Some(session_id) = self.session.clone() else {
            return;
        };
        self.outbox.push(Command::SetHead {
            session_id,
            item_id,
        });
        self.view.set_head(item_id);
    }

    fn submit(&mut self) {
        let Some(session_id) = self.session.clone() else {
            return;
        };
        if self.composer.is_blank() {
            return;
        }
        let text = self.composer.take();
        self.outbox
            .push(Command::SubmitMessage { session_id, text });
    }

    fn cancel(&mut self) {
        let Some(session_id) = self.session.clone() else {
            return;
        };
        if !self.view.running() {
            return;
        }
        self.cancelling = true;
        self.outbox.push(Command::CancelTurn { session_id });
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use psi_core::fake::{FakeModel, FakeResponse, FakeTool};
    use psi_core::hook::HookRegistry;
    use psi_core::item::CompletionStatus;
    use psi_core::model::{ModelEvent, ToolCallRequest};
    use psi_core::tool::{ToolEffect, ToolRegistry};
    use psi_core::{Harness, HarnessConfig};
    use serde_json::json;
    use tokio::sync::mpsc;

    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn typed(app: &mut App, keys: &str) {
        for c in keys.chars() {
            app.on_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    /// Drives one app against one real harness: sends whatever the app asked
    /// for, then feeds events back until the driver has seen enough.
    struct Driver {
        app: App,
        commands: mpsc::Sender<Command>,
        events: mpsc::Receiver<Event>,
        lines: Vec<String>,
        turns: usize,
    }

    impl Driver {
        /// Sends the app's pending commands and keeps its scrollback.
        async fn pump(&mut self) {
            for command in self.app.take_commands() {
                self.commands.send(command).await.unwrap();
            }
            self.lines
                .extend(self.app.take_scrollback().into_iter().map(|line| line.text));
        }

        async fn until(&mut self, done: impl Fn(&Self) -> bool) {
            loop {
                self.pump().await;
                if done(self) {
                    return;
                }
                let event = tokio::time::timeout(Duration::from_secs(5), self.events.recv())
                    .await
                    .expect("timed out waiting for an event")
                    .expect("event channel closed");
                if matches!(event.payload, EventPayload::TurnFinished { .. }) {
                    self.turns += 1;
                }
                self.app.on_event(event);
            }
        }

        /// The scrollback so far, blank separator lines dropped.
        fn transcript(&self) -> Vec<&str> {
            self.lines
                .iter()
                .map(String::as_str)
                .filter(|line| !line.is_empty())
                .collect()
        }
    }

    fn driver(script: Vec<FakeResponse>, sessions: &tempfile::TempDir) -> Driver {
        let mut tools = ToolRegistry::new();
        tools.register(FakeTool::canned(
            "read_file",
            ToolEffect::ReadOnly,
            "fake file contents",
        ));
        tools.register(FakeTool::canned(
            "apply_patch",
            ToolEffect::Mutating,
            "updated src/lib.sh",
        ));
        let (commands, events) = Harness::spawn(HarnessConfig {
            model: Arc::new(FakeModel::new(script)),
            tools,
            hooks: HookRegistry::new(),
            workspace: PathBuf::from("/fixture"),
            sessions_dir: sessions.path().to_path_buf(),
            trace: None,
            speculation: None,
        })
        .unwrap();
        Driver {
            app: App::new(),
            commands,
            events,
            lines: Vec::new(),
            turns: 0,
        }
    }

    fn text(message: &str) -> FakeResponse {
        FakeResponse::new(vec![
            ModelEvent::TextDelta {
                delta: message.into(),
            },
            ModelEvent::Completed,
        ])
    }

    /// The milestone's interactive slice, minus the terminal: start a session,
    /// stream a response, watch a tool run, see a diff, cancel a turn, edit a
    /// past message to fork, and switch branches.
    #[tokio::test]
    async fn the_tui_drives_a_session_from_keystrokes() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(
            vec![
                FakeResponse::new(vec![
                    ModelEvent::ToolCallCompleted {
                        call: ToolCallRequest {
                            call_id: "call-1".into(),
                            tool: "read_file".into(),
                            arguments: json!({ "path": "src/lib.sh" }),
                        },
                    },
                    ModelEvent::Completed,
                ]),
                FakeResponse::new(vec![
                    ModelEvent::ToolCallCompleted {
                        call: ToolCallRequest {
                            call_id: "call-2".into(),
                            tool: "apply_patch".into(),
                            arguments: json!({
                                "path": "src/lib.sh",
                                "old_text": "echo 41",
                                "new_text": "echo 42",
                            }),
                        },
                    },
                    ModelEvent::Completed,
                ]),
                text("Fixed it."),
                FakeResponse::hanging(vec![ModelEvent::TextDelta {
                    delta: "starting to".into(),
                }]),
                text("Fixed it differently."),
            ],
            &sessions,
        );

        // A session starts on its own, then the first prompt streams a turn
        // that runs a tool and shows a diff.
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        typed(&mut driver.app, "fix it");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;
        assert_eq!(
            driver.transcript(),
            [
                "> fix it",
                "• read_file {\"path\":\"src/lib.sh\"}",
                "  fake file contents",
                "• apply_patch src/lib.sh",
                "  -echo 41",
                "  +echo 42",
                "  updated src/lib.sh",
                "Fixed it.",
            ]
        );

        // The next turn hangs; Esc from normal mode cancels it, and the
        // cancellation is visible.
        driver.lines.clear();
        typed(&mut driver.app, "and again");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.view.running()).await;
        driver.app.on_key(key(KeyCode::Esc)); // insert -> normal
        driver.app.on_key(key(KeyCode::Esc)); // normal -> cancel the turn
        assert!(driver.app.cancelling);
        driver.until(|d| d.turns == 2).await;
        assert_eq!(
            driver.transcript(),
            [
                "> and again",
                "starting to",
                "  [cancelled]",
                "psi: turn cancelled",
            ]
        );
        assert!(!driver.app.cancelling);

        // Branch mode edits the second message: the head moves to the item
        // before it, and submitting the edited text forks.
        driver.lines.clear();
        driver.app.on_key(ctrl('b'));
        assert!(matches!(driver.app.mode, AppMode::Branch(_)));
        driver.app.on_key(key(KeyCode::Enter));
        assert_eq!(driver.app.composer.text(), "and again");
        typed(&mut driver.app, " please");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 3).await;
        assert_eq!(driver.app.view.leaves().len(), 2);
        assert_eq!(
            driver.transcript(),
            [
                "psi: branch of 6 items",
                "> fix it",
                "• read_file {\"path\":\"src/lib.sh\"}",
                "  fake file contents",
                "• apply_patch src/lib.sh",
                "  -echo 41",
                "  +echo 42",
                "  updated src/lib.sh",
                "Fixed it.",
                "> and again please",
                "Fixed it differently.",
            ]
        );

        // Tab switches back to the cancelled branch, which reprints it.
        driver.lines.clear();
        driver.app.on_key(ctrl('b'));
        driver.app.on_key(key(KeyCode::Tab));
        driver.pump().await;
        assert_eq!(
            driver.transcript().last(),
            Some(&"  [cancelled]"),
            "{:?}",
            driver.transcript()
        );

        // Ctrl-C at the prompt quits, which is what restores the terminal.
        driver.app.on_key(key(KeyCode::Esc));
        driver.app.on_key(ctrl('c'));
        assert!(driver.app.should_quit());

        // Both branches are durable, which is the harness's doing, not the
        // TUI's.
        driver
            .commands
            .send(Command::LoadSession {
                session_id: driver.app.session.clone().unwrap(),
            })
            .await
            .unwrap();
        let snapshot = loop {
            let event = driver.events.recv().await.unwrap();
            if let EventPayload::SessionLoaded { snapshot } = event.payload {
                break snapshot;
            }
        };
        let forks: Vec<&str> = snapshot
            .items
            .iter()
            .filter_map(|item| match &item.payload {
                ItemPayload::UserMessage { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(forks, ["fix it", "and again", "and again please"]);
        assert_eq!(
            snapshot
                .items
                .iter()
                .filter(|item| item.status == CompletionStatus::Cancelled)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn resuming_loads_the_most_recent_session() {
        let sessions = tempfile::tempdir().unwrap();
        let mut first = driver(vec![text("one")], &sessions);
        first.app.start(false);
        first.until(|d| d.app.session.is_some()).await;
        typed(&mut first.app, "hello");
        first.app.on_key(key(KeyCode::Enter));
        first.until(|d| d.turns == 1).await;
        let session = first.app.session.clone().unwrap();

        let mut second = driver(vec![text("two")], &sessions);
        second.app.start(true);
        second.until(|d| d.app.session.is_some()).await;
        assert_eq!(second.app.session, Some(session.clone()));
        assert_eq!(
            second.transcript(),
            [
                format!("psi: continuing {} (2 items)", session.0).as_str(),
                "> hello",
                "one",
            ]
        );
    }

    #[tokio::test]
    async fn an_empty_prompt_submits_nothing() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(Vec::new(), &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        typed(&mut driver.app, "   ");
        driver.app.on_key(key(KeyCode::Enter));
        assert!(driver.app.take_commands().is_empty());
        assert_eq!(driver.app.composer.text(), "   ");
    }

    #[tokio::test]
    async fn branch_mode_stays_shut_while_a_turn_runs() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(
            vec![FakeResponse::hanging(vec![ModelEvent::TextDelta {
                delta: "thinking".into(),
            }])],
            &sessions,
        );
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        typed(&mut driver.app, "go");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.view.running()).await;
        driver.app.on_key(ctrl('b'));
        assert!(matches!(driver.app.mode, AppMode::Compose));
    }
}

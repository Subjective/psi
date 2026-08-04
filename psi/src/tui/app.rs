//! The TUI's state machine: keys and protocol events in, protocol commands
//! out. It speaks only `Command` and `Event`, so branching, cancellation and
//! persistence stay harness behaviour that the TUI merely asks for.
//!
//! Keeping the terminal out of this file is what lets the milestone's
//! interactive slice be tested: a test can press keys and feed the real
//! harness's events without a terminal at all.
//!
//! The key and command surface is `HELP`, which `/help` prints into scrollback
//! and `psi --help` prints to the shell, so there is one list of it.
//!
//! Four overlays open under the composer — branch mode, the `/resume` session
//! list, the `@` file picker, and the `/` command palette. They are one widget
//! over different rows, and only one is open at a time.

use std::path::PathBuf;

use crossterm::event::{Event as TerminalEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use psi_core::item::{ItemId, ItemPayload};
use psi_core::protocol::{Command, Event, EventPayload};
use psi_core::session::SessionId;

use super::composer::{Composer, Mode, Outcome};
use super::files;
use super::history::History;
use super::view::{self, DisplayLine, Tone, View};

/// Wall-clock milliseconds, for the ages the session picker shows. The harness
/// stamps its own; this is the client asking what time it is now.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis() as u64
}

/// Everything Psi answers a key or a slash with. `/help` prints it into
/// scrollback; `psi --help` prints it to the shell.
pub const HELP: &[&str] = &[
    "keys, composing:",
    "  enter          send · ^j newline (alt-enter too)",
    "  esc            leave insert mode; again, cancel the running turn",
    "  ^c             cancel the running turn, or quit at the prompt",
    "  ^p / ^n        earlier and later prompts",
    "  ^a ^e ^u ^k ^w start · end · delete to start · to end · word back",
    "  @              file picker: type to filter, enter or tab inserts",
    "  ^b             branch tree: type to filter · enter edit · tab jump",
    "  normal mode    i a o O x, h j k l, w b e, 0 $, dd, and counts",
    "",
    "commands:",
    "  /new           start a fresh session",
    "  /resume        pick a session to load",
    "  /fork          branch mode, same as ^b",
    "  /help          this list",
    "  /quit          leave",
];

/// The branch picker: the whole tree of user messages with their depths, and
/// the ones still matching the query. The composer is the query box while the
/// picker is open, so `draft` holds whatever was being typed for Esc to
/// restore.
struct Branch {
    draft: String,
    matches: Vec<view::MessageRow>,
    selected: usize,
}

/// `/resume`: the sessions the harness listed, newest first, each as the id
/// that loads it and the row that names it. The rows are built when the
/// listing arrives rather than per frame, so the ages they show are the ages
/// at the moment the picker opened; `matches` is them, narrowed by the query.
struct Sessions {
    rows: Vec<(SessionId, String)>,
    draft: String,
    matches: Vec<(SessionId, String)>,
    selected: usize,
}

/// The slash commands, each with the line the `/` palette shows for it.
const COMMANDS: &[(&str, &str)] = &[
    ("/new", "start a fresh session"),
    ("/resume", "pick a session to load"),
    ("/fork", "branch mode, same as ^b"),
    ("/help", "keys and commands"),
    ("/quit", "leave"),
];

/// The `/` palette: the commands still matching what is typed after the slash.
struct Commands {
    matches: Vec<(&'static str, &'static str)>,
    selected: usize,
}

/// The `@` picker: where its `@` was, the walk it opened with, and the
/// matches the query typed since selects.
struct Files {
    /// Character offset just past the `@`, where the query starts.
    at: usize,
    entries: Vec<String>,
    matches: Vec<String>,
    selected: usize,
}

enum AppMode {
    Compose,
    Branch(Branch),
    Sessions(Sessions),
    Files(Files),
    Commands(Commands),
}

pub struct App {
    session: Option<SessionId>,
    view: View,
    composer: Composer,
    mode: AppMode,
    /// The root the `@` picker walks. The TUI reads it directly: completing a
    /// path is the client's business and crosses no protocol.
    workspace: PathBuf,
    /// Submitted prompts, appended as they are sent.
    history: History,
    /// Commands the loop has yet to send. The app never awaits the harness, so
    /// a key press can never be delayed by one.
    outbox: Vec<Command>,
    /// Set once cancellation is asked for, cleared when the turn reports how it
    /// ended, so the status line can say the difference.
    cancelling: bool,
    quit: bool,
}

impl App {
    /// `prompts` are the persisted prompts `history` was opened with; recall
    /// walks them and this run's as one list.
    pub fn new(workspace: PathBuf, history: History, prompts: Vec<String>) -> Self {
        Self {
            session: None,
            view: View::new(),
            composer: Composer::new(prompts),
            mode: AppMode::Compose,
            workspace,
            history,
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

    /// What the viewport shows above the composer: whatever is still
    /// streaming.
    pub fn live(&self) -> Vec<DisplayLine> {
        self.view.live()
    }

    /// The open picker, drawn below the composer so opening one never moves
    /// the text being typed.
    pub fn overlay(&self) -> Vec<DisplayLine> {
        match &self.mode {
            AppMode::Compose => Vec::new(),
            AppMode::Branch(branch) => match branch.matches.is_empty() {
                true => vec![DisplayLine::new(Tone::Notice, "no messages match")],
                false => view::picker(
                    branch
                        .matches
                        .iter()
                        .map(|row| {
                            let text = match self.view.item(row.id).map(|item| &item.payload) {
                                Some(ItemPayload::UserMessage { text }) => text.replace('\n', " "),
                                _ => String::new(),
                            };
                            // An alternative announces itself; what follows it
                            // sits flat underneath.
                            let indent = match row.alternative {
                                true => format!("{}└ ", "  ".repeat(row.depth.saturating_sub(1))),
                                false => "  ".repeat(row.depth),
                            };
                            format!("{indent}{text}")
                        })
                        .collect(),
                    branch.selected,
                ),
            },
            AppMode::Sessions(sessions) => {
                match (sessions.rows.is_empty(), sessions.matches.is_empty()) {
                    (true, _) => vec![DisplayLine::new(Tone::Notice, "no sessions yet")],
                    (false, true) => vec![DisplayLine::new(Tone::Notice, "no sessions match")],
                    _ => view::picker(
                        sessions
                            .matches
                            .iter()
                            .map(|(_, row)| row.clone())
                            .collect(),
                        sessions.selected,
                    ),
                }
            }
            AppMode::Files(files) => match files.matches.is_empty() {
                true => vec![DisplayLine::new(Tone::Notice, "no files match")],
                false => view::picker(files.matches.clone(), files.selected),
            },
            AppMode::Commands(commands) => match commands.matches.is_empty() {
                true => vec![DisplayLine::new(Tone::Notice, "no commands match")],
                false => view::picker(
                    commands
                        .matches
                        .iter()
                        .map(|(name, blurb)| format!("{name}  {blurb}"))
                        .collect(),
                    commands.selected,
                ),
            },
        }
    }

    /// The status row, which carries news and nothing else: an idle prompt
    /// says nothing at all. The composer's mode is on the cursor's shape, and
    /// every key is in `/help`.
    pub fn status(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.cancelling {
            parts.push("cancelling".to_string());
        } else if self.view.running() {
            // Esc leaves insert mode before it reaches the turn, so cancelling
            // from insert really does take two.
            let escapes = match self.composer.mode() {
                Mode::Normal => "esc",
                Mode::Insert => "esc esc",
            };
            parts.push(format!("running · {escapes} cancels"));
        }
        // Which branch the head is on, once there is more than one to be on.
        let leaves = self.view.leaves();
        if leaves.len() > 1
            && let Some(at) = leaves.iter().position(|id| Some(*id) == self.view.head())
        {
            parts.push(format!("branch {}/{}", at + 1, leaves.len()));
        }
        parts.join("  ")
    }

    pub fn on_event(&mut self, event: Event) {
        match &event.payload {
            EventPayload::SessionCreated { meta } => self.session = Some(meta.id.clone()),
            EventPayload::SessionLoaded { snapshot } => {
                self.session = Some(snapshot.meta.id.clone())
            }
            EventPayload::SessionsListed { sessions } => match &mut self.mode {
                // `/resume` asked; the listing is the picker's rows.
                AppMode::Sessions(picker) => {
                    let now = now_ms();
                    picker.rows = sessions
                        .iter()
                        .map(|meta| {
                            let age = view::age(meta.created_at_ms, now);
                            (meta.id.clone(), format!("{}  {age}", meta.id.0))
                        })
                        .collect();
                    picker.matches = picker.rows.clone();
                    picker.selected = 0;
                }
                // Startup asked, to continue where the user left off. Nothing
                // to continue is not an error: start fresh.
                _ => self.outbox.push(match sessions.first() {
                    Some(meta) => Command::LoadSession {
                        session_id: meta.id.clone(),
                    },
                    None => Command::CreateSession,
                }),
            },
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
            TerminalEvent::Paste(text) => {
                self.composer.paste(&text);
                // A paste into an open picker is more query, or the end of one.
                self.filter_files();
            }
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
            // With any overlay open, the two history keys move its selection.
            (KeyCode::Char('p'), true) => {
                match self.mode {
                    AppMode::Compose => self.composer.recall_previous(),
                    _ => self.move_selection(-1),
                }
                return;
            }
            (KeyCode::Char('n'), true) => {
                match self.mode {
                    AppMode::Compose => self.composer.recall_next(),
                    _ => self.move_selection(1),
                }
                return;
            }
            _ => {}
        }
        match self.mode {
            AppMode::Compose => self.compose_key(key),
            AppMode::Branch(_) => self.branch_key(key),
            AppMode::Sessions(_) => self.sessions_key(key),
            AppMode::Files(_) => self.files_key(key),
            AppMode::Commands(_) => self.commands_key(key),
        }
    }

    fn compose_key(&mut self, key: KeyEvent) {
        // Esc means "back out of what I am doing": out of insert mode first,
        // and only then out of a running turn.
        if key.code == KeyCode::Esc && self.composer.mode() == Mode::Normal {
            self.cancel();
            return;
        }
        let insert = self.composer.mode() == Mode::Insert;
        if self.composer.key(key) == Outcome::Submit {
            self.submit();
            return;
        }
        // `@` opens the file picker over whatever is typed after it — the `@`
        // the composer just inserted, not a chord that happens to carry one.
        let modified = key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        if insert && key.code == KeyCode::Char('@') && !modified {
            self.mode = AppMode::Files(Files {
                at: self.composer.offset(),
                entries: files::walk(&self.workspace),
                matches: Vec::new(),
                selected: 0,
            });
            self.filter_files();
        }
        // A `/` opening an empty prompt is a command coming; anywhere else it
        // is a character.
        if insert && key.code == KeyCode::Char('/') && !modified && self.composer.text() == "/" {
            self.mode = AppMode::Commands(Commands {
                matches: Vec::new(),
                selected: 0,
            });
            self.filter_commands();
        }
    }

    /// A submitted line starting with `/` is a command to Psi itself, not a
    /// message to the model.
    fn command(&mut self, line: &str) {
        let name = line.split_whitespace().next().unwrap_or_default();
        match name {
            "new" | "resume" if self.view.running() => {
                self.view.notice("psi: wait for the turn to finish");
            }
            "new" => self.outbox.push(Command::CreateSession),
            "resume" => {
                // The submit that carried `/resume` already cleared the
                // composer, which is the picker's query box from here.
                self.mode = AppMode::Sessions(Sessions {
                    rows: Vec::new(),
                    draft: String::new(),
                    matches: Vec::new(),
                    selected: 0,
                });
                self.outbox.push(Command::ListSessions);
            }
            "fork" => self.toggle_branch_mode(),
            "help" => {
                for line in HELP {
                    self.view.notice(*line);
                }
            }
            "quit" => self.quit = true,
            other => self
                .view
                .notice(format!("psi: no command /{other} — try /help")),
        }
    }

    /// Keys in the session picker: typing goes to the composer, which is the
    /// query box while the picker is open — like every typed picker, selection
    /// is arrows and ^p/^n.
    fn sessions_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Enter => self.load_selected_session(),
            KeyCode::Esc => self.close_picker(),
            _ => {
                self.composer.key(key);
                self.filter_sessions();
            }
        }
    }

    /// Narrows the rows to subsequence matches of the composer's text, keeping
    /// their newest-first order: typing filters, it does not re-rank.
    fn filter_sessions(&mut self) {
        let query = self.composer.text();
        let AppMode::Sessions(picker) = &mut self.mode else {
            return;
        };
        picker.matches = picker
            .rows
            .iter()
            .filter(|(_, row)| files::score(&query, row).is_some())
            .cloned()
            .collect();
        picker.selected = picker.selected.min(picker.matches.len().saturating_sub(1));
    }

    /// Closes the session or branch picker, giving the composer back what was
    /// being typed before it became the query box.
    fn close_picker(&mut self) {
        let draft = match &self.mode {
            AppMode::Sessions(picker) => picker.draft.clone(),
            AppMode::Branch(branch) => branch.draft.clone(),
            _ => return,
        };
        self.composer.load(&draft);
        self.mode = AppMode::Compose;
    }

    /// Keys in the file picker. Everything the picker does not claim is typing,
    /// which is what filters it — including `k` and `j`, which are ordinary
    /// characters in a path.
    fn files_key(&mut self, key: KeyEvent) {
        match key.code {
            // The typed text stays: the picker is a completion, not a prompt.
            KeyCode::Esc => self.mode = AppMode::Compose,
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Enter | KeyCode::Tab => self.insert_selected_path(),
            _ => {
                self.composer.key(key);
                self.filter_files();
            }
        }
    }

    /// Keys in the command palette: typing filters, Enter runs the selection,
    /// Tab only completes it into the prompt.
    fn commands_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.mode = AppMode::Compose,
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Enter => self.run_selected_command(),
            KeyCode::Tab => self.complete_selected_command(),
            _ => {
                self.composer.key(key);
                self.filter_commands();
            }
        }
    }

    /// Re-ranks the commands against what is typed after the slash. The
    /// palette closes when the slash is gone or the line takes whitespace —
    /// it is a name being completed, not a prompt.
    fn filter_commands(&mut self) {
        if !matches!(self.mode, AppMode::Commands(_)) {
            return;
        }
        let text = self.composer.text();
        let query = text
            .strip_prefix('/')
            .filter(|query| !query.chars().any(char::is_whitespace))
            .map(str::to_string);
        let Some(query) = query else {
            self.mode = AppMode::Compose;
            return;
        };
        let names: Vec<String> = COMMANDS
            .iter()
            .map(|(name, _)| name.trim_start_matches('/').to_string())
            .collect();
        let matches: Vec<(&'static str, &'static str)> = files::rank(&names, &query)
            .iter()
            .filter_map(|name| {
                COMMANDS
                    .iter()
                    .find(|(full, _)| full.trim_start_matches('/') == name)
                    .copied()
            })
            .collect();
        let AppMode::Commands(commands) = &mut self.mode else {
            return;
        };
        commands.selected = commands.selected.min(matches.len().saturating_sub(1));
        commands.matches = matches;
    }

    /// Runs the selection as if it had been typed in full and sent. With
    /// nothing matching, the line goes through as typed and the unknown-command
    /// notice answers it.
    fn run_selected_command(&mut self) {
        let AppMode::Commands(commands) = &self.mode else {
            return;
        };
        if let Some((name, _)) = commands.matches.get(commands.selected).copied() {
            self.composer.replace_range(0, name);
        }
        self.mode = AppMode::Compose;
        self.submit();
    }

    fn complete_selected_command(&mut self) {
        let AppMode::Commands(commands) = &self.mode else {
            return;
        };
        if let Some((name, _)) = commands.matches.get(commands.selected).copied() {
            self.composer.replace_range(0, name);
        }
        self.mode = AppMode::Compose;
    }

    /// Re-ranks the walk against the query typed since the `@`. The picker
    /// closes when its `@` is gone, and when the query takes whitespace: a
    /// completion that has to span a space is not what `@` is for.
    fn filter_files(&mut self) {
        let AppMode::Files(files) = &self.mode else {
            return;
        };
        let query = self
            .composer
            .text_after(files.at)
            .filter(|query| !query.chars().any(char::is_whitespace));
        let Some(query) = query else {
            self.mode = AppMode::Compose;
            return;
        };
        let matches = files::rank(&files.entries, &query);
        let AppMode::Files(files) = &mut self.mode else {
            return;
        };
        files.selected = files.selected.min(matches.len().saturating_sub(1));
        files.matches = matches;
    }

    /// Replaces `@query` with the selected path.
    fn insert_selected_path(&mut self) {
        let AppMode::Files(files) = &self.mode else {
            return;
        };
        let Some(path) = files.matches.get(files.selected).cloned() else {
            return;
        };
        // The `@` sits one character before the query it opened.
        let at = files.at.saturating_sub(1);
        self.composer.replace_range(at, &path);
        self.mode = AppMode::Compose;
    }

    fn load_selected_session(&mut self) {
        let AppMode::Sessions(picker) = &self.mode else {
            return;
        };
        let Some((session_id, _)) = picker.matches.get(picker.selected) else {
            return;
        };
        self.outbox.push(Command::LoadSession {
            session_id: session_id.clone(),
        });
        self.close_picker();
    }

    /// Keys in the branch tree: typing goes to the composer as the query,
    /// Enter edits the selection to fork, Tab jumps to the branch it is on.
    fn branch_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::Tab => self.jump_to_selected(),
            KeyCode::Enter => self.fork(),
            KeyCode::Esc => self.close_picker(),
            _ => {
                self.composer.key(key);
                self.filter_branch();
            }
        }
    }

    /// Narrows the tree to subsequence matches of the composer's text, keeping
    /// its depth-first order and each match's own indent.
    fn filter_branch(&mut self) {
        let query = self.composer.text();
        let tree = self.view.message_tree();
        let AppMode::Branch(branch) = &mut self.mode else {
            return;
        };
        branch.matches = tree
            .into_iter()
            .filter(|row| {
                let text = match self.view.item(row.id).map(|item| &item.payload) {
                    Some(ItemPayload::UserMessage { text }) => text.as_str(),
                    _ => "",
                };
                files::score(&query, text).is_some()
            })
            .collect();
        branch.selected = branch.selected.min(branch.matches.len().saturating_sub(1));
    }

    /// Moves the open overlay's selection, clamped to its rows.
    fn move_selection(&mut self, step: isize) {
        let rows = match &self.mode {
            AppMode::Compose => return,
            AppMode::Branch(branch) => branch.matches.len(),
            AppMode::Sessions(picker) => picker.matches.len(),
            AppMode::Files(files) => files.matches.len(),
            AppMode::Commands(commands) => commands.matches.len(),
        };
        let selected = match &mut self.mode {
            AppMode::Compose => return,
            AppMode::Branch(branch) => &mut branch.selected,
            AppMode::Sessions(picker) => &mut picker.selected,
            AppMode::Files(files) => &mut files.selected,
            AppMode::Commands(commands) => &mut commands.selected,
        };
        *selected = (*selected as isize + step).clamp(0, rows.saturating_sub(1) as isize) as usize;
    }

    fn toggle_branch_mode(&mut self) {
        if matches!(self.mode, AppMode::Branch(_)) {
            self.close_picker();
            return;
        }
        // A `set_head` sent mid-turn would be held until the turn ended and
        // then move the head under the user; branch mode waits instead.
        if self.view.running() || self.view.user_messages().is_empty() {
            return;
        }
        let matches = self.view.message_tree();
        // Open pointing at the active path's newest message.
        let selected = self
            .view
            .user_messages()
            .last()
            .and_then(|at| matches.iter().position(|row| row.id == *at))
            .unwrap_or(0);
        // The composer becomes the query box; what was being typed comes back
        // when the picker closes without choosing.
        let draft = self.composer.text();
        self.composer.load("");
        self.mode = AppMode::Branch(Branch {
            draft,
            matches,
            selected,
        });
    }

    /// Moves the head to the tip of the branch the selection is on — how an
    /// abandoned future comes back. The branch reprints, because terminal
    /// scrollback only ever grows.
    fn jump_to_selected(&mut self) {
        let AppMode::Branch(branch) = &self.mode else {
            return;
        };
        let Some(row) = branch.matches.get(branch.selected).copied() else {
            return;
        };
        let tip = self.view.tip_of(row.id);
        self.set_head(Some(tip));
        self.close_picker();
    }

    /// Editing a past message is `set_head` to the item before it followed by a
    /// submit (docs/design.md, "Data Model"). This half moves the head and puts
    /// the old text back in the composer; submitting it is what forks.
    fn fork(&mut self) {
        let AppMode::Branch(branch) = &self.mode else {
            return;
        };
        let Some(row) = branch.matches.get(branch.selected).copied() else {
            return;
        };
        let Some(item) = self.view.item(row.id) else {
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
        if self.composer.is_blank() {
            return;
        }
        // A command is not a message: it never reaches the harness, and it
        // runs before there is a session to send anything to. It also stays
        // out of the persisted history, which is the prompts; this run's
        // recall holds everything typed either way.
        if let Some(command) = self.composer.text().trim().strip_prefix('/') {
            let command = command.to_string();
            self.composer.take();
            self.command(&command);
            return;
        }
        let Some(session_id) = self.session.clone() else {
            return;
        };
        let text = self.composer.take();
        self.history.append(&text);
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

        /// The scrollback so far, blank separator lines dropped and tool
        /// durations normalised — a fake tool takes real time, so the number is
        /// whatever the machine was doing.
        fn transcript(&self) -> Vec<String> {
            self.lines
                .iter()
                .filter(|line| !line.is_empty())
                .map(|line| match line.starts_with("• ") {
                    true => match line.rsplit_once(" · ") {
                        Some((call, _)) => format!("{call} · Nms"),
                        None => line.clone(),
                    },
                    false => line.clone(),
                })
                .collect()
        }
    }

    fn driver(script: Vec<FakeResponse>, sessions: &tempfile::TempDir) -> Driver {
        driver_in(script, sessions, PathBuf::from("/fixture"))
    }

    /// A driver whose workspace is a real directory, which is what the `@`
    /// picker walks.
    fn driver_in(
        script: Vec<FakeResponse>,
        sessions: &tempfile::TempDir,
        workspace: PathBuf,
    ) -> Driver {
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
            workspace: workspace.clone(),
            sessions_dir: sessions.path().to_path_buf(),
            trace: None,
            speculation: None,
        })
        .unwrap();
        // The history lives in the test's sessions directory, so no test ever
        // reads or writes the user's own.
        let (history, prompts) = History::open(sessions.path());
        Driver {
            app: App::new(workspace, history, prompts),
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

        // A session starts on its own and says which one it is, then the first
        // prompt streams a turn that runs a tool and shows a diff.
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        let session = driver.app.session.clone().unwrap();
        assert_eq!(
            driver.transcript(),
            [format!("psi: new session {}", session.0)]
        );
        driver.lines.clear();

        typed(&mut driver.app, "fix it");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;
        // Every call carries what it cost, on the line that names it.
        assert_eq!(
            driver.transcript(),
            [
                "> fix it",
                "• read_file {\"path\":\"src/lib.sh\"} · Nms",
                "  fake file contents",
                "• apply_patch src/lib.sh · Nms",
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
                "• read_file {\"path\":\"src/lib.sh\"} · Nms",
                "  fake file contents",
                "• apply_patch src/lib.sh · Nms",
                "  -echo 41",
                "  +echo 42",
                "  updated src/lib.sh",
                "Fixed it.",
                "> and again please",
                "Fixed it differently.",
            ]
        );
        // A tree with two branches is news; which one the head is on is all
        // the status row says about it.
        assert_eq!(driver.app.status(), "branch 2/2");

        // The tree opens on the active path's newest message; the cancelled
        // branch's message sits one row up. Tab jumps back to it, reprinting.
        driver.lines.clear();
        driver.app.on_key(ctrl('b'));
        driver.app.on_key(key(KeyCode::Up));
        driver.app.on_key(key(KeyCode::Tab));
        driver.pump().await;
        assert_eq!(
            driver.transcript().last().map(String::as_str),
            Some("  [cancelled]"),
            "{:?}",
            driver.transcript()
        );
        assert_eq!(driver.app.status(), "branch 1/2");

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
    async fn slash_new_switches_to_a_fresh_session() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(vec![text("one")], &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        let first = driver.app.session.clone().unwrap();
        typed(&mut driver.app, "hello");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;
        driver.lines.clear();

        typed(&mut driver.app, "/new");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.session != Some(first.clone())).await;
        let second = driver.app.session.clone().unwrap();
        assert_eq!(
            driver.transcript(),
            [format!("psi: new session {}", second.0)]
        );
        // The view is the new session's: the old tree is gone, not hidden.
        assert!(driver.app.view.user_messages().is_empty());
        assert!(driver.app.view.leaves().is_empty());

        // The first session is untouched on disk, which is what /resume finds.
        driver
            .commands
            .send(Command::LoadSession { session_id: first })
            .await
            .unwrap();
        driver
            .until(|d| d.app.view.user_messages().len() == 1)
            .await;
    }

    #[tokio::test]
    async fn slash_resume_loads_the_session_picked_from_the_list() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(vec![text("one")], &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        let first = driver.app.session.clone().unwrap();
        typed(&mut driver.app, "hello");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;
        typed(&mut driver.app, "/new");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.session != Some(first.clone())).await;
        driver.lines.clear();

        typed(&mut driver.app, "/resume");
        driver.app.on_key(key(KeyCode::Enter));
        driver
            .until(|d| matches!(&d.app.mode, AppMode::Sessions(picker) if !picker.rows.is_empty()))
            .await;
        // Newest first, so the session the run started in is the second row.
        let rows: Vec<String> = driver
            .app
            .overlay()
            .iter()
            .map(|line| line.text.clone())
            .collect();
        // Each row is the id that loads it and how long ago it started.
        assert_eq!(rows[1], format!("  {}  just now", first.0));

        driver.app.on_key(key(KeyCode::Down));
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.session == Some(first.clone())).await;
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert_eq!(
            driver.transcript(),
            [
                format!("psi: continuing {} (2 items)", first.0),
                "> hello".to_string(),
                "one".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn picker_queries_type_through_the_composer_and_esc_restores_the_draft() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(vec![text("one")], &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        typed(&mut driver.app, "go");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;

        typed(&mut driver.app, "half a thought");
        driver.app.on_key(ctrl('b'));
        // The composer is the query box now, so typing is visible there.
        assert!(driver.app.composer.is_blank());
        typed(&mut driver.app, "zz");
        assert_eq!(driver.app.composer.text(), "zz");
        let rows: Vec<String> = driver
            .app
            .overlay()
            .iter()
            .map(|line| line.text.clone())
            .collect();
        assert_eq!(rows, ["no messages match"]);
        driver.app.on_key(key(KeyCode::Backspace));
        driver.app.on_key(key(KeyCode::Backspace));
        assert_eq!(driver.app.overlay().len(), 1);

        // Esc hands the composer back what was being typed.
        driver.app.on_key(key(KeyCode::Esc));
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert_eq!(driver.app.composer.text(), "half a thought");
    }

    #[tokio::test]
    async fn the_command_palette_filters_completes_and_runs() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(Vec::new(), &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        driver.lines.clear();

        // `/` alone offers every command; typing filters them.
        typed(&mut driver.app, "/");
        assert!(matches!(driver.app.mode, AppMode::Commands(_)));
        assert_eq!(driver.app.overlay().len(), COMMANDS.len());
        typed(&mut driver.app, "he");
        let rows: Vec<String> = driver
            .app
            .overlay()
            .iter()
            .map(|line| line.text.clone())
            .collect();
        assert_eq!(rows, ["> /help  keys and commands"]);

        // Enter runs the selection and clears the prompt.
        driver.app.on_key(key(KeyCode::Enter));
        driver.pump().await;
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert!(driver.app.composer.is_blank());
        let printed: Vec<&str> = HELP.iter().copied().filter(|l| !l.is_empty()).collect();
        assert_eq!(driver.transcript(), printed);

        // Tab completes the name without running anything.
        typed(&mut driver.app, "/re");
        driver.app.on_key(key(KeyCode::Tab));
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert_eq!(driver.app.composer.text(), "/resume");
        assert!(driver.app.take_commands().is_empty());
        driver.app.on_key(key(KeyCode::Esc));

        // A slash past the first character is a character.
        driver.app.on_key(key(KeyCode::Char('i')));
        typed(&mut driver.app, "a /");
        assert!(matches!(driver.app.mode, AppMode::Compose));
    }

    #[tokio::test]
    async fn slash_help_lists_the_keys_and_an_unknown_command_says_so() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(Vec::new(), &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        driver.lines.clear();

        typed(&mut driver.app, "/help");
        driver.app.on_key(key(KeyCode::Enter));
        driver.pump().await;
        // The transcript drops blank lines; the help's section break is one.
        let printed: Vec<&str> = HELP.iter().copied().filter(|l| !l.is_empty()).collect();
        assert_eq!(driver.transcript(), printed);
        assert!(driver.app.composer.is_blank());

        driver.lines.clear();
        typed(&mut driver.app, "/nope");
        driver.app.on_key(key(KeyCode::Enter));
        driver.pump().await;
        assert_eq!(driver.transcript(), ["psi: no command /nope — try /help"]);

        // Nothing a command does reaches the harness as a message.
        assert!(driver.app.take_commands().is_empty());
        typed(&mut driver.app, "/quit");
        driver.app.on_key(key(KeyCode::Enter));
        assert!(driver.app.should_quit());
    }

    #[tokio::test]
    async fn slash_fork_opens_branch_mode_like_ctrl_b() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(vec![text("one")], &sessions);
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        typed(&mut driver.app, "hello");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.turns == 1).await;

        typed(&mut driver.app, "/fork");
        driver.app.on_key(key(KeyCode::Enter));
        assert!(matches!(driver.app.mode, AppMode::Branch(_)));
        driver.app.on_key(key(KeyCode::Enter));
        assert_eq!(driver.app.composer.text(), "hello");
    }

    #[tokio::test]
    async fn the_file_picker_completes_a_workspace_path_into_the_prompt() {
        let sessions = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspace.path().join("src/tui")).unwrap();
        std::fs::write(workspace.path().join("src/tui/composer.rs"), "").unwrap();
        std::fs::write(workspace.path().join("README.md"), "").unwrap();
        let mut driver = driver_in(Vec::new(), &sessions, workspace.path().to_path_buf());
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;

        // `@` opens the picker; what follows filters it.
        typed(&mut driver.app, "look at @comp");
        assert!(matches!(driver.app.mode, AppMode::Files(_)));
        let rows: Vec<String> = driver
            .app
            .overlay()
            .iter()
            .map(|line| line.text.clone())
            .collect();
        assert_eq!(rows, ["> src/tui/composer.rs"]);
        driver.app.on_key(key(KeyCode::Enter));
        assert_eq!(driver.app.composer.text(), "look at src/tui/composer.rs");
        assert!(matches!(driver.app.mode, AppMode::Compose));

        // Ctrl-N walks the matches; ties rank the shorter path first.
        typed(&mut driver.app, " @src");
        driver.app.on_key(ctrl('n'));
        driver.app.on_key(key(KeyCode::Tab));
        assert_eq!(
            driver.app.composer.text(),
            "look at src/tui/composer.rs src/tui/"
        );

        // Esc closes the picker and keeps what was typed.
        typed(&mut driver.app, " @REA");
        driver.app.on_key(key(KeyCode::Esc));
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert_eq!(
            driver.app.composer.text(),
            "look at src/tui/composer.rs src/tui/ @REA"
        );
        // And the buffer is still a buffer: the picker never left insert mode.
        assert_eq!(driver.app.composer.mode(), Mode::Insert);
    }

    #[tokio::test]
    async fn the_file_picker_closes_when_its_marker_or_its_word_ends() {
        let sessions = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("README.md"), "").unwrap();
        let mut driver = driver_in(Vec::new(), &sessions, workspace.path().to_path_buf());
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;

        typed(&mut driver.app, "@RE");
        assert!(matches!(driver.app.mode, AppMode::Files(_)));
        // A space is the end of a path, so it is the end of the picker.
        typed(&mut driver.app, " ");
        assert!(matches!(driver.app.mode, AppMode::Compose));

        typed(&mut driver.app, "@R");
        assert!(matches!(driver.app.mode, AppMode::Files(_)));
        driver.app.on_key(key(KeyCode::Backspace));
        driver.app.on_key(key(KeyCode::Backspace));
        assert!(matches!(driver.app.mode, AppMode::Compose));
        assert_eq!(driver.app.composer.text(), "@RE ");
    }

    #[tokio::test]
    async fn prompts_persist_between_runs_and_commands_do_not() {
        let sessions = tempfile::tempdir().unwrap();
        let mut first = driver(vec![text("one")], &sessions);
        first.app.start(false);
        first.until(|d| d.app.session.is_some()).await;
        typed(&mut first.app, "remember this");
        first.app.on_key(key(KeyCode::Enter));
        first.until(|d| d.turns == 1).await;
        typed(&mut first.app, "/help");
        first.app.on_key(key(KeyCode::Enter));

        // One JSON string per line, so a prompt's own line breaks survive.
        let file = std::fs::read_to_string(sessions.path().join("history")).unwrap();
        assert_eq!(file, "\"remember this\"\n");

        let mut second = driver(vec![text("two")], &sessions);
        second.app.start(false);
        second.until(|d| d.app.session.is_some()).await;
        typed(&mut second.app, "and this");
        second.app.on_key(key(KeyCode::Enter));
        second.until(|d| d.turns == 1).await;

        // Persisted and in-process prompts are one list to walk back through.
        second.app.on_key(ctrl('p'));
        assert_eq!(second.app.composer.text(), "and this");
        second.app.on_key(ctrl('p'));
        assert_eq!(second.app.composer.text(), "remember this");
        second.app.on_key(ctrl('n'));
        assert_eq!(second.app.composer.text(), "and this");

        assert_eq!(
            std::fs::read_to_string(sessions.path().join("history")).unwrap(),
            "\"remember this\"\n\"and this\"\n"
        );
    }

    #[tokio::test]
    async fn the_status_row_reports_news_and_nothing_else() {
        let sessions = tempfile::tempdir().unwrap();
        let mut driver = driver(
            vec![FakeResponse::hanging(vec![ModelEvent::TextDelta {
                delta: "thinking".into(),
            }])],
            &sessions,
        );
        driver.app.start(false);
        driver.until(|d| d.app.session.is_some()).await;
        // An idle prompt in insert mode has nothing to say.
        assert_eq!(driver.app.status(), "");
        // Normal mode is on the cursor's shape, not the status row.
        driver.app.on_key(key(KeyCode::Esc));
        assert_eq!(driver.app.status(), "");

        driver.app.on_key(key(KeyCode::Char('i')));
        typed(&mut driver.app, "go");
        driver.app.on_key(key(KeyCode::Enter));
        driver.until(|d| d.app.view.running()).await;
        // From insert mode, the first Esc only leaves insert mode.
        assert_eq!(driver.app.status(), "running · esc esc cancels");
        driver.app.on_key(key(KeyCode::Esc));
        assert_eq!(driver.app.status(), "running · esc cancels");

        driver.app.on_key(key(KeyCode::Esc));
        assert_eq!(driver.app.status(), "cancelling");
        driver.until(|d| d.turns == 1).await;
        // The turn is over and normal mode lives on the cursor: nothing to say.
        assert_eq!(driver.app.status(), "");
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

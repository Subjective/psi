//! The view model: protocol events in, displayable lines out.
//!
//! Nothing here knows about Ratatui. A `DisplayLine` is text plus a tone; the
//! drawing layer decides what a tone looks like and how it wraps, so the
//! renderer can be replaced — the design doc's open question about a custom
//! line-diff renderer — without touching event handling.
//!
//! The view is also the client's mirror of the session tree. It is built from
//! the same two facts the harness records: `session_loaded` carries a snapshot,
//! and every `item_finished` carries a complete item with its parent. That is
//! enough to derive the active path, the branch tips, and which past messages
//! can be edited, so branch navigation needs nothing the protocol does not
//! already send.

use std::collections::HashMap;

use psi_core::item::{CompletionStatus, Item, ItemId, ItemKind, ItemPayload};
use psi_core::model::Usage;
use psi_core::protocol::EventPayload;
use psi_core::session::SessionSnapshot;

/// What a line is, so the drawing layer can style it. Tones exist because
/// something renders them differently; there is no tone without a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    User,
    Assistant,
    Reasoning,
    /// A tool call, and the header of a diff.
    Tool,
    ToolOutput,
    DiffAdded,
    DiffRemoved,
    /// Psi speaking about itself: branch moves, cancellation, prompts.
    Notice,
    Error,
    /// The selected row of the branch list; the drawing layer also scrolls the
    /// list to keep it visible.
    Selected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayLine {
    pub tone: Tone,
    pub text: String,
}

impl DisplayLine {
    pub fn new(tone: Tone, text: impl Into<String>) -> Self {
        Self {
            tone,
            text: text.into(),
        }
    }

    fn blank() -> Self {
        Self::new(Tone::Notice, String::new())
    }
}

/// How many lines of one tool result or one diff reach the scrollback. The
/// durable item keeps all of it; this only bounds what a single call may push
/// past the user.
const MAX_BLOCK_LINES: usize = 24;

/// An item that has started but not finished: `item_started` and its deltas
/// have arrived, the durable record has not.
struct Open {
    id: ItemId,
    kind: ItemKind,
    text: String,
    /// Complete lines of `text` already flushed to the scrollback, so a long
    /// message scrolls as it streams instead of landing all at once.
    flushed: usize,
}

pub struct View {
    items: Vec<Item>,
    index: HashMap<ItemId, usize>,
    head: Option<ItemId>,
    open: Option<Open>,
    /// Lines that have become final, waiting to be handed to the terminal.
    scrollback: Vec<DisplayLine>,
    /// Whether anything has been emitted yet, so blocks are separated by a
    /// blank line without one opening the session.
    emitted: bool,
    running: bool,
    usage: Option<Usage>,
}

impl View {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            index: HashMap::new(),
            head: None,
            open: None,
            scrollback: Vec::new(),
            emitted: false,
            running: false,
            usage: None,
        }
    }

    pub fn running(&self) -> bool {
        self.running
    }

    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    pub fn head(&self) -> Option<ItemId> {
        self.head
    }

    pub fn item(&self, id: ItemId) -> Option<&Item> {
        self.index.get(&id).map(|position| &self.items[*position])
    }

    /// The active conversation: root to head, in order.
    fn path(&self) -> Vec<&Item> {
        let mut path = Vec::new();
        let mut cursor = self.head;
        while let Some(id) = cursor {
            let Some(item) = self.item(id) else { break };
            path.push(item);
            cursor = item.parent_id;
        }
        path.reverse();
        path
    }

    /// The user messages on the active path, oldest first. These are what
    /// branch mode offers to edit.
    pub fn user_messages(&self) -> Vec<ItemId> {
        self.path()
            .into_iter()
            .filter(|item| item.payload.kind() == ItemKind::UserMessage)
            .map(|item| item.id)
            .collect()
    }

    /// Every branch tip in the tree, in id order. A leaf is any item that is no
    /// item's parent, which is the design doc's definition of a branch.
    pub fn leaves(&self) -> Vec<ItemId> {
        let parents: Vec<ItemId> = self
            .items
            .iter()
            .filter_map(|item| item.parent_id)
            .collect();
        self.items
            .iter()
            .map(|item| item.id)
            .filter(|id| !parents.contains(id))
            .collect()
    }

    pub fn take_scrollback(&mut self) -> Vec<DisplayLine> {
        std::mem::take(&mut self.scrollback)
    }

    /// The lines that are still changing: the tail of whatever is streaming.
    /// Redrawn every frame, never written to the scrollback until it is final.
    pub fn live(&self) -> Vec<DisplayLine> {
        let Some(open) = &self.open else {
            return if self.running {
                vec![DisplayLine::new(Tone::Notice, "… working")]
            } else {
                Vec::new()
            };
        };
        match open.kind {
            ItemKind::AssistantMessage | ItemKind::Reasoning => {
                let tone = tone_of(open.kind);
                let tail = open.text.split('\n').next_back().unwrap_or_default();
                vec![DisplayLine::new(tone, tail)]
            }
            // `item_started` names the kind, not the tool: a call's name arrives
            // with its complete record. Until then the arguments are all there
            // is to show.
            ItemKind::ToolCall => vec![DisplayLine::new(
                Tone::Tool,
                format!("• {}", open.text.trim()),
            )],
            ItemKind::ToolResult => {
                let tool = match self.items.last().map(|item| &item.payload) {
                    Some(ItemPayload::ToolCall { tool, .. }) => tool.as_str(),
                    _ => "tool",
                };
                vec![DisplayLine::new(Tone::Notice, format!("… running {tool}"))]
            }
            ItemKind::UserMessage => Vec::new(),
        }
    }

    /// The branch picker: the past messages of the active path, plus where that
    /// path sits among the tree's branches.
    pub fn branch_lines(&self, selected: usize, leaf: usize, leaves: usize) -> Vec<DisplayLine> {
        let mut lines = vec![DisplayLine::new(
            Tone::Notice,
            format!("branch {}/{leaves} — edit a past message to fork", leaf + 1),
        )];
        for (position, id) in self.user_messages().into_iter().enumerate() {
            let text = match self.item(id).map(|item| &item.payload) {
                Some(ItemPayload::UserMessage { text }) => text.replace('\n', " "),
                _ => String::new(),
            };
            let (tone, marker) = if position == selected {
                (Tone::Selected, ">")
            } else {
                (Tone::User, " ")
            };
            lines.push(DisplayLine::new(tone, format!("{marker} {text}")));
        }
        lines
    }

    pub fn apply(&mut self, payload: &EventPayload) {
        match payload {
            EventPayload::SessionLoaded { snapshot } => self.load(snapshot),
            EventPayload::TurnStarted { .. } => self.running = true,
            EventPayload::ItemStarted { item_id, kind } => {
                self.open = Some(Open {
                    id: *item_id,
                    kind: *kind,
                    text: String::new(),
                    flushed: 0,
                });
            }
            EventPayload::ItemDelta { item_id, delta } => {
                if let Some(open) = self.open.as_mut()
                    && open.id == *item_id
                {
                    open.text.push_str(delta);
                }
                self.flush_streamed_lines();
            }
            EventPayload::ItemFinished { item } => self.finish(item),
            EventPayload::TurnFinished {
                status,
                error,
                usage,
                ..
            } => {
                self.running = false;
                self.open = None;
                if usage.is_some() {
                    self.usage = *usage;
                }
                match status {
                    CompletionStatus::Cancelled => {
                        self.push(DisplayLine::new(Tone::Notice, "psi: turn cancelled"));
                    }
                    CompletionStatus::Failed => {
                        let error = error.as_deref().unwrap_or("the turn failed");
                        self.push(DisplayLine::new(Tone::Error, format!("psi: {error}")));
                    }
                    CompletionStatus::Completed => {}
                }
            }
            EventPayload::SessionCreated { .. } | EventPayload::SessionsListed { .. } => {}
        }
    }

    /// Moves head client-side after the same `set_head` was sent to the
    /// harness, and reprints the branch it selects. The terminal's scrollback
    /// is append-only, so a branch switch is a fresh printing of that branch
    /// rather than a rewrite of what is above.
    pub fn set_head(&mut self, head: Option<ItemId>) {
        self.head = head;
        let items: Vec<Item> = self.path().into_iter().cloned().collect();
        self.push(DisplayLine::new(
            Tone::Notice,
            format!("psi: branch of {} items", items.len()),
        ));
        for item in &items {
            self.render(item, 0);
        }
    }

    fn load(&mut self, snapshot: &SessionSnapshot) {
        self.items = snapshot.items.clone();
        self.index = self
            .items
            .iter()
            .enumerate()
            .map(|(position, item)| (item.id, position))
            .collect();
        self.head = snapshot.head;
        self.push(DisplayLine::new(
            Tone::Notice,
            format!(
                "psi: continuing {} ({} items)",
                snapshot.meta.id.0,
                snapshot.items.len()
            ),
        ));
        for item in self.path().into_iter().cloned().collect::<Vec<_>>() {
            self.render(&item, 0);
        }
    }

    /// Hands complete lines of a streaming item to the scrollback as they
    /// close, so only the unfinished tail stays in the viewport.
    fn flush_streamed_lines(&mut self) {
        let Some(open) = &self.open else { return };
        if !matches!(open.kind, ItemKind::AssistantMessage | ItemKind::Reasoning) {
            return;
        }
        let tone = tone_of(open.kind);
        // Everything before the last line break is final; the piece after it is
        // still being written.
        let mut complete: Vec<String> = open.text.split('\n').map(str::to_string).collect();
        complete.pop();
        if complete.len() <= open.flushed {
            return;
        }
        // The block is separated once, when its first line lands, not on every
        // flush that follows.
        let first = open.flushed == 0;
        let lines = complete.split_off(open.flushed);
        if let Some(open) = self.open.as_mut() {
            open.flushed += lines.len();
        }
        if first {
            self.separate();
        }
        for line in lines {
            self.push(DisplayLine::new(tone, line));
        }
    }

    fn finish(&mut self, item: &Item) {
        let flushed = match self.open.take() {
            Some(open) if open.id == item.id => open.flushed,
            _ => 0,
        };
        if !self.index.contains_key(&item.id) {
            self.index.insert(item.id, self.items.len());
            self.items.push(item.clone());
        }
        self.head = Some(item.id);
        self.render(item, flushed);
    }

    /// One item as scrollback lines. `skip` is the number of leading lines
    /// already flushed while the item streamed.
    fn render(&mut self, item: &Item, skip: usize) {
        match &item.payload {
            ItemPayload::UserMessage { text } => {
                self.separate();
                for line in text.lines() {
                    self.push(DisplayLine::new(Tone::User, format!("> {line}")));
                }
            }
            ItemPayload::AssistantMessage { text } => {
                self.text_block(text, skip, Tone::Assistant);
            }
            ItemPayload::Reasoning { text, .. } => {
                self.text_block(text, skip, Tone::Reasoning);
            }
            ItemPayload::ToolCall {
                tool, arguments, ..
            } => {
                self.separate();
                let mut lines = Vec::new();
                // Diffs render from the `apply_patch` call's arguments; they are
                // not items of their own (docs/design.md, "Data Model").
                if tool == "apply_patch" {
                    lines = diff_lines(arguments);
                }
                if lines.is_empty() {
                    let arguments = compact(arguments);
                    lines.push(DisplayLine::new(
                        Tone::Tool,
                        format!("• {tool} {arguments}"),
                    ));
                }
                self.push_capped(lines);
            }
            ItemPayload::ToolResult {
                content, truncated, ..
            } => {
                let tone = if item.status == CompletionStatus::Failed {
                    Tone::Error
                } else {
                    Tone::ToolOutput
                };
                let mut lines: Vec<DisplayLine> = content
                    .lines()
                    .map(|line| DisplayLine::new(tone, format!("  {line}")))
                    .collect();
                if *truncated {
                    lines.push(DisplayLine::new(Tone::Notice, "  [output truncated]"));
                }
                self.push_capped(lines);
            }
        }
        // A tool result already carries its failure as its content; anything
        // else needs the error said out loud.
        if item.payload.kind() != ItemKind::ToolResult
            && let Some(error) = &item.error
        {
            self.push(DisplayLine::new(Tone::Error, format!("  {error}")));
        }
        if item.status == CompletionStatus::Cancelled {
            self.push(DisplayLine::new(Tone::Notice, "  [cancelled]"));
        }
    }

    fn text_block(&mut self, text: &str, skip: usize, tone: Tone) {
        // Split the way the streaming flush counts lines, so `skip` lines up
        // with what the viewport already handed to the scrollback.
        let mut lines: Vec<&str> = text.split('\n').collect();
        if lines.last() == Some(&"") {
            lines.pop();
        }
        let lines: Vec<&str> = lines.into_iter().skip(skip).collect();
        if lines.is_empty() {
            return;
        }
        if skip == 0 {
            self.separate();
        }
        for line in lines {
            self.push(DisplayLine::new(tone, line));
        }
    }

    fn push_capped(&mut self, lines: Vec<DisplayLine>) {
        let total = lines.len();
        for line in lines.into_iter().take(MAX_BLOCK_LINES) {
            self.push(line);
        }
        if total > MAX_BLOCK_LINES {
            let hidden = total - MAX_BLOCK_LINES;
            self.push(DisplayLine::new(
                Tone::Notice,
                format!("  … {hidden} more lines"),
            ));
        }
    }

    /// A blank line between blocks, except at the very start of the session.
    fn separate(&mut self) {
        if self.emitted {
            self.scrollback.push(DisplayLine::blank());
        }
    }

    fn push(&mut self, line: DisplayLine) {
        self.emitted = true;
        self.scrollback.push(line);
    }
}

fn tone_of(kind: ItemKind) -> Tone {
    match kind {
        ItemKind::Reasoning => Tone::Reasoning,
        _ => Tone::Assistant,
    }
}

/// A tool call's arguments on one line. Tool schemas are the model's business,
/// not the TUI's, so every tool but `apply_patch` renders the same way.
fn compact(arguments: &serde_json::Value) -> String {
    match arguments {
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// One side of a patch as lines. Text that ends in a line break ends the last
/// line rather than starting an empty one, so a whole-file replacement does not
/// report a change that is not there.
fn patch_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// An `apply_patch` call as a diff. Only the lines that differ are shown: the
/// call's `old_text` carries whatever surrounding lines it needed to be unique,
/// and repeating those as context says nothing about the edit.
///
/// Returns nothing for arguments that are not an `apply_patch` call, including
/// the half-streamed ones a failed call records, so the caller falls back to
/// printing them plainly.
fn diff_lines(arguments: &serde_json::Value) -> Vec<DisplayLine> {
    let (Some(path), Some(old), Some(new)) = (
        arguments.get("path").and_then(|v| v.as_str()),
        arguments.get("old_text").and_then(|v| v.as_str()),
        arguments.get("new_text").and_then(|v| v.as_str()),
    ) else {
        return Vec::new();
    };

    let (old, new) = (patch_lines(old), patch_lines(new));
    let prefix = old
        .iter()
        .zip(new.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let mut lines = vec![DisplayLine::new(
        Tone::Tool,
        format!("• apply_patch {path}"),
    )];
    for line in &old[prefix..old.len() - suffix] {
        lines.push(DisplayLine::new(Tone::DiffRemoved, format!("  -{line}")));
    }
    for line in &new[prefix..new.len() - suffix] {
        lines.push(DisplayLine::new(Tone::DiffAdded, format!("  +{line}")));
    }
    lines
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use psi_core::item::{TurnId, WorkspaceRevision};
    use psi_core::session::{SessionId, SessionMeta};
    use serde_json::json;

    use super::*;

    fn item(id: u64, parent: Option<u64>, payload: ItemPayload) -> Item {
        Item {
            id: ItemId(id),
            parent_id: parent.map(ItemId),
            turn_id: TurnId(0),
            created_at_ms: 0,
            status: CompletionStatus::Completed,
            error: None,
            payload,
        }
    }

    fn user(id: u64, parent: Option<u64>, text: &str) -> Item {
        item(
            id,
            parent,
            ItemPayload::UserMessage {
                text: text.to_string(),
            },
        )
    }

    fn assistant(id: u64, parent: Option<u64>, text: &str) -> Item {
        item(
            id,
            parent,
            ItemPayload::AssistantMessage {
                text: text.to_string(),
            },
        )
    }

    fn call(id: u64, parent: Option<u64>, tool: &str, arguments: serde_json::Value) -> Item {
        item(
            id,
            parent,
            ItemPayload::ToolCall {
                tool: tool.to_string(),
                call_id: "call-1".to_string(),
                arguments,
                cwd: PathBuf::from("/workspace"),
                revision: WorkspaceRevision(0),
            },
        )
    }

    fn finished(item: Item) -> EventPayload {
        EventPayload::ItemFinished { item }
    }

    fn started(id: u64, kind: ItemKind) -> EventPayload {
        EventPayload::ItemStarted {
            item_id: ItemId(id),
            kind,
        }
    }

    fn delta(id: u64, delta: &str) -> EventPayload {
        EventPayload::ItemDelta {
            item_id: ItemId(id),
            delta: delta.to_string(),
        }
    }

    /// The scrollback as `(tone, text)` pairs, which is exactly what the
    /// drawing layer receives.
    fn drain(view: &mut View) -> Vec<(Tone, String)> {
        view.take_scrollback()
            .into_iter()
            .map(|line| (line.tone, line.text))
            .collect()
    }

    fn texts(view: &mut View) -> Vec<String> {
        drain(view).into_iter().map(|(_, text)| text).collect()
    }

    #[test]
    fn a_turn_renders_as_a_user_message_and_a_reply() {
        let mut view = View::new();
        view.apply(&EventPayload::TurnStarted { turn_id: TurnId(0) });
        view.apply(&started(0, ItemKind::UserMessage));
        view.apply(&finished(user(0, None, "make the test pass")));
        view.apply(&started(1, ItemKind::AssistantMessage));
        view.apply(&delta(1, "All"));
        view.apply(&delta(1, " done."));
        view.apply(&finished(assistant(1, Some(0), "All done.")));
        view.apply(&EventPayload::TurnFinished {
            turn_id: TurnId(0),
            status: CompletionStatus::Completed,
            error: None,
            usage: None,
        });

        assert_eq!(
            drain(&mut view),
            [
                (Tone::User, "> make the test pass".to_string()),
                (Tone::Notice, String::new()),
                (Tone::Assistant, "All done.".to_string()),
            ]
        );
        assert!(!view.running());
    }

    #[test]
    fn streaming_text_stays_live_until_a_line_closes() {
        let mut view = View::new();
        view.apply(&started(0, ItemKind::AssistantMessage));
        view.apply(&delta(0, "first half"));
        assert!(drain(&mut view).is_empty());
        assert_eq!(
            view.live(),
            [DisplayLine::new(Tone::Assistant, "first half")]
        );

        // The line break makes the line final: it leaves the viewport for the
        // scrollback, and only the new tail stays live.
        view.apply(&delta(0, " of it\nsecond"));
        assert_eq!(texts(&mut view), ["first half of it"]);
        assert_eq!(view.live(), [DisplayLine::new(Tone::Assistant, "second")]);

        view.apply(&finished(assistant(
            0,
            None,
            "first half of it\nsecond line",
        )));
        assert_eq!(texts(&mut view), ["second line"]);
        assert!(view.live().is_empty());
    }

    #[test]
    fn a_message_streaming_line_by_line_is_separated_once() {
        let mut view = View::new();
        view.apply(&finished(user(0, None, "go")));
        view.apply(&started(1, ItemKind::AssistantMessage));
        for chunk in ["one\n", "two\n", "three"] {
            view.apply(&delta(1, chunk));
        }
        view.apply(&finished(assistant(1, Some(0), "one\ntwo\nthree")));
        assert_eq!(texts(&mut view), ["> go", "", "one", "two", "three"]);
    }

    #[test]
    fn reasoning_and_assistant_text_carry_different_tones() {
        let mut view = View::new();
        view.apply(&started(0, ItemKind::Reasoning));
        view.apply(&delta(0, "Let me look."));
        assert_eq!(view.live()[0].tone, Tone::Reasoning);
        view.apply(&finished(item(
            0,
            None,
            ItemPayload::Reasoning {
                text: "Let me look.".to_string(),
                provider_data: None,
            },
        )));
        assert_eq!(drain(&mut view), [(Tone::Reasoning, "Let me look.".into())]);
    }

    #[test]
    fn a_tool_call_and_its_result_render_as_one_block() {
        let mut view = View::new();
        view.apply(&started(0, ItemKind::ToolCall));
        view.apply(&delta(0, "{\"path\":"));
        assert_eq!(view.live(), [DisplayLine::new(Tone::Tool, "• {\"path\":")]);
        view.apply(&finished(call(
            0,
            None,
            "read_file",
            json!({ "path": "src/lib.sh" }),
        )));
        view.apply(&started(1, ItemKind::ToolResult));
        // The result item carries no tool name; the call before it does.
        assert_eq!(
            view.live(),
            [DisplayLine::new(Tone::Notice, "… running read_file")]
        );
        view.apply(&finished(item(
            1,
            Some(0),
            ItemPayload::ToolResult {
                call_id: "call-1".to_string(),
                content: "answer() {\n  echo 41\n}".to_string(),
                duration_ms: 3,
                truncated: false,
            },
        )));

        assert_eq!(
            drain(&mut view),
            [
                (
                    Tone::Tool,
                    "• read_file {\"path\":\"src/lib.sh\"}".to_string()
                ),
                (Tone::ToolOutput, "  answer() {".to_string()),
                (Tone::ToolOutput, "    echo 41".to_string()),
                (Tone::ToolOutput, "  }".to_string()),
            ]
        );
    }

    #[test]
    fn a_failed_tool_result_is_toned_as_an_error() {
        let mut view = View::new();
        let mut result = item(
            0,
            None,
            ItemPayload::ToolResult {
                call_id: "call-1".to_string(),
                content: "read_file ../secret: escapes the workspace root".to_string(),
                duration_ms: 1,
                truncated: false,
            },
        );
        result.status = CompletionStatus::Failed;
        result.error = Some("read_file ../secret: escapes the workspace root".to_string());
        view.apply(&finished(result));
        assert_eq!(
            drain(&mut view),
            [(
                Tone::Error,
                "  read_file ../secret: escapes the workspace root".to_string()
            )]
        );
    }

    #[test]
    fn apply_patch_arguments_render_as_a_diff() {
        let mut view = View::new();
        view.apply(&finished(call(
            0,
            None,
            "apply_patch",
            json!({
                "path": "src/lib.sh",
                "old_text": "answer() {\n  echo 41\n}",
                "new_text": "answer() {\n  echo 42\n}",
            }),
        )));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch src/lib.sh".to_string()),
                (Tone::DiffRemoved, "  -  echo 41".to_string()),
                (Tone::DiffAdded, "  +  echo 42".to_string()),
            ]
        );
    }

    #[test]
    fn creating_a_file_renders_as_added_lines_only() {
        let mut view = View::new();
        // The trailing line break ends the last line; it is not a third one.
        view.apply(&finished(call(
            0,
            None,
            "apply_patch",
            json!({ "path": "new.txt", "old_text": "", "new_text": "one\ntwo\n" }),
        )));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch new.txt".to_string()),
                (Tone::DiffAdded, "  +one".to_string()),
                (Tone::DiffAdded, "  +two".to_string()),
            ]
        );
    }

    #[test]
    fn a_call_whose_arguments_never_finished_still_renders() {
        let mut view = View::new();
        let mut broken = call(0, None, "apply_patch", serde_json::Value::Null);
        broken.status = CompletionStatus::Failed;
        broken.error = Some("the response completed before the arguments did".to_string());
        view.apply(&finished(broken));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch ".to_string()),
                (
                    Tone::Error,
                    "  the response completed before the arguments did".to_string()
                ),
            ]
        );
    }

    #[test]
    fn a_long_tool_result_is_capped() {
        let mut view = View::new();
        let content: String = (0..40)
            .map(|n| format!("line {n}\n"))
            .collect::<Vec<_>>()
            .concat();
        view.apply(&finished(item(
            0,
            None,
            ItemPayload::ToolResult {
                call_id: "call-1".to_string(),
                content,
                duration_ms: 1,
                truncated: false,
            },
        )));
        let lines = texts(&mut view);
        assert_eq!(lines.len(), MAX_BLOCK_LINES + 1);
        assert_eq!(lines[MAX_BLOCK_LINES], "  … 16 more lines");
    }

    #[test]
    fn cancellation_is_visible_on_the_item_and_the_turn() {
        let mut view = View::new();
        view.apply(&EventPayload::TurnStarted { turn_id: TurnId(0) });
        view.apply(&started(0, ItemKind::AssistantMessage));
        view.apply(&delta(0, "partial"));
        let mut partial = assistant(0, None, "partial");
        partial.status = CompletionStatus::Cancelled;
        view.apply(&finished(partial));
        view.apply(&EventPayload::TurnFinished {
            turn_id: TurnId(0),
            status: CompletionStatus::Cancelled,
            error: None,
            usage: None,
        });
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Assistant, "partial".to_string()),
                (Tone::Notice, "  [cancelled]".to_string()),
                (Tone::Notice, "psi: turn cancelled".to_string()),
            ]
        );
        assert!(!view.running());
    }

    #[test]
    fn a_failed_turn_reports_its_error() {
        let mut view = View::new();
        view.apply(&EventPayload::TurnStarted { turn_id: TurnId(0) });
        view.apply(&EventPayload::TurnFinished {
            turn_id: TurnId(0),
            status: CompletionStatus::Failed,
            error: Some("connection reset".to_string()),
            usage: None,
        });
        assert_eq!(
            drain(&mut view),
            [(Tone::Error, "psi: connection reset".to_string())]
        );
    }

    #[test]
    fn a_running_turn_with_nothing_open_still_says_so() {
        let mut view = View::new();
        view.apply(&EventPayload::TurnStarted { turn_id: TurnId(0) });
        assert_eq!(view.live(), [DisplayLine::new(Tone::Notice, "… working")]);
    }

    /// A fork: two replies under one user message, so the tree has two leaves.
    fn forked() -> View {
        let mut view = View::new();
        for payload in [
            finished(user(0, None, "first")),
            finished(assistant(1, Some(0), "one")),
            finished(user(2, Some(1), "second")),
            finished(assistant(3, Some(2), "two")),
            finished(user(4, Some(1), "second, revised")),
            finished(assistant(5, Some(4), "three")),
        ] {
            view.apply(&payload);
        }
        view
    }

    #[test]
    fn the_tree_derives_the_path_the_leaves_and_the_editable_messages() {
        let view = forked();
        assert_eq!(view.leaves(), [ItemId(3), ItemId(5)]);
        assert_eq!(view.head(), Some(ItemId(5)));
        assert_eq!(
            view.path().iter().map(|item| item.id).collect::<Vec<_>>(),
            [ItemId(0), ItemId(1), ItemId(4), ItemId(5)]
        );
        assert_eq!(view.user_messages(), [ItemId(0), ItemId(4)]);
    }

    #[test]
    fn switching_branches_reprints_the_branch_it_selects() {
        let mut view = forked();
        drain(&mut view);
        view.set_head(Some(ItemId(3)));
        assert_eq!(
            texts(&mut view),
            [
                "psi: branch of 4 items",
                "",
                "> first",
                "",
                "one",
                "",
                "> second",
                "",
                "two",
            ]
        );
        assert_eq!(view.user_messages(), [ItemId(0), ItemId(2)]);
    }

    #[test]
    fn the_branch_list_marks_the_selected_message() {
        let view = forked();
        let lines = view.branch_lines(1, 1, 2);
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.tone, line.text.as_str()))
                .collect::<Vec<_>>(),
            [
                (Tone::Notice, "branch 2/2 — edit a past message to fork"),
                (Tone::User, "  first"),
                (Tone::Selected, "> second, revised"),
            ]
        );
    }

    #[test]
    fn a_loaded_session_prints_the_path_it_resumes() {
        let mut view = View::new();
        view.apply(&EventPayload::SessionLoaded {
            snapshot: SessionSnapshot {
                meta: SessionMeta {
                    id: SessionId("s1".to_string()),
                    created_at_ms: 0,
                },
                items: vec![user(0, None, "first"), assistant(1, Some(0), "one")],
                head: Some(ItemId(1)),
            },
        });
        assert_eq!(
            texts(&mut view),
            ["psi: continuing s1 (2 items)", "", "> first", "", "one"]
        );
    }
}

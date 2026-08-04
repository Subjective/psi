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
use psi_core::protocol::EventPayload;
use psi_core::session::{SessionId, SessionSnapshot};

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
    /// A line inside a fenced code block in assistant text.
    Code,
    /// Psi speaking about itself: branch moves, cancellation, prompts.
    Notice,
    Error,
    /// The selected row of a picker; the drawing layer also scrolls the list
    /// to keep it visible.
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

/// A rendered tool call waiting for its result, which is what tells it how
/// long it took.
struct PendingCall {
    call_id: String,
    lines: Vec<DisplayLine>,
}

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
    /// Tool calls whose lines are held until their results arrive, oldest
    /// first. A response can make several calls before any of them runs, so
    /// the call id is what pairs a call with the result that times it.
    pending_calls: Vec<PendingCall>,
    /// Lines that have become final, waiting to be handed to the terminal.
    scrollback: Vec<DisplayLine>,
    /// Whether anything has been emitted yet, so blocks are separated by a
    /// blank line without one opening the session.
    emitted: bool,
    /// Whether the assistant text being rendered is inside a ``` fence. It
    /// spans the streaming flush and the item's final render, which are two
    /// halves of one message.
    fenced: bool,
    running: bool,
}

impl View {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            index: HashMap::new(),
            head: None,
            open: None,
            pending_calls: Vec::new(),
            scrollback: Vec::new(),
            emitted: false,
            fenced: false,
            running: false,
        }
    }

    pub fn running(&self) -> bool {
        self.running
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

    /// Psi answering the user directly rather than reporting an event: what a
    /// slash command has to say for itself.
    pub fn notice(&mut self, text: impl Into<String>) {
        self.push(DisplayLine::new(Tone::Notice, text));
    }

    /// The lines that are still changing: the tail of whatever is streaming.
    /// Redrawn every frame, never written to the scrollback until it is final.
    ///
    /// A tail that starts a block is shown with the blank line its block will
    /// carry once it flushes, so nothing shifts when it does; a tail mid-block
    /// stays flush against the lines of it already above.
    pub fn live(&self) -> Vec<DisplayLine> {
        let lines = self.tail();
        let mid_block = matches!(&self.open, Some(open) if open.flushed > 0);
        if lines.is_empty() || !self.emitted || mid_block {
            return lines;
        }
        let mut led = vec![DisplayLine::blank()];
        led.extend(lines);
        led
    }

    fn tail(&self) -> Vec<DisplayLine> {
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
            // Calls run in the order they were made, so the oldest one still
            // held is the one running. It is held back until it can carry its
            // duration, which makes it what the viewport shows meanwhile.
            ItemKind::ToolResult => match self.pending_calls.first() {
                Some(pending) => pending.lines.clone(),
                None => vec![DisplayLine::new(Tone::Notice, "… working")],
            },
            ItemKind::UserMessage => Vec::new(),
        }
    }

    /// Every user message in the tree, depth-first in id order, each with how
    /// many user messages sit above it. The branch picker shows the whole
    /// tree, abandoned futures included, indented by that depth.
    pub fn message_tree(&self) -> Vec<(ItemId, usize)> {
        // Each user message hangs off its nearest user-message ancestor.
        let mut children: HashMap<Option<ItemId>, Vec<ItemId>> = HashMap::new();
        for item in &self.items {
            if item.payload.kind() != ItemKind::UserMessage {
                continue;
            }
            let mut cursor = item.parent_id;
            let parent = loop {
                match cursor.and_then(|id| self.item(id)) {
                    Some(above) if above.payload.kind() == ItemKind::UserMessage => {
                        break Some(above.id);
                    }
                    Some(above) => cursor = above.parent_id,
                    None => break None,
                }
            };
            children.entry(parent).or_default().push(item.id);
        }
        let mut tree = Vec::new();
        let mut stack: Vec<(ItemId, usize)> = children
            .get(&None)
            .map(|roots| roots.iter().rev().map(|id| (*id, 0)).collect())
            .unwrap_or_default();
        while let Some((id, depth)) = stack.pop() {
            tree.push((id, depth));
            if let Some(kids) = children.get(&Some(id)) {
                stack.extend(kids.iter().rev().map(|kid| (*kid, depth + 1)));
            }
        }
        tree
    }

    /// The tip of the branch below an item: the newest child, followed down
    /// until there is none. Jumping back to a branch lands here.
    pub fn tip_of(&self, id: ItemId) -> ItemId {
        let mut tip = id;
        loop {
            let next = self
                .items
                .iter()
                .filter(|item| item.parent_id == Some(tip))
                .map(|item| item.id)
                .max();
            match next {
                Some(child) => tip = child,
                None => return tip,
            }
        }
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
            EventPayload::TurnFinished { status, error, .. } => {
                self.running = false;
                self.open = None;
                // A turn that ended before a call's result still owes the call
                // its line.
                self.flush_pending();
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
                // Turns separate when they end, not when the next prompt
                // arrives, so the blank sits above the idle composer instead of
                // pushing the next echo off its rows.
                self.separate();
            }
            EventPayload::SessionCreated { meta } => self.reset(&meta.id),
            EventPayload::SessionsListed { .. } => {}
        }
    }

    /// Starts on a fresh session: the tree the client mirrors is another
    /// session's now, and the id is said out loud because it is the only place
    /// the TUI names it.
    fn reset(&mut self, id: &SessionId) {
        self.items.clear();
        self.index.clear();
        self.head = None;
        self.open = None;
        self.pending_calls.clear();
        self.running = false;
        self.separate();
        self.notice(format!("psi: new session {}", id.0));
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
            // The live flow separates turns as they finish; a reprint
            // recreates that spacing before each prompt.
            if matches!(item.payload, ItemPayload::UserMessage { .. }) {
                self.separate();
            }
            self.render(item, 0);
        }
        self.flush_pending();
        // The trailing blank a finished turn leaves, so the composer keeps its
        // distance from the reprinted branch too.
        self.separate();
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
            if matches!(item.payload, ItemPayload::UserMessage { .. }) {
                self.separate();
            }
            self.render(&item, 0);
        }
        self.flush_pending();
        self.separate();
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
            self.fenced = false;
        }
        for line in lines {
            self.push_text(&line, tone);
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
                // The composer's own gutters, and no separator: the echo lands
                // exactly on the rows where the prompt was typed. The blank
                // above it was already pushed when the previous turn finished.
                for (number, line) in text.lines().enumerate() {
                    let gutter = if number == 0 { "> " } else { "  " };
                    self.push(DisplayLine::new(Tone::User, format!("{gutter}{line}")));
                }
            }
            ItemPayload::AssistantMessage { text } => {
                self.text_block(text, skip, Tone::Assistant);
            }
            ItemPayload::Reasoning { text, .. } => {
                self.text_block(text, skip, Tone::Reasoning);
            }
            ItemPayload::ToolCall {
                tool,
                call_id,
                arguments,
                ..
            } => {
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
                self.pending_calls.push(PendingCall {
                    call_id: call_id.clone(),
                    lines,
                });
            }
            ItemPayload::ToolResult {
                call_id,
                content,
                truncated,
                duration_ms,
            } => {
                self.flush_call(call_id, *duration_ms);
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
            self.annotate(item, DisplayLine::new(Tone::Error, format!("  {error}")));
        }
        if item.status == CompletionStatus::Cancelled {
            self.annotate(item, DisplayLine::new(Tone::Notice, "  [cancelled]"));
        }
    }

    /// A line about the item just rendered. A tool call's own lines are still
    /// held, so its error joins them rather than printing above them.
    fn annotate(&mut self, item: &Item, line: DisplayLine) {
        match (&item.payload, self.pending_calls.last_mut()) {
            (ItemPayload::ToolCall { call_id, .. }, Some(pending))
                if pending.call_id == *call_id =>
            {
                pending.lines.push(line)
            }
            _ => self.push(line),
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
            self.fenced = false;
        }
        for line in lines {
            self.push_text(line, tone);
        }
    }

    /// One line of a text block. A ``` line opens or closes a fenced block and
    /// is not itself printed; what it fences is indented and dimmed. This is
    /// the whole of Psi's markdown: a fence is the one piece of it a terminal
    /// reader needs, because code inside prose is what the eye is hunting for.
    fn push_text(&mut self, line: &str, tone: Tone) {
        if tone != Tone::Assistant {
            self.push(DisplayLine::new(tone, line));
            return;
        }
        if line.trim_start().starts_with("```") {
            self.fenced = !self.fenced;
            return;
        }
        match self.fenced {
            true => self.push(DisplayLine::new(Tone::Code, format!("  {line}"))),
            false => self.push(DisplayLine::new(tone, line)),
        }
    }

    /// Pushes the held call this result belongs to, with the time it took
    /// written onto the call itself. Terminal scrollback only grows, so a
    /// duration reaches the line that names its call only by waiting for it.
    fn flush_call(&mut self, call_id: &str, duration_ms: u64) {
        let Some(at) = self
            .pending_calls
            .iter()
            .position(|pending| pending.call_id == call_id)
        else {
            return;
        };
        let mut pending = self.pending_calls.remove(at);
        if let Some(first) = pending.lines.first_mut() {
            first
                .text
                .push_str(&format!(" · {}", duration(duration_ms)));
        }
        self.separate();
        self.push_capped(pending.lines);
    }

    /// Pushes every call still waiting on a result, timeless: nothing is going
    /// to arrive to time them now.
    fn flush_pending(&mut self) {
        for pending in std::mem::take(&mut self.pending_calls) {
            self.separate();
            self.push_capped(pending.lines);
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

/// A selection list: one row per entry with the selected one marked. Branch
/// mode, `/resume` and the `@` picker are the same widget over different rows;
/// their keys live in `/help`, not in a header.
pub fn picker(rows: Vec<String>, selected: usize) -> Vec<DisplayLine> {
    rows.into_iter()
        .enumerate()
        .map(|(position, row)| {
            let (tone, marker) = if position == selected {
                (Tone::Selected, ">")
            } else {
                (Tone::User, " ")
            };
            DisplayLine::new(tone, format!("{marker} {row}"))
        })
        .collect()
}

/// How long ago something happened, coarsely. A session list is chosen from by
/// recency, so the row needs an order of magnitude, not a timestamp.
pub fn age(created_at_ms: u64, now_ms: u64) -> String {
    match now_ms.saturating_sub(created_at_ms) / 1000 {
        seconds if seconds < 60 => "just now".to_string(),
        seconds if seconds < 3600 => format!("{}m ago", seconds / 60),
        seconds if seconds < 86_400 => format!("{}h ago", seconds / 3600),
        seconds => format!("{}d ago", seconds / 86_400),
    }
}

/// How long a tool call took. Sub-second calls are the common case; a call
/// that ran for minutes reads as a number a person can hold.
fn duration(ms: u64) -> String {
    match ms {
        ms if ms < 1000 => format!("{ms}ms"),
        ms => format!("{:.1}s", ms as f64 / 1000.0),
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

    fn call(
        id: u64,
        parent: Option<u64>,
        tool: &str,
        arguments: serde_json::Value,
        call_id: &str,
    ) -> Item {
        item(
            id,
            parent,
            ItemPayload::ToolCall {
                tool: tool.to_string(),
                call_id: call_id.to_string(),
                arguments,
                cwd: PathBuf::from("/workspace"),
                revision: WorkspaceRevision(0),
            },
        )
    }

    fn result(
        id: u64,
        parent: Option<u64>,
        content: &str,
        duration_ms: u64,
        call_id: &str,
    ) -> Item {
        item(
            id,
            parent,
            ItemPayload::ToolResult {
                call_id: call_id.to_string(),
                content: content.to_string(),
                duration_ms,
                truncated: false,
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
                (Tone::Notice, String::new()),
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
    fn a_fenced_block_is_indented_and_its_fences_are_not_printed() {
        let mut view = View::new();
        let text = "try this:\n```sh\necho 42\n```\nthat is all.";
        view.apply(&started(0, ItemKind::AssistantMessage));
        // The fence spans the streaming flush and the final render, so it is
        // fed in two pieces on purpose.
        view.apply(&delta(0, "try this:\n```sh\necho 42\n"));
        view.apply(&finished(assistant(0, None, text)));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Assistant, "try this:".to_string()),
                (Tone::Code, "  echo 42".to_string()),
                (Tone::Assistant, "that is all.".to_string()),
            ]
        );
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
            "call-1",
        )));
        view.apply(&started(1, ItemKind::ToolResult));
        // The call is held back until it can carry its duration, so it is what
        // the viewport shows while the tool runs.
        assert_eq!(
            view.live(),
            [DisplayLine::new(
                Tone::Tool,
                "• read_file {\"path\":\"src/lib.sh\"}"
            )]
        );
        view.apply(&finished(result(
            1,
            Some(0),
            "answer() {\n  echo 41\n}",
            3,
            "call-1",
        )));

        assert_eq!(
            drain(&mut view),
            [
                (
                    Tone::Tool,
                    "• read_file {\"path\":\"src/lib.sh\"} · 3ms".to_string()
                ),
                (Tone::ToolOutput, "  answer() {".to_string()),
                (Tone::ToolOutput, "    echo 41".to_string()),
                (Tone::ToolOutput, "  }".to_string()),
            ]
        );
    }

    /// A response can make several calls before any of them runs, so the
    /// calls arrive together and the results follow one at a time. Each call
    /// waits for the result that carries its own id.
    #[test]
    fn parallel_calls_pair_with_their_own_results() {
        let mut view = View::new();
        view.apply(&finished(call(
            0,
            None,
            "read_file",
            json!({ "path": "a" }),
            "call-a",
        )));
        view.apply(&finished(call(
            1,
            Some(0),
            "read_file",
            json!({ "path": "b" }),
            "call-b",
        )));
        view.apply(&finished(result(2, Some(1), "contents of a", 5, "call-a")));
        view.apply(&finished(result(3, Some(2), "contents of b", 9, "call-b")));
        assert_eq!(
            texts(&mut view),
            [
                "• read_file {\"path\":\"a\"} · 5ms",
                "  contents of a",
                "",
                "• read_file {\"path\":\"b\"} · 9ms",
                "  contents of b",
            ]
        );
    }

    #[test]
    fn a_failed_tool_result_is_toned_as_an_error() {
        let mut view = View::new();
        let mut failed = result(
            0,
            None,
            "read_file ../secret: escapes the workspace root",
            1,
            "call-1",
        );
        failed.status = CompletionStatus::Failed;
        failed.error = Some("read_file ../secret: escapes the workspace root".to_string());
        view.apply(&finished(failed));
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
            "call-1",
        )));
        // A call that ran for more than a second reads in seconds.
        view.apply(&finished(result(
            1,
            Some(0),
            "updated src/lib.sh",
            1240,
            "call-1",
        )));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch src/lib.sh · 1.2s".to_string()),
                (Tone::DiffRemoved, "  -  echo 41".to_string()),
                (Tone::DiffAdded, "  +  echo 42".to_string()),
                (Tone::ToolOutput, "  updated src/lib.sh".to_string()),
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
            "call-1",
        )));
        view.apply(&finished(result(
            1,
            Some(0),
            "created new.txt",
            4,
            "call-1",
        )));
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch new.txt · 4ms".to_string()),
                (Tone::DiffAdded, "  +one".to_string()),
                (Tone::DiffAdded, "  +two".to_string()),
                (Tone::ToolOutput, "  created new.txt".to_string()),
            ]
        );
    }

    #[test]
    fn a_call_whose_arguments_never_finished_still_renders() {
        let mut view = View::new();
        let mut broken = call(0, None, "apply_patch", serde_json::Value::Null, "call-1");
        broken.status = CompletionStatus::Failed;
        broken.error = Some("the response completed before the arguments did".to_string());
        view.apply(&finished(broken));
        // The call never runs, so no result ever carries it out; the turn's end
        // is what flushes it.
        view.apply(&EventPayload::TurnFinished {
            turn_id: TurnId(0),
            status: CompletionStatus::Completed,
            error: None,
            usage: None,
        });
        assert_eq!(
            drain(&mut view),
            [
                (Tone::Tool, "• apply_patch ".to_string()),
                (
                    Tone::Error,
                    "  the response completed before the arguments did".to_string()
                ),
                (Tone::Notice, String::new()),
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
        view.apply(&finished(result(0, None, &content, 1, "call-1")));
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
                (Tone::Notice, String::new()),
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
            [
                (Tone::Error, "psi: connection reset".to_string()),
                (Tone::Notice, String::new()),
            ]
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
                "",
            ]
        );
        assert_eq!(view.user_messages(), [ItemId(0), ItemId(2)]);
    }

    #[test]
    fn a_tail_opening_a_block_carries_its_coming_separator() {
        let mut view = View::new();
        view.apply(&EventPayload::TurnStarted { turn_id: TurnId(0) });
        view.apply(&finished(user(0, None, "hi")));
        // The prompt is out; the placeholder and a streamed first line show
        // behind the blank their block will keep once it flushes.
        assert_eq!(
            view.live()[0],
            DisplayLine::new(Tone::Notice, String::new())
        );
        view.apply(&started(1, ItemKind::AssistantMessage));
        view.apply(&delta(1, "Hey"));
        assert_eq!(
            view.live()
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["", "Hey"]
        );
        // A tail mid-block sits flush against its flushed lines.
        view.apply(&delta(1, " there\nand"));
        drain(&mut view);
        assert_eq!(
            view.live()
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["and"]
        );
    }

    #[test]
    fn the_message_tree_indents_forks_and_finds_their_tips() {
        let view = forked();
        // Both "second"s hang off "first": the abandoned one stays visible.
        assert_eq!(
            view.message_tree(),
            [(ItemId(0), 0), (ItemId(2), 1), (ItemId(4), 1)]
        );
        // Jumping back to the abandoned branch lands on its tip.
        assert_eq!(view.tip_of(ItemId(2)), ItemId(3));
        assert_eq!(view.tip_of(ItemId(0)), ItemId(5));
    }

    #[test]
    fn a_sessions_age_is_coarse() {
        let hour = 3_600_000;
        assert_eq!(age(hour, hour + 1_000), "just now");
        assert_eq!(age(hour, hour + 90_000), "1m ago");
        assert_eq!(age(hour, hour + 2 * hour), "2h ago");
        assert_eq!(age(hour, hour + 72 * hour), "3d ago");
        // A clock that went backwards is not an error worth having.
        assert_eq!(age(hour, 0), "just now");
    }

    #[test]
    fn a_created_session_clears_the_tree_and_names_itself() {
        let mut view = forked();
        drain(&mut view);
        view.apply(&EventPayload::SessionCreated {
            meta: SessionMeta {
                id: SessionId("s2".to_string()),
                created_at_ms: 0,
            },
        });
        assert_eq!(texts(&mut view), ["", "psi: new session s2"]);
        assert!(view.leaves().is_empty());
        assert_eq!(view.head(), None);
        assert!(view.user_messages().is_empty());
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
            ["psi: continuing s1 (2 items)", "", "> first", "", "one", ""]
        );
    }
}

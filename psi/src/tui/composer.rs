//! The composer: Psi's own modal editor over a `ropey` rope (docs/design.md,
//! "Terminal-native TUI and composer").
//!
//! Normal-mode keys resolve through Vim's grammar — `[count] operator [count]
//! motion` — so the MVP's small surface and the Vim features that follow it are
//! one code path: a new motion or operator is a table entry rather than another
//! branch. Text objects are the grammar's other right-hand operand and enter as
//! a second table when they ship.
//!
//! Positions are character offsets into the rope. In normal mode the cursor
//! rests *on* a character, so it stops one short of the line break; in insert
//! mode it may sit past the last character of a line.
//!
//! Insert mode also carries the readline bindings a terminal prompt is expected
//! to have — Ctrl-A/E, Ctrl-U/K/W. They are not Vim, and they are not a second
//! grammar either: each one is a motion the grammar already computes, applied
//! to the line the cursor is on.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ropey::Rope;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
}

/// What a key did, for the caller that owns submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Handled,
    /// Enter was pressed; the caller decides whether to submit the buffer.
    Submit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Motion {
    Left,
    Right,
    Up,
    Down,
    WordForward,
    WordBackward,
    WordEnd,
    LineStart,
    LineEnd,
}

/// How much of the way to its target a motion takes with it when an operator is
/// pending. Motions used on their own ignore this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Up to but not including the target character.
    Exclusive,
    /// Through the target character.
    Inclusive,
    /// Whole lines, the cursor's line through the target's.
    Linewise,
}

/// `[count] motion`, and the right-hand operand of `[count] operator`.
const MOTIONS: &[(char, Motion, Scope)] = &[
    ('h', Motion::Left, Scope::Exclusive),
    ('l', Motion::Right, Scope::Exclusive),
    ('j', Motion::Down, Scope::Linewise),
    ('k', Motion::Up, Scope::Linewise),
    ('w', Motion::WordForward, Scope::Exclusive),
    ('b', Motion::WordBackward, Scope::Exclusive),
    ('e', Motion::WordEnd, Scope::Inclusive),
    ('0', Motion::LineStart, Scope::Exclusive),
    ('$', Motion::LineEnd, Scope::Inclusive),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Delete,
    Change,
}

/// `[count] operator`. Doubling the key (`dd`, `cc`) applies it linewise.
const OPERATORS: &[(char, Operator)] = &[('d', Operator::Delete), ('c', Operator::Change)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    InsertBefore,
    InsertAfter,
    InsertAtLineStart,
    InsertAtLineEnd,
    OpenBelow,
    OpenAbove,
    DeleteChar,
}

/// What an insert-mode control key does with a motion the grammar already
/// computes: go there, or take everything between here and there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readline {
    Move(Motion),
    Delete(Motion),
}

/// `Ctrl-<key>` in insert mode. Ctrl-W deletes by the same word rule `b` moves
/// by, so a prompt and a Vim motion never disagree about where a word starts.
const READLINE: &[(char, Readline)] = &[
    ('a', Readline::Move(Motion::LineStart)),
    ('e', Readline::Move(Motion::LineEnd)),
    ('u', Readline::Delete(Motion::LineStart)),
    ('k', Readline::Delete(Motion::LineEnd)),
    ('w', Readline::Delete(Motion::WordBackward)),
];

/// Normal-mode keys that act on their own, taking only a count.
const ACTIONS: &[(char, Action)] = &[
    ('i', Action::InsertBefore),
    ('a', Action::InsertAfter),
    ('I', Action::InsertAtLineStart),
    ('A', Action::InsertAtLineEnd),
    ('o', Action::OpenBelow),
    ('O', Action::OpenAbove),
    ('x', Action::DeleteChar),
];

/// The half-typed command: everything the grammar has accepted but not yet
/// resolved into an edit.
#[derive(Debug, Default)]
struct Pending {
    /// Digits typed since the last resolved command — the `[count]` on either
    /// side of an operator.
    count: Option<usize>,
    operator: Option<Operator>,
    /// The count that preceded the operator, multiplied with the motion's.
    operator_count: usize,
}

/// The character classes Vim's word motions step between: `w` crosses from one
/// class to the next, skipping blanks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Blank,
    Word,
    Punct,
}

fn class(c: char) -> Class {
    if c.is_whitespace() {
        Class::Blank
    } else if c.is_alphanumeric() || c == '_' {
        Class::Word
    } else {
        Class::Punct
    }
}

pub struct Composer {
    text: Rope,
    cursor: usize,
    mode: Mode,
    /// The column `j` and `k` aim for, so crossing a short line and coming back
    /// returns to the column the user left.
    column_intent: usize,
    pending: Pending,
    /// Submitted prompts, oldest first.
    history: Vec<String>,
    /// Position in `history` while recalling; `None` is the live buffer.
    recall: Option<usize>,
    /// The live buffer set aside while recalling, restored when recall walks
    /// back past the newest entry.
    stashed: String,
}

impl Composer {
    /// Starts in insert mode: the first thing a new session wants is a typed
    /// prompt, not a motion. `history` is the prompts persisted by earlier
    /// runs, oldest first; recall walks them and this run's as one list.
    pub fn new(history: Vec<String>) -> Self {
        Self {
            text: Rope::new(),
            cursor: 0,
            mode: Mode::Insert,
            column_intent: 0,
            pending: Pending::default(),
            history,
            recall: None,
            stashed: String::new(),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn text(&self) -> String {
        self.text.to_string()
    }

    /// The buffer as display lines, line breaks removed.
    pub fn lines(&self) -> Vec<String> {
        (0..self.text.len_lines())
            .map(|line| {
                let (start, end) = self.line_bounds(line);
                self.text.slice(start..end).to_string()
            })
            .collect()
    }

    /// The cursor as `(line, column)`, both zero-based.
    pub fn cursor(&self) -> (usize, usize) {
        let line = self.text.char_to_line(self.cursor);
        (line, self.cursor - self.text.line_to_char(line))
    }

    /// The cursor as a character offset, which is how the file picker
    /// remembers where its `@` was.
    pub fn offset(&self) -> usize {
        self.cursor
    }

    /// The characters between `from` and the cursor. `None` once the cursor has
    /// moved back past `from`, which is how the file picker learns its `@` was
    /// deleted or left behind.
    pub fn text_after(&self, from: usize) -> Option<String> {
        if self.cursor < from || from > self.text.len_chars() {
            return None;
        }
        Some(self.text.slice(from..self.cursor).to_string())
    }

    /// Replaces the characters between `from` and the cursor — the file picker
    /// swapping its `@query` for the path it selected.
    pub fn replace_range(&mut self, from: usize, text: &str) {
        let from = from.min(self.cursor);
        self.text.remove(from..self.cursor);
        self.cursor = from;
        self.insert_str(text);
    }

    pub fn is_blank(&self) -> bool {
        self.text().trim().is_empty()
    }

    /// Loads text for editing — a past user message being forked. Insert mode
    /// with the cursor at the end is where an edit starts.
    pub fn load(&mut self, text: &str) {
        self.text = Rope::from_str(text);
        self.cursor = self.text.len_chars();
        self.mode = Mode::Insert;
        self.column_intent = self.cursor().1;
        self.pending = Pending::default();
        self.recall = None;
        self.stashed.clear();
    }

    /// Takes the buffer to submit it, recording it in the history a later
    /// recall walks.
    pub fn take(&mut self) -> String {
        let text = self.text();
        if self.history.last().map(String::as_str) != Some(text.as_str()) {
            self.history.push(text.clone());
        }
        self.text = Rope::new();
        self.cursor = 0;
        self.mode = Mode::Insert;
        self.column_intent = 0;
        self.pending = Pending::default();
        self.recall = None;
        self.stashed.clear();
        text
    }

    /// Pasted text is inserted literally, so a multiline paste cannot submit
    /// itself at its first line break. Terminals send those breaks as carriage
    /// returns; the buffer keeps one kind of line break so a line is a line.
    pub fn paste(&mut self, text: &str) {
        self.mode = Mode::Insert;
        self.insert_str(&text.replace("\r\n", "\n").replace('\r', "\n"));
    }

    /// Walks back through submitted prompts. The live buffer is set aside on
    /// the first step so walking forward again returns to it.
    pub fn recall_previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.recall {
            None => {
                self.stashed = self.text.to_string();
                self.history.len() - 1
            }
            Some(0) => return,
            Some(index) => index - 1,
        };
        self.recall = Some(next);
        let entry = self.history[next].clone();
        self.replace(&entry);
    }

    pub fn recall_next(&mut self) {
        let Some(index) = self.recall else { return };
        if index + 1 < self.history.len() {
            self.recall = Some(index + 1);
            let entry = self.history[index + 1].clone();
            self.replace(&entry);
        } else {
            self.recall = None;
            let stashed = std::mem::take(&mut self.stashed);
            self.replace(&stashed);
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Outcome {
        match self.mode {
            Mode::Insert => self.insert_key(key),
            Mode::Normal => self.normal_key(key),
        }
    }

    fn insert_key(&mut self, key: KeyEvent) -> Outcome {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            // Enter submits, so a newline needs a key of its own. Terminals
            // send Ctrl-J as the newline byte itself and Alt-Enter is the
            // binding most editors use; both are accepted.
            KeyCode::Enter if alt => self.insert_str("\n"),
            KeyCode::Char('j') if control => self.insert_str("\n"),
            KeyCode::Enter => return Outcome::Submit,
            KeyCode::Char(c) if control => {
                if let Some(action) = READLINE
                    .iter()
                    .find(|(key, _)| *key == c)
                    .map(|(_, action)| *action)
                {
                    self.readline(action);
                }
            }
            KeyCode::Char(c) if !alt => self.insert_str(&c.to_string()),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Esc => {
                self.mode = Mode::Normal;
                self.cursor = self.cursor.saturating_sub(1).max(self.line_bounds_at().0);
                self.clamp();
                self.column_intent = self.cursor().1;
            }
            KeyCode::Left => self.move_by(Motion::Left, 1),
            KeyCode::Right => self.move_by(Motion::Right, 1),
            KeyCode::Up => self.move_by(Motion::Up, 1),
            KeyCode::Down => self.move_by(Motion::Down, 1),
            _ => {}
        }
        Outcome::Handled
    }

    fn normal_key(&mut self, key: KeyEvent) -> Outcome {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter if !alt => return Outcome::Submit,
            KeyCode::Char(c) if !control && !alt => self.normal_char(c),
            KeyCode::Esc => self.pending = Pending::default(),
            KeyCode::Left => self.move_by(Motion::Left, 1),
            KeyCode::Right => self.move_by(Motion::Right, 1),
            KeyCode::Up => self.move_by(Motion::Up, 1),
            KeyCode::Down => self.move_by(Motion::Down, 1),
            _ => {}
        }
        Outcome::Handled
    }

    /// One step of the grammar: a digit extends the count, an operator waits
    /// for its operand, a motion or action resolves the command.
    fn normal_char(&mut self, c: char) {
        // A leading zero is the line-start motion; after a digit it is part of
        // the count.
        if c.is_ascii_digit() && (c != '0' || self.pending.count.is_some()) {
            let digit = c as usize - '0' as usize;
            self.pending.count = Some(self.pending.count.unwrap_or(0) * 10 + digit);
            return;
        }
        if let Some(operator) = OPERATORS
            .iter()
            .find(|(key, _)| *key == c)
            .map(|(_, op)| *op)
        {
            match self.pending.operator {
                // A doubled operator (`dd`, `cc`) is its linewise form.
                Some(pending) if pending == operator => {
                    let count = self.pending.operator_count * self.pending.count.unwrap_or(1);
                    self.pending = Pending::default();
                    let first = self.text.char_to_line(self.cursor);
                    match operator {
                        Operator::Delete => self.delete_lines(first, count),
                        Operator::Change => self.change_lines(first, count),
                    }
                }
                Some(_) => self.pending = Pending::default(),
                None => {
                    self.pending.operator = Some(operator);
                    self.pending.operator_count = self.pending.count.take().unwrap_or(1);
                }
            }
            return;
        }
        if let Some((motion, scope)) = MOTIONS
            .iter()
            .find(|(key, _, _)| *key == c)
            .map(|(_, motion, scope)| (*motion, *scope))
        {
            let count = self.pending.count.take().unwrap_or(1);
            match self.pending.operator.take() {
                Some(operator) => {
                    let count = self.pending.operator_count * count;
                    self.pending = Pending::default();
                    self.operate(operator, motion, scope, count);
                }
                None => {
                    self.pending = Pending::default();
                    self.move_by(motion, count);
                }
            }
            return;
        }
        if let Some(action) = ACTIONS.iter().find(|(key, _)| *key == c).map(|(_, a)| *a) {
            let count = self.pending.count.take().unwrap_or(1);
            self.pending = Pending::default();
            self.act(action, count);
            return;
        }
        // A key the grammar does not name abandons what was typed, rather than
        // leaving a count to attach itself to the next one.
        self.pending = Pending::default();
    }

    fn readline(&mut self, action: Readline) {
        match action {
            Readline::Move(motion) => self.cursor = self.readline_target(motion),
            Readline::Delete(motion) => {
                let target = self.readline_target(motion);
                let (start, end) = (self.cursor.min(target), self.cursor.max(target));
                if end > start {
                    self.text.remove(start..end);
                }
                self.cursor = start;
            }
        }
        self.clamp();
        self.column_intent = self.cursor().1;
    }

    /// Where a readline binding aims. Insert mode's end of line is past the
    /// last character, one further than `$` goes.
    fn readline_target(&self, motion: Motion) -> usize {
        match motion {
            Motion::LineEnd => self.line_bounds_at().1,
            other => self.target(other, 1),
        }
    }

    fn act(&mut self, action: Action, count: usize) {
        match action {
            Action::InsertBefore => self.mode = Mode::Insert,
            Action::InsertAfter => {
                let (_, end) = self.line_bounds_at();
                self.mode = Mode::Insert;
                self.cursor = (self.cursor + 1).min(end);
            }
            Action::InsertAtLineStart => {
                let (start, end) = self.line_bounds_at();
                self.mode = Mode::Insert;
                self.cursor = self
                    .text
                    .slice(start..end)
                    .chars()
                    .position(|character| !character.is_whitespace())
                    .map_or(end, |offset| start + offset);
            }
            Action::InsertAtLineEnd => {
                let (_, end) = self.line_bounds_at();
                self.mode = Mode::Insert;
                self.cursor = end;
            }
            Action::OpenBelow => {
                let (_, end) = self.line_bounds_at();
                self.text.insert_char(end, '\n');
                self.cursor = end + 1;
                self.mode = Mode::Insert;
            }
            Action::OpenAbove => {
                let (start, _) = self.line_bounds_at();
                self.text.insert_char(start, '\n');
                self.cursor = start;
                self.mode = Mode::Insert;
            }
            Action::DeleteChar => {
                let (_, end) = self.line_bounds_at();
                let last = (self.cursor + count).min(end);
                if last > self.cursor {
                    self.text.remove(self.cursor..last);
                }
                self.clamp();
            }
        }
        self.column_intent = self.cursor().1;
    }

    fn operate(&mut self, operator: Operator, motion: Motion, scope: Scope, count: usize) {
        // Vim defines `cw` as `ce` when it starts on a nonblank character. It
        // changes the word itself while leaving the following whitespace in
        // place, which is what makes replacement text read naturally. Unlike a
        // plain `e`, its first stop is the end of the word under the cursor
        // even when the cursor already sits there, so `cw` never crosses the
        // following whitespace — or the line break — into the next word.
        let change_word = operator == Operator::Change
            && motion == Motion::WordForward
            && self.cursor < self.text.len_chars()
            && class(self.text.char(self.cursor)) != Class::Blank;
        let (motion, scope) = if change_word {
            (Motion::WordEnd, Scope::Inclusive)
        } else {
            (motion, scope)
        };
        let mut target = if change_word {
            self.change_word_target(count)
        } else {
            self.target(motion, count)
        };
        // Vim's one exception to `w`: under an operator it stops at the end of
        // a line rather than joining that line to the next.
        if motion == Motion::WordForward && target > self.cursor {
            let mut last = target;
            while last > self.cursor && self.text.char(last - 1).is_whitespace() {
                last -= 1;
            }
            if self.text.slice(last..target).chars().any(|c| c == '\n') {
                target = last;
            }
        }
        match scope {
            Scope::Linewise => {
                let from = self.text.char_to_line(self.cursor);
                let to = self.text.char_to_line(target);
                let first = from.min(to);
                let count = to.abs_diff(from) + 1;
                match operator {
                    Operator::Delete => self.delete_lines(first, count),
                    Operator::Change => self.change_lines(first, count),
                }
            }
            Scope::Exclusive | Scope::Inclusive => {
                let (start, mut end) = (self.cursor.min(target), self.cursor.max(target));
                if scope == Scope::Inclusive && end < self.text.len_chars() {
                    // A charwise delete never takes the line break with it.
                    if self.text.char(end) != '\n' {
                        end += 1;
                    }
                }
                if end > start {
                    self.text.remove(start..end);
                }
                // A change enters Insert mode even when its motion covered
                // nothing — `c$` on an empty line — as Vim does.
                if operator == Operator::Change {
                    self.mode = Mode::Insert;
                }
                self.cursor = start;
                self.clamp();
            }
        }
        self.column_intent = self.cursor().1;
    }

    /// Replaces one or more whole lines with a single editable line. When
    /// lines follow the changed range, its newline is inserted explicitly;
    /// changing the tail already leaves an empty final line behind.
    fn change_lines(&mut self, first: usize, count: usize) {
        let last = (first + count).min(self.text.len_lines());
        let followed = last < self.text.len_lines();
        let start = self.text.line_to_char(first);
        let end = self.text.line_to_char(last);
        self.text.remove(start..end);
        if followed {
            self.text.insert_char(start, '\n');
        }
        self.cursor = start;
        self.mode = Mode::Insert;
        self.clamp();
        self.column_intent = 0;
    }

    fn delete_lines(&mut self, first: usize, count: usize) {
        let last = (first + count).min(self.text.len_lines());
        let start = self.text.line_to_char(first);
        let end = self.text.line_to_char(last);
        self.text.remove(start..end);
        // Deleting the last lines of the buffer leaves the cursor on what is
        // now the last line, as Vim does.
        self.cursor = self
            .text
            .line_to_char(first.min(self.text.len_lines().saturating_sub(1)));
        self.clamp();
        self.column_intent = 0;
    }

    fn move_by(&mut self, motion: Motion, count: usize) {
        self.cursor = self.target(motion, count);
        self.clamp();
        if !matches!(motion, Motion::Up | Motion::Down) {
            self.column_intent = self.cursor().1;
        }
    }

    /// Where a motion lands, as a character offset. Operators use the same
    /// answer as plain movement, which is what makes `dw` fall out of `w`.
    fn target(&self, motion: Motion, count: usize) -> usize {
        let len = self.text.len_chars();
        match motion {
            Motion::Left => {
                let (start, _) = self.line_bounds_at();
                self.cursor.saturating_sub(count).max(start)
            }
            Motion::Right => {
                let (_, end) = self.line_bounds_at();
                (self.cursor + count).min(end)
            }
            Motion::Up | Motion::Down => {
                let line = self.text.char_to_line(self.cursor);
                let target = if motion == Motion::Up {
                    line.saturating_sub(count)
                } else {
                    (line + count).min(self.text.len_lines() - 1)
                };
                let (start, end) = self.line_bounds(target);
                (start + self.column_intent).min(end)
            }
            Motion::LineStart => self.line_bounds_at().0,
            Motion::LineEnd => {
                let (start, end) = self.line_bounds_at();
                end.saturating_sub(1).max(start)
            }
            Motion::WordForward => {
                let mut at = self.cursor;
                for _ in 0..count {
                    if at >= len {
                        break;
                    }
                    let from = class(self.text.char(at));
                    if from != Class::Blank {
                        while at < len && class(self.text.char(at)) == from {
                            at += 1;
                        }
                    }
                    while at < len && class(self.text.char(at)) == Class::Blank {
                        at += 1;
                    }
                }
                at
            }
            Motion::WordBackward => {
                let mut at = self.cursor;
                for _ in 0..count {
                    if at == 0 {
                        break;
                    }
                    at -= 1;
                    while at > 0 && class(self.text.char(at)) == Class::Blank {
                        at -= 1;
                    }
                    let from = class(self.text.char(at));
                    while at > 0 && class(self.text.char(at - 1)) == from {
                        at -= 1;
                    }
                }
                at
            }
            Motion::WordEnd => self.word_end(self.cursor, count),
        }
    }

    /// Where `count` repeats of `e` land, starting from `from`.
    fn word_end(&self, from: usize, count: usize) -> usize {
        let len = self.text.len_chars();
        let mut at = from;
        for _ in 0..count {
            if at + 1 >= len {
                break;
            }
            at += 1;
            while at < len && class(self.text.char(at)) == Class::Blank {
                at += 1;
            }
            if at < len {
                let run = class(self.text.char(at));
                while at + 1 < len && class(self.text.char(at + 1)) == run {
                    at += 1;
                }
            }
        }
        at.min(len.saturating_sub(1))
    }

    /// Where `cw` ends: the end of the word under the cursor — the cursor
    /// itself when it already sits there, which Vim counts as the first end —
    /// then a plain `e` for each count beyond the first.
    fn change_word_target(&self, count: usize) -> usize {
        let mut at = self.cursor;
        while at + 1 < self.text.len_chars()
            && class(self.text.char(at + 1)) == class(self.text.char(at))
        {
            at += 1;
        }
        match count {
            0 | 1 => at,
            _ => self.word_end(at, count - 1),
        }
    }

    fn insert_str(&mut self, text: &str) {
        self.text.insert(self.cursor, text);
        self.cursor += text.chars().count();
        self.column_intent = self.cursor().1;
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.text.remove(self.cursor - 1..self.cursor);
        self.cursor -= 1;
        self.column_intent = self.cursor().1;
    }

    fn replace(&mut self, text: &str) {
        self.text = Rope::from_str(text);
        self.cursor = self.text.len_chars();
        self.clamp();
        self.column_intent = self.cursor().1;
    }

    /// Keeps the cursor inside the buffer and, in normal mode, off the line
    /// break.
    fn clamp(&mut self) {
        self.cursor = self.cursor.min(self.text.len_chars());
        let (start, end) = self.line_bounds_at();
        let limit = match self.mode {
            Mode::Insert => end,
            Mode::Normal => end.saturating_sub(1).max(start),
        };
        self.cursor = self.cursor.clamp(start, limit);
    }

    fn line_bounds_at(&self) -> (usize, usize) {
        self.line_bounds(self.text.char_to_line(self.cursor))
    }

    /// First and last character offsets of a line, the line break excluded.
    fn line_bounds(&self, line: usize) -> (usize, usize) {
        let start = self.text.line_to_char(line);
        let slice = self.text.line(line);
        let len = slice.len_chars();
        let breaks = usize::from(len > 0 && slice.char(len - 1) == '\n');
        (start, start + len - breaks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Composer {
        Composer::new(Vec::new())
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    /// Types a run of characters as individual key presses.
    fn typed(composer: &mut Composer, keys: &str) {
        for c in keys.chars() {
            let modifiers = if c.is_uppercase() {
                KeyModifiers::SHIFT
            } else {
                KeyModifiers::NONE
            };
            composer.key(KeyEvent::new(KeyCode::Char(c), modifiers));
        }
    }

    fn escape(composer: &mut Composer) {
        composer.key(key(KeyCode::Esc));
    }

    /// Text, cursor and mode after a run of keys, which is the whole state a
    /// composer test cares about.
    fn state(composer: &Composer) -> (String, (usize, usize), Mode) {
        (composer.text(), composer.cursor(), composer.mode())
    }

    #[test]
    fn insert_mode_types_and_backspaces() {
        let mut composer = fresh();
        typed(&mut composer, "hello");
        composer.key(key(KeyCode::Backspace));
        assert_eq!(state(&composer), ("hell".into(), (0, 4), Mode::Insert));
    }

    #[test]
    fn escape_leaves_insert_mode_on_the_last_character() {
        let mut composer = fresh();
        typed(&mut composer, "hi");
        escape(&mut composer);
        assert_eq!(state(&composer), ("hi".into(), (0, 1), Mode::Normal));
    }

    #[test]
    fn enter_asks_to_submit_from_both_modes() {
        let mut composer = fresh();
        typed(&mut composer, "ask");
        assert_eq!(composer.key(key(KeyCode::Enter)), Outcome::Submit);
        escape(&mut composer);
        assert_eq!(composer.key(key(KeyCode::Enter)), Outcome::Submit);
    }

    #[test]
    fn ctrl_j_and_alt_enter_insert_newlines() {
        let mut composer = fresh();
        typed(&mut composer, "one");
        composer.key(ctrl('j'));
        typed(&mut composer, "two");
        composer.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));
        typed(&mut composer, "three");
        assert_eq!(
            state(&composer),
            ("one\ntwo\nthree".into(), (2, 5), Mode::Insert)
        );
    }

    #[test]
    fn ctrl_a_and_ctrl_e_go_to_the_ends_of_the_cursors_line() {
        let mut composer = fresh();
        composer.paste("first\nsecond");
        composer.key(ctrl('a'));
        assert_eq!(
            state(&composer),
            ("first\nsecond".into(), (1, 0), Mode::Insert)
        );
        composer.key(ctrl('e'));
        // Insert mode's end of line is past the last character, not on it.
        assert_eq!(composer.cursor(), (1, 6));
    }

    #[test]
    fn ctrl_u_and_ctrl_k_delete_to_the_ends_of_the_cursors_line() {
        let mut composer = fresh();
        composer.paste("keep\nthrow away this");
        composer.key(ctrl('a'));
        typed(&mut composer, "abc");
        composer.key(ctrl('u'));
        assert_eq!(
            state(&composer),
            ("keep\nthrow away this".into(), (1, 0), Mode::Insert)
        );

        typed(&mut composer, "throw ");
        composer.key(ctrl('k'));
        assert_eq!(
            state(&composer),
            ("keep\nthrow ".into(), (1, 6), Mode::Insert)
        );
        // Neither takes the line break with it.
        composer.key(ctrl('u'));
        assert_eq!(composer.lines(), ["keep", ""]);
    }

    #[test]
    fn ctrl_w_deletes_back_by_the_word_rule_b_moves_by() {
        let mut composer = fresh();
        composer.paste("alpha beta_two, gamma");
        composer.key(ctrl('w'));
        assert_eq!(composer.text(), "alpha beta_two, ");
        // `b` steps between character classes, so the punctuation is its own
        // word and goes on its own.
        composer.key(ctrl('w'));
        assert_eq!(composer.text(), "alpha beta_two");
        composer.key(ctrl('w'));
        assert_eq!(state(&composer), ("alpha ".into(), (0, 6), Mode::Insert));
        composer.key(ctrl('w'));
        composer.key(ctrl('w'));
        assert_eq!(composer.text(), "");
    }

    #[test]
    fn a_paste_inserts_its_line_breaks_instead_of_submitting() {
        let mut composer = fresh();
        composer.paste("first\nsecond");
        assert_eq!(composer.text(), "first\nsecond");
        assert_eq!(composer.cursor(), (1, 6));

        // A terminal sends pasted line breaks as carriage returns.
        let mut composer = fresh();
        composer.paste("first\rsecond\r\nthird");
        assert_eq!(composer.lines(), ["first", "second", "third"]);
    }

    #[test]
    fn hjkl_moves_and_stops_at_the_buffer_edges() {
        let mut composer = fresh();
        composer.paste("abc\ndefgh");
        escape(&mut composer);
        // Normal mode has no column past the last character, so coming up from
        // a longer line lands on it.
        typed(&mut composer, "kk");
        assert_eq!(composer.cursor(), (0, 2));
        typed(&mut composer, "0");
        assert_eq!(composer.cursor(), (0, 0));
        typed(&mut composer, "hhh");
        assert_eq!(composer.cursor(), (0, 0));
        typed(&mut composer, "llllll");
        assert_eq!(composer.cursor(), (0, 2));
        typed(&mut composer, "jj$");
        assert_eq!(composer.cursor(), (1, 4));
    }

    #[test]
    fn j_and_k_keep_the_column_across_a_short_line() {
        let mut composer = fresh();
        composer.paste("longest line\nab\nanother long one");
        escape(&mut composer);
        typed(&mut composer, "kk$");
        assert_eq!(composer.cursor(), (0, 11));
        typed(&mut composer, "j");
        assert_eq!(composer.cursor(), (1, 1));
        typed(&mut composer, "j");
        assert_eq!(composer.cursor(), (2, 11));
    }

    #[test]
    fn word_motions_step_between_character_classes() {
        let mut composer = fresh();
        composer.paste("alpha beta_two, gamma");
        escape(&mut composer);
        typed(&mut composer, "0");
        typed(&mut composer, "w");
        assert_eq!(composer.cursor(), (0, 6));
        typed(&mut composer, "w");
        assert_eq!(composer.cursor(), (0, 14));
        typed(&mut composer, "e");
        assert_eq!(composer.cursor(), (0, 20));
        typed(&mut composer, "b");
        assert_eq!(composer.cursor(), (0, 16));
        typed(&mut composer, "bb");
        assert_eq!(composer.cursor(), (0, 6));
    }

    #[test]
    fn counts_multiply_across_motions() {
        let mut composer = fresh();
        composer.paste("one two three four five");
        escape(&mut composer);
        typed(&mut composer, "0");
        typed(&mut composer, "3w");
        assert_eq!(composer.cursor(), (0, 14));
        typed(&mut composer, "2b");
        assert_eq!(composer.cursor(), (0, 4));
        typed(&mut composer, "12l");
        assert_eq!(composer.cursor(), (0, 16));
    }

    #[test]
    fn x_deletes_a_count_of_characters_within_the_line() {
        let mut composer = fresh();
        composer.paste("abcdef\ngh");
        escape(&mut composer);
        typed(&mut composer, "kk0");
        typed(&mut composer, "2x");
        assert_eq!(composer.text(), "cdef\ngh");
        typed(&mut composer, "9x");
        assert_eq!(state(&composer), ("\ngh".into(), (0, 0), Mode::Normal));
    }

    #[test]
    fn delete_with_a_motion_uses_the_motions_own_scope() {
        let mut composer = fresh();
        composer.paste("alpha beta gamma");
        escape(&mut composer);
        typed(&mut composer, "0dw");
        assert_eq!(composer.text(), "beta gamma");
        typed(&mut composer, "de");
        assert_eq!(composer.text(), " gamma");
        typed(&mut composer, "d$");
        assert_eq!(composer.text(), "");
    }

    #[test]
    fn change_with_a_motion_deletes_then_enters_insert_mode() {
        let mut composer = fresh();
        composer.paste("alpha beta gamma");
        escape(&mut composer);
        typed(&mut composer, "0cw");
        assert_eq!(
            state(&composer),
            (" beta gamma".into(), (0, 0), Mode::Insert)
        );

        typed(&mut composer, "new");
        assert_eq!(
            state(&composer),
            ("new beta gamma".into(), (0, 3), Mode::Insert)
        );
    }

    /// On a word's last character, `cw` changes only up to the end of that
    /// word — the character itself — never the whitespace or word after it.
    /// The expectations here are what headless Neovim reports.
    #[test]
    fn cw_at_a_word_end_stays_inside_the_word() {
        let mut composer = fresh();
        composer.paste("one two");
        escape(&mut composer);
        typed(&mut composer, "0ecw");
        assert_eq!(state(&composer), ("on two".into(), (0, 2), Mode::Insert));

        // A single-character word is its own end.
        let mut composer = fresh();
        composer.paste("a b");
        escape(&mut composer);
        typed(&mut composer, "0cw");
        assert_eq!(state(&composer), (" b".into(), (0, 0), Mode::Insert));

        // So is a punctuation run's.
        let mut composer = fresh();
        composer.paste("one, two");
        escape(&mut composer);
        typed(&mut composer, "0eecw");
        assert_eq!(state(&composer), ("one two".into(), (0, 3), Mode::Insert));
    }

    #[test]
    fn cw_at_a_line_end_never_joins_the_next_line() {
        let mut composer = fresh();
        composer.paste("one\ntwo");
        escape(&mut composer);
        typed(&mut composer, "k0ecw");
        assert_eq!(state(&composer), ("on\ntwo".into(), (0, 2), Mode::Insert));
    }

    /// `2cw` from a word's last character counts that end as the first of the
    /// two, as Vim does.
    #[test]
    fn a_counted_cw_takes_the_current_end_as_its_first() {
        let mut composer = fresh();
        composer.paste("one two three");
        escape(&mut composer);
        typed(&mut composer, "0e2cw");
        assert_eq!(state(&composer), ("on three".into(), (0, 2), Mode::Insert));
    }

    /// `c$` on an empty line has nothing to remove but still starts the edit,
    /// as Vim's does.
    #[test]
    fn a_change_covering_nothing_still_enters_insert_mode() {
        let mut composer = fresh();
        composer.paste("one\n\ntwo");
        escape(&mut composer);
        typed(&mut composer, "k");
        typed(&mut composer, "c$");
        assert_eq!(
            state(&composer),
            ("one\n\ntwo".into(), (1, 0), Mode::Insert)
        );

        typed(&mut composer, "X");
        assert_eq!(
            state(&composer),
            ("one\nX\ntwo".into(), (1, 1), Mode::Insert)
        );
    }

    #[test]
    fn change_counts_multiply_on_both_sides_of_the_operator() {
        let mut composer = fresh();
        composer.paste("one two three four five");
        escape(&mut composer);
        typed(&mut composer, "02c2w");
        assert_eq!(state(&composer), (" five".into(), (0, 0), Mode::Insert));
    }

    #[test]
    fn counts_on_both_sides_of_an_operator_multiply() {
        let mut composer = fresh();
        composer.paste("one two three four five six");
        escape(&mut composer);
        typed(&mut composer, "0");
        typed(&mut composer, "2d2w");
        assert_eq!(composer.text(), "five six");
    }

    #[test]
    fn dd_deletes_whole_lines_and_a_count_deletes_several() {
        let mut composer = fresh();
        composer.paste("one\ntwo\nthree\nfour");
        escape(&mut composer);
        typed(&mut composer, "kkk");
        typed(&mut composer, "dd");
        assert_eq!(composer.text(), "two\nthree\nfour");
        typed(&mut composer, "2dd");
        assert_eq!(state(&composer), ("four".into(), (0, 0), Mode::Normal));
        typed(&mut composer, "dd");
        assert_eq!(state(&composer), ("".into(), (0, 0), Mode::Normal));
    }

    #[test]
    fn cc_replaces_counted_lines_with_one_editable_line() {
        let mut composer = fresh();
        composer.paste("one\ntwo\nthree\nfour");
        escape(&mut composer);
        typed(&mut composer, "kkk2cc");
        assert_eq!(
            state(&composer),
            ("\nthree\nfour".into(), (0, 0), Mode::Insert)
        );

        typed(&mut composer, "new");
        assert_eq!(
            state(&composer),
            ("new\nthree\nfour".into(), (0, 3), Mode::Insert)
        );
    }

    #[test]
    fn a_linewise_motion_makes_change_linewise() {
        let mut composer = fresh();
        composer.paste("one\ntwo\nthree");
        escape(&mut composer);
        typed(&mut composer, "kkcj");
        assert_eq!(state(&composer), ("\nthree".into(), (0, 0), Mode::Insert));

        let mut composer = fresh();
        composer.paste("one\ntwo");
        escape(&mut composer);
        typed(&mut composer, "cc");
        assert_eq!(state(&composer), ("one\n".into(), (1, 0), Mode::Insert));
    }

    #[test]
    fn a_linewise_motion_makes_its_operator_linewise() {
        let mut composer = fresh();
        composer.paste("one\ntwo\nthree");
        escape(&mut composer);
        typed(&mut composer, "kk");
        // `j` is linewise, so `dj` takes both lines whole rather than the
        // characters between the two cursors.
        typed(&mut composer, "dj");
        assert_eq!(state(&composer), ("three".into(), (0, 0), Mode::Normal));
    }

    #[test]
    fn i_a_o_and_shift_o_enter_insert_mode_where_vim_puts_them() {
        let mut composer = fresh();
        composer.paste("ab");
        escape(&mut composer);
        typed(&mut composer, "i");
        assert_eq!(state(&composer), ("ab".into(), (0, 1), Mode::Insert));
        escape(&mut composer);
        typed(&mut composer, "$a!");
        assert_eq!(state(&composer), ("ab!".into(), (0, 3), Mode::Insert));
        escape(&mut composer);
        typed(&mut composer, "oline");
        assert_eq!(state(&composer), ("ab!\nline".into(), (1, 4), Mode::Insert));
        escape(&mut composer);
        typed(&mut composer, "Otop");
        assert_eq!(
            state(&composer),
            ("ab!\ntop\nline".into(), (1, 3), Mode::Insert)
        );
    }

    #[test]
    fn shift_i_and_shift_a_insert_at_the_lines_text_boundaries() {
        let mut composer = fresh();
        composer.paste("  alpha  ");
        escape(&mut composer);
        typed(&mut composer, "IX");
        assert_eq!(
            state(&composer),
            ("  Xalpha  ".into(), (0, 3), Mode::Insert)
        );

        escape(&mut composer);
        typed(&mut composer, "AY");
        assert_eq!(
            state(&composer),
            ("  Xalpha  Y".into(), (0, 11), Mode::Insert)
        );

        let mut composer = fresh();
        composer.paste("   ");
        escape(&mut composer);
        typed(&mut composer, "IX");
        assert_eq!(state(&composer), ("   X".into(), (0, 4), Mode::Insert));
    }

    #[test]
    fn multiline_editing_survives_a_round_trip_through_normal_mode() {
        let mut composer = fresh();
        typed(&mut composer, "first");
        composer.key(ctrl('j'));
        typed(&mut composer, "second");
        escape(&mut composer);
        typed(&mut composer, "k0");
        assert_eq!(composer.cursor(), (0, 0));
        // `dw` on the last word of a line stops at the line break instead of
        // pulling the next line up, as Vim does.
        typed(&mut composer, "dw");
        assert_eq!(composer.text(), "\nsecond");
        typed(&mut composer, "ihello");
        assert_eq!(
            state(&composer),
            ("hello\nsecond".into(), (0, 5), Mode::Insert)
        );
    }

    #[test]
    fn history_recalls_earlier_prompts_and_returns_to_the_live_buffer() {
        let mut composer = fresh();
        typed(&mut composer, "one");
        assert_eq!(composer.take(), "one");
        typed(&mut composer, "two");
        assert_eq!(composer.take(), "two");
        typed(&mut composer, "draft");

        composer.recall_previous();
        assert_eq!(composer.text(), "two");
        composer.recall_previous();
        assert_eq!(composer.text(), "one");
        composer.recall_previous();
        assert_eq!(composer.text(), "one");
        composer.recall_next();
        assert_eq!(composer.text(), "two");
        composer.recall_next();
        assert_eq!(composer.text(), "draft");
        composer.recall_next();
        assert_eq!(composer.text(), "draft");
    }

    #[test]
    fn the_text_after_a_marker_is_readable_and_replaceable() {
        let mut composer = fresh();
        typed(&mut composer, "look at @com");
        let at = composer.offset() - 3;
        assert_eq!(composer.text_after(at).as_deref(), Some("com"));

        composer.replace_range(at - 1, "psi/src/tui/composer.rs");
        assert_eq!(composer.text(), "look at psi/src/tui/composer.rs");

        // Backspacing past the marker is how the picker learns it is gone.
        let mut composer = fresh();
        typed(&mut composer, "@c");
        let at = composer.offset() - 1;
        composer.key(key(KeyCode::Backspace));
        composer.key(key(KeyCode::Backspace));
        assert_eq!(composer.text_after(at), None);
    }

    #[test]
    fn loading_a_past_message_leaves_it_ready_to_edit() {
        let mut composer = fresh();
        composer.load("make the test pass");
        assert_eq!(
            state(&composer),
            ("make the test pass".into(), (0, 18), Mode::Insert)
        );
    }

    #[test]
    fn an_unknown_key_abandons_a_half_typed_command() {
        let mut composer = fresh();
        composer.paste("one two three");
        escape(&mut composer);
        typed(&mut composer, "0");
        // `2z` is not a command: the count must not carry into the `w`.
        typed(&mut composer, "2zw");
        assert_eq!(composer.cursor(), (0, 4));

        // Two different operators are not a command, and neither one carries
        // forward to the motion that follows them.
        typed(&mut composer, "0dcw");
        assert_eq!(
            state(&composer),
            ("one two three".into(), (0, 4), Mode::Normal)
        );
    }
}

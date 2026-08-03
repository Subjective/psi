//! The drawing layer: `DisplayLine`s and a composer become terminal cells.
//! Everything Ratatui-shaped lives here, so replacing it — the design doc's
//! open question about a custom line-diff renderer — touches no event handling.
//!
//! Rendering is inline: finished lines are inserted *above* a small viewport,
//! which makes them ordinary terminal scrollback the user can scroll, select
//! and copy after Psi exits. The viewport itself holds only what is still
//! changing: the streaming tail, the composer, and one status row.
//!
//! Wrapping is Psi's own because `insert_before` must be told the exact number
//! of rows it is given. Widths are counted in characters, which is right for
//! everything but double-width scripts.

use std::io::{self, Stdout};

use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{cursor, execute};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::{Terminal, TerminalOptions, Viewport};

use super::app::App;
use super::composer::Mode;
use super::view::{DisplayLine, Tone};

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Rows the inline viewport holds. Ratatui fixes an inline viewport's height
/// when the terminal is built, so this is a constant: enough for a couple of
/// lines of streaming text above a short composer and the status row, and few
/// enough that an idle Psi is a prompt rather than a panel.
const VIEWPORT_ROWS: u16 = 6;

/// The composer's gutter: a prompt marker on the first row, alignment on the
/// rest.
const GUTTER: usize = 2;

/// Puts the terminal in the state the viewport needs and installs the hook that
/// takes it back out again if Psi panics.
pub fn enter() -> io::Result<Tui> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic with raw mode still on leaves a terminal that does not echo;
        // restoring first is what makes the message readable.
        let _ = restore();
        previous(info);
    }));
    enable_raw_mode()?;
    // Everything past here has already changed the terminal, so a failure —
    // a terminal that never answers the cursor query, say — puts it back
    // before reporting itself.
    open().inspect_err(|_| {
        let _ = restore();
    })
}

fn open() -> io::Result<Tui> {
    let mut stdout = io::stdout();
    // Without bracketed paste a pasted newline is indistinguishable from Enter,
    // so a multiline paste would submit itself halfway through.
    execute!(stdout, EnableBracketedPaste)?;
    Terminal::with_options(
        CrosstermBackend::new(stdout),
        TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_ROWS),
        },
    )
}

/// Undoes `enter`. Safe to call more than once, and from a panic hook, which is
/// why it takes no terminal.
pub fn restore() -> io::Result<()> {
    let mut stdout = io::stdout();
    execute!(stdout, DisableBracketedPaste, cursor::Show)?;
    disable_raw_mode()
}

/// Clears the viewport and leaves the cursor where it started, so the shell
/// prompt comes back directly under the transcript.
pub fn leave(terminal: &mut Tui) -> io::Result<()> {
    let top = terminal.get_frame().area().y;
    terminal.clear()?;
    terminal.set_cursor_position(Position::new(0, top))?;
    terminal.show_cursor()
}

/// Moves finished lines into the terminal's scrollback, above the viewport.
pub fn scrollback(terminal: &mut Tui, lines: Vec<DisplayLine>) -> io::Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let width = terminal.size()?.width.max(1);
    let rows = wrap_all(&lines, width as usize);
    terminal.insert_before(rows.len() as u16, |buffer| {
        for (offset, (tone, text)) in rows.iter().enumerate() {
            buffer.set_string(
                buffer.area.x,
                buffer.area.y + offset as u16,
                text,
                style(*tone),
            );
        }
    })
}

/// Draws the viewport: the streaming tail, the composer, and the status row.
/// The composer is anchored to the bottom and grows upward, so the prompt sits
/// against the shell it will hand back to.
pub fn frame(terminal: &mut Tui, app: &App) -> io::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        // A viewport too narrow for the gutter has nowhere to put a prompt.
        if area.height < 2 || area.width as usize <= GUTTER {
            return;
        }
        let width = area.width as usize;

        let composer = app.composer();
        let (rows, cursor) = wrap_composer(&composer.lines(), composer.cursor(), width - GUTTER);
        let available = area.height - 1;
        let composer_rows = (rows.len() as u16).clamp(1, available);
        let live_rows = available - composer_rows;

        let live = wrap_all(&app.live(), width);
        let live = window(&live, live_rows as usize);
        for (offset, (tone, text)) in live.iter().enumerate() {
            frame.render_widget(
                text.as_str(),
                Rect::new(area.x, area.y + offset as u16, area.width, 1),
            );
            frame.buffer_mut().set_style(
                Rect::new(area.x, area.y + offset as u16, area.width, 1),
                style(*tone),
            );
        }

        // Scroll the composer so the cursor's row is the last one shown.
        let top = area.y + available - composer_rows;
        let first = cursor.0.saturating_sub(composer_rows as usize - 1);
        for offset in 0..composer_rows as usize {
            let Some(row) = rows.get(first + offset) else {
                break;
            };
            let gutter = if first + offset == 0 { "> " } else { "  " };
            let y = top + offset as u16;
            frame.render_widget(gutter, Rect::new(area.x, y, GUTTER as u16, 1));
            frame.render_widget(
                row.as_str(),
                Rect::new(area.x + GUTTER as u16, y, area.width - GUTTER as u16, 1),
            );
        }
        frame.set_cursor_position(Position::new(
            area.x + (GUTTER + cursor.1) as u16,
            top + (cursor.0 - first) as u16,
        ));

        let status = Rect::new(area.x, area.y + area.height - 1, area.width, 1);
        frame.render_widget(truncate(&app.status(), width).as_str(), status);
        frame
            .buffer_mut()
            .set_style(status, status_style(composer.mode()));
    })?;
    Ok(())
}

fn style(tone: Tone) -> Style {
    match tone {
        Tone::User => Style::default().fg(Color::Cyan),
        Tone::Assistant => Style::default(),
        Tone::Reasoning => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
        Tone::Tool => Style::default().fg(Color::Blue),
        Tone::ToolOutput => Style::default().fg(Color::DarkGray),
        Tone::DiffAdded => Style::default().fg(Color::Green),
        Tone::DiffRemoved => Style::default().fg(Color::Red),
        Tone::Notice => Style::default().fg(Color::DarkGray),
        Tone::Error => Style::default().fg(Color::Red),
        Tone::Selected => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::REVERSED),
    }
}

fn status_style(mode: Mode) -> Style {
    let colour = match mode {
        Mode::Insert => Color::Green,
        Mode::Normal => Color::Blue,
    };
    Style::default().fg(Color::Black).bg(colour)
}

fn wrap_all(lines: &[DisplayLine], width: usize) -> Vec<(Tone, String)> {
    lines
        .iter()
        .flat_map(|line| {
            wrap(&line.text, width)
                .into_iter()
                .map(move |row| (line.tone, row))
        })
        .collect()
}

/// Wraps at the last space that fits, so prose breaks between words and a long
/// unbroken run (a path, a URL) breaks at the edge instead of overflowing.
fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= width {
        return vec![text.to_string()];
    }
    let mut rows = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = start + width;
        if end >= chars.len() {
            rows.push(chars[start..].iter().collect());
            break;
        }
        // The character at `end` counts: a word that ends exactly at the edge
        // breaks cleanly, and the space it broke at is dropped.
        match chars[start..=end].iter().rposition(|c| *c == ' ') {
            Some(space) if space > 0 => {
                rows.push(chars[start..start + space].iter().collect());
                start += space + 1;
            }
            _ => {
                rows.push(chars[start..end].iter().collect());
                start = end;
            }
        }
    }
    rows
}

/// Wraps the composer's lines and follows the cursor through the wrap, so the
/// terminal cursor lands on the character the buffer says it is on.
fn wrap_composer(
    lines: &[String],
    cursor: (usize, usize),
    width: usize,
) -> (Vec<String>, (usize, usize)) {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut position = (0, 0);
    for (number, line) in lines.iter().enumerate() {
        // Hard wrap: an editor's cursor arithmetic has to be exact, and a
        // column that moves with the words is not.
        let chars: Vec<char> = line.chars().collect();
        let first = rows.len();
        for chunk in chars.chunks(width) {
            rows.push(chunk.iter().collect());
        }
        if chars.is_empty() {
            rows.push(String::new());
        }
        if number == cursor.0 {
            position = (first + cursor.1 / width, cursor.1 % width);
            // The insert cursor sits one past the text, so a line exactly
            // filling its rows puts it on a row that does not exist yet.
            if position.0 == rows.len() {
                rows.push(String::new());
            }
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    (rows, position)
}

/// The `height` rows worth showing: the end of a stream, or the selected row of
/// a list that is taller than the space it has.
fn window(rows: &[(Tone, String)], height: usize) -> Vec<(Tone, String)> {
    // A full-height composer leaves the live region no rows at all.
    if height == 0 {
        return Vec::new();
    }
    if rows.len() <= height {
        return rows.to_vec();
    }
    let selected = rows.iter().position(|(tone, _)| *tone == Tone::Selected);
    let last = match selected {
        Some(row) => row.max(height - 1),
        None => rows.len() - 1,
    };
    rows[last + 1 - height..=last].to_vec()
}

fn truncate(text: &str, width: usize) -> String {
    let mut out: String = text.chars().take(width).collect();
    let pad = width.saturating_sub(out.chars().count());
    out.extend(std::iter::repeat_n(' ', pad));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_breaks_prose_at_spaces_and_long_runs_at_the_edge() {
        assert_eq!(wrap("hello there world", 11), ["hello there", "world"]);
        assert_eq!(wrap("short", 11), ["short"]);
        assert_eq!(
            wrap("/a/very/long/path/with/no/spaces", 10),
            ["/a/very/lo", "ng/path/wi", "th/no/spac", "es"]
        );
    }

    #[test]
    fn the_composer_cursor_follows_its_text_through_the_wrap() {
        let lines = vec!["0123456789ab".to_string(), "second".to_string()];
        let (rows, cursor) = wrap_composer(&lines, (0, 11), 10);
        assert_eq!(rows, ["0123456789", "ab", "second"]);
        assert_eq!(cursor, (1, 1));

        let (_, cursor) = wrap_composer(&lines, (1, 3), 10);
        assert_eq!(cursor, (2, 3));
    }

    #[test]
    fn an_empty_composer_still_has_a_row_to_put_the_cursor_on() {
        let (rows, cursor) = wrap_composer(&[String::new()], (0, 0), 10);
        assert_eq!(rows, [""]);
        assert_eq!(cursor, (0, 0));
    }

    #[test]
    fn a_window_shows_the_end_of_a_stream_and_the_selected_row_of_a_list() {
        let rows: Vec<(Tone, String)> = (0..5)
            .map(|n| (Tone::Assistant, format!("row {n}")))
            .collect();
        assert_eq!(
            window(&rows, 2),
            [
                (Tone::Assistant, "row 3".to_string()),
                (Tone::Assistant, "row 4".to_string())
            ]
        );

        // The selected row is the last one shown, so walking up the list
        // scrolls it into view.
        let mut list = rows.clone();
        list[1].0 = Tone::Selected;
        assert_eq!(
            window(&list, 2),
            [
                (Tone::Assistant, "row 0".to_string()),
                (Tone::Selected, "row 1".to_string())
            ]
        );
    }
}

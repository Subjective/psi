//! The interface protocol: clients send commands, the harness emits events.
//! The in-process TUI and future external transports consume the same types.

use serde::{Deserialize, Serialize};

use crate::item::{CompletionStatus, Item, ItemId, ItemKind, TurnId};
use crate::session::{SessionId, SessionMeta, SessionSnapshot};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Command {
    CreateSession,
    LoadSession {
        session_id: SessionId,
    },
    ListSessions,
    SubmitMessage {
        session_id: SessionId,
        text: String,
    },
    CancelTurn {
        session_id: SessionId,
    },
    /// The branch primitive. `None` moves head before the first item, so the
    /// very first message can be edited into a fork.
    SetHead {
        session_id: SessionId,
        item_id: Option<ItemId>,
    },
}

/// Every event is sequenced and timestamped at emission, so trace export
/// (Milestone 5) is an assembly step rather than a retrofit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub seq: u64,
    pub timestamp_ms: u64,
    /// `None` only for `sessions_listed`, which spans sessions.
    pub session_id: Option<SessionId>,
    #[serde(flatten)]
    pub payload: EventPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventPayload {
    SessionCreated {
        meta: SessionMeta,
    },
    SessionLoaded {
        snapshot: SessionSnapshot,
    },
    SessionsListed {
        sessions: Vec<SessionMeta>,
    },
    TurnStarted {
        turn_id: TurnId,
    },
    ItemStarted {
        item_id: ItemId,
        kind: ItemKind,
    },
    ItemDelta {
        item_id: ItemId,
        delta: String,
    },
    /// Carries the complete durable item; its `status` and `error` fields are
    /// the statuses this event reports.
    ItemFinished {
        item: Item,
    },
    TurnFinished {
        turn_id: TurnId,
        status: CompletionStatus,
        /// Present when `status` is `Failed`; a turn that fails before any
        /// item starts records its error here alone.
        error: Option<String>,
    },
}

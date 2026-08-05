use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::item::{CompletionStatus, Item, ItemId, ItemPayload, TurnId};
use crate::store::SessionLog;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: SessionId,
    pub created_at_ms: u64,
}

/// The full durable state of a session, as carried by `session_loaded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub meta: SessionMeta,
    pub items: Vec<Item>,
    pub head: Option<ItemId>,
}

#[derive(Debug, thiserror::Error)]
#[error("unknown item: {0:?}")]
pub struct UnknownItem(pub ItemId);

/// One durable conversation: an append-only tree of items plus a `head`
/// pointer. The active conversation is the path from the root to `head`;
/// appending under a non-leaf head starts a new branch.
///
/// Every mutation writes through to the session's log before it is visible, so
/// there is no in-memory state a restart cannot recover.
#[derive(Debug)]
pub struct Session {
    pub meta: SessionMeta,
    items: Vec<Item>,
    index: HashMap<ItemId, usize>,
    head: Option<ItemId>,
    next_item_id: u64,
    next_turn_id: u64,
    /// Dropped after a failed write: the session keeps serving from memory,
    /// but nothing more reaches disk.
    log: Option<SessionLog>,
    log_error: Option<String>,
}

impl Session {
    pub fn new(meta: SessionMeta, log: SessionLog) -> Self {
        Self {
            meta,
            items: Vec::new(),
            index: HashMap::new(),
            head: None,
            next_item_id: 0,
            next_turn_id: 0,
            log: Some(log),
            log_error: None,
        }
    }

    /// Rebuilds a session from its log. The id counters continue past
    /// everything on disk, so a resumed session never reuses an id and item
    /// order stays append order.
    pub fn restore(snapshot: SessionSnapshot, log: SessionLog) -> Self {
        let index = snapshot
            .items
            .iter()
            .enumerate()
            .map(|(position, item)| (item.id, position))
            .collect();
        let next_item_id = snapshot.items.iter().map(|i| i.id.0 + 1).max().unwrap_or(0);
        let next_turn_id = snapshot
            .items
            .iter()
            .map(|i| i.turn_id.0 + 1)
            .max()
            .unwrap_or(0);
        Self {
            meta: snapshot.meta,
            items: snapshot.items,
            index,
            head: snapshot.head,
            next_item_id,
            next_turn_id,
            log: Some(log),
            log_error: None,
        }
    }

    /// The first log write that failed, taken so it is reported once. After it
    /// the session runs in memory only, and `turn_finished` is the sole place
    /// in the protocol that can tell a client its history stopped persisting.
    pub fn take_log_error(&mut self) -> Option<String> {
        self.log_error.take()
    }

    pub fn begin_turn(&mut self) -> TurnId {
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id += 1;
        turn_id
    }

    /// Hands out the id an in-progress item will be appended under, so
    /// `item_started` and `item_delta` events can reference it before the
    /// complete item exists. Every reserved id must be appended before the
    /// next reservation, which keeps id order equal to append order.
    pub fn reserve_item_id(&mut self) -> ItemId {
        let id = ItemId(self.next_item_id);
        self.next_item_id += 1;
        id
    }

    /// Appends a complete item under the current head and advances head to it.
    pub fn append(
        &mut self,
        id: ItemId,
        turn_id: TurnId,
        payload: ItemPayload,
        status: CompletionStatus,
        error: Option<String>,
        created_at_ms: u64,
    ) -> &Item {
        debug_assert!(!self.index.contains_key(&id), "item id appended twice");
        let item = Item {
            id,
            parent_id: self.head,
            turn_id,
            created_at_ms,
            status,
            error,
            payload,
        };
        self.log(|log| log.append_item(&item));
        self.index.insert(id, self.items.len());
        self.items.push(item);
        self.head = Some(id);
        self.items.last().expect("just pushed")
    }

    /// Moves head to any existing item, or to `None` (before the first item).
    /// This is the branch primitive: submitting afterwards forks the tree.
    pub fn set_head(&mut self, head: Option<ItemId>) -> Result<(), UnknownItem> {
        if let Some(id) = head {
            if !self.index.contains_key(&id) {
                return Err(UnknownItem(id));
            }
        }
        self.head = head;
        // An appended item records its own head move, so only this one needs a
        // record of its own.
        self.log(|log| log.set_head(head));
        Ok(())
    }

    fn log(&mut self, write: impl FnOnce(&mut SessionLog) -> std::io::Result<()>) {
        let Some(log) = self.log.as_mut() else { return };
        if let Err(err) = write(log) {
            self.log_error
                .get_or_insert_with(|| format!("session log: {err}"));
            self.log = None;
        }
    }

    /// The active conversation: root to head, in order.
    pub fn path_to_head(&self) -> Vec<&Item> {
        let mut path = Vec::new();
        let mut cursor = self.head;
        while let Some(id) = cursor {
            let item = &self.items[self.index[&id]];
            path.push(item);
            cursor = item.parent_id;
        }
        path.reverse();
        path
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            meta: self.meta.clone(),
            items: self.items.clone(),
            head: self.head,
        }
    }
}

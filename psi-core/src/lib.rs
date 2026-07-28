//! The Psi harness: sessions as append-only item trees, the interface
//! protocol, the turn engine, tools, and the model boundary.
//!
//! `docs/design.md` is the authoritative design.

pub mod bench;
pub mod engine;
pub mod fake;
pub mod hook;
pub mod item;
pub mod model;
pub mod openai;
pub mod protocol;
pub mod responses;
pub mod session;
pub mod store;
pub mod tool;
pub mod tools;
pub mod trace;

pub use engine::{Harness, HarnessConfig};

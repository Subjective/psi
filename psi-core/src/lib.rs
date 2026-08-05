//! The Psi harness: sessions as append-only item trees, the interface
//! protocol, the turn engine, tools, and the model boundary.
//!
//! `docs/design.md` is the authoritative design.

pub mod engine;
pub mod fake;
pub mod item;
pub mod model;
pub mod protocol;
pub mod session;
pub mod tool;

pub use engine::Harness;

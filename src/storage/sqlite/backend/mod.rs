//! SQLite Backend implementation for Experiment Storage.
//!
//! Sovereign, local-first storage using SQLite with WAL mode.
//!
//! # Toyota Way: (Heijunka)
//!
//! SQLite provides consistent, predictable performance without external dependencies.

mod sqlite_backend;
mod state;

pub use sqlite_backend::SqliteBackend;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

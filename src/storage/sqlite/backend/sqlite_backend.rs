//! SQLite Backend core implementation.
//!
//! Contains the SqliteBackend struct and its core methods.

use super::state::SqliteState;
use crate::storage::Result;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// SQLite backend for experiment storage
///
/// Currently uses an in-memory implementation with SQLite-compatible schema.
/// Will be upgraded to actual SQLite via rusqlite/sqlx.
#[derive(Debug)]
pub struct SqliteBackend {
    pub(crate) path: String,
    #[cfg(test)]
    pub(crate) state: Arc<RwLock<SqliteState>>,
    #[cfg(not(test))]
    pub(crate) state: Arc<RwLock<SqliteState>>,
}

impl SqliteBackend {
    /// Open or create a SQLite database at the given path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database file (use ":memory:" for in-memory)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        Ok(Self { path: path_str, state: Arc::new(RwLock::new(SqliteState::default())) })
    }

    /// Open an in-memory database
    pub fn open_in_memory() -> Result<Self> {
        Self::open(":memory:")
    }

    /// Get the database path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Generate a unique ID
    pub(crate) fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must not be before UNIX epoch")
            .as_nanos();
        format!("{ts:x}")
    }
}

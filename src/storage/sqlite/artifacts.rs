//! Artifact operations for SQLite Backend.
//!
//! Contains artifact retrieval and listing methods.

use super::backend::SqliteBackend;
use super::types::ArtifactRef;
use crate::storage::{Result, StorageError};

impl SqliteBackend {
    /// Get artifact data by SHA-256 hash
    pub fn get_artifact_data(&self, sha256: &str) -> Result<Vec<u8>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        state
            .artifact_data
            .get(sha256)
            .cloned()
            .ok_or_else(|| StorageError::Backend(format!("Artifact not found: {sha256}")))
    }

    /// List artifacts for a run
    pub fn list_artifacts(&self, run_id: &str) -> Result<Vec<ArtifactRef>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        Ok(state.artifacts.get(run_id).cloned().unwrap_or_default())
    }
}

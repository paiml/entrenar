//! Metrics operations for SQLite Backend.
//!
//! Contains metric logging and retrieval methods (ExperimentStorage trait implementation).

use super::backend::SqliteBackend;
use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};
use chrono::{DateTime, Utc};
use rusqlite::params;
use sha2::{Digest, Sha256};

/// Map a RunStatus to its SQLite TEXT representation
fn status_to_str(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Pending => "pending",
        RunStatus::Running => "running",
        RunStatus::Success => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

/// Parse a SQLite TEXT status back to RunStatus
pub(crate) fn str_to_status(s: &str) -> RunStatus {
    match s {
        "pending" => RunStatus::Pending,
        "running" => RunStatus::Running,
        "completed" => RunStatus::Success,
        "failed" => RunStatus::Failed,
        "cancelled" => RunStatus::Cancelled,
        _ => RunStatus::Failed,
    }
}

impl ExperimentStorage for SqliteBackend {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = Self::generate_id();
        let config_json = config.map(|c| c.to_string());
        let now = Utc::now().to_rfc3339();

        let conn = self.lock_conn()?;
        conn.execute(
            "INSERT INTO experiments (id, name, config, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, name, config_json, now, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to create experiment: {e}")))?;

        Ok(id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        let conn = self.lock_conn()?;

        // Verify experiment exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = ?1)",
                [experiment_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check experiment: {e}")))?;

        if !exists {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = Self::generate_id();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO runs (id, experiment_id, status, start_time) VALUES (?1, ?2, 'pending', ?3)",
            params![id, experiment_id, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to create run: {e}")))?;

        Ok(id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let conn = self.lock_conn()?;

        let current_status: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        if current_status != "pending" {
            return Err(StorageError::InvalidState(format!(
                "Cannot start run in {current_status} status"
            )));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE runs SET status = 'running', start_time = ?1 WHERE id = ?2",
            params![now, run_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to start run: {e}")))?;

        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let conn = self.lock_conn()?;

        let current_status: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        if current_status != "running" {
            return Err(StorageError::InvalidState(format!(
                "Cannot complete run in {current_status} status"
            )));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE runs SET status = ?1, end_time = ?2 WHERE id = ?3",
            params![status_to_str(status), now, run_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to complete run: {e}")))?;

        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)",
                [run_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO metrics (run_id, key, step, value, timestamp) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![run_id, key, step as i64, value, now],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log metric: {e}")))?;

        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)",
                [run_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        // Compute SHA-256 for content-addressable storage
        let mut hasher = Sha256::new();
        hasher.update(data);
        let sha256 = format!("{:x}", hasher.finalize());

        let id = Self::generate_id();
        let size = data.len() as i64;

        conn.execute(
            "INSERT INTO artifacts (id, run_id, path, size_bytes, sha256, data) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, run_id, key, size, sha256, data],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log artifact: {e}")))?;

        Ok(sha256)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)",
                [run_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let mut stmt = conn
            .prepare("SELECT step, value, timestamp FROM metrics WHERE run_id = ?1 AND key = ?2 ORDER BY step")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare metrics query: {e}")))?;

        let points = stmt
            .query_map(params![run_id, key], |row| {
                let step: i64 = row.get(0)?;
                let value: f64 = row.get(1)?;
                let ts_str: String = row.get(2)?;
                let timestamp: DateTime<Utc> = ts_str
                    .parse()
                    .unwrap_or_else(|_| Utc::now());
                Ok(MetricPoint::with_timestamp(step as u64, value, timestamp))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query metrics: {e}")))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| StorageError::Backend(format!("Failed to read metric row: {e}")))?;

        Ok(points)
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let conn = self.lock_conn()?;

        let status_str: String = conn
            .query_row("SELECT status FROM runs WHERE id = ?1", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run status: {e}")),
            })?;

        Ok(str_to_status(&status_str))
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)",
                [run_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        conn.execute(
            "INSERT OR REPLACE INTO span_ids (run_id, span_id) VALUES (?1, ?2)",
            params![run_id, span_id],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to set span ID: {e}")))?;

        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)",
                [run_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let result = conn.query_row(
            "SELECT span_id FROM span_ids WHERE run_id = ?1",
            [run_id],
            |row| row.get(0),
        );

        match result {
            Ok(span_id) => Ok(Some(span_id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::Backend(format!("Failed to get span ID: {e}"))),
        }
    }
}

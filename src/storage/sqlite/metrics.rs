//! Metrics operations for SQLite Backend.
//!
//! Contains metric logging and retrieval methods (ExperimentStorage trait implementation).

use super::backend::SqliteBackend;
use super::types::{ArtifactRef, Experiment, Run};
use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

impl ExperimentStorage for SqliteBackend {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        let id = Self::generate_id();
        let now = Utc::now();

        let experiment = Experiment {
            id: id.clone(),
            name: name.to_string(),
            description: None,
            config,
            tags: HashMap::new(),
            created_at: now,
            updated_at: now,
        };

        state.experiments.insert(id.clone(), experiment);
        Ok(id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.experiments.contains_key(experiment_id) {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = Self::generate_id();

        let run = Run {
            id: id.clone(),
            experiment_id: experiment_id.to_string(),
            status: RunStatus::Pending,
            start_time: Utc::now(),
            end_time: None,
            params: HashMap::new(),
            tags: HashMap::new(),
        };

        state.runs.insert(id.clone(), run);
        Ok(id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        let run = state
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Pending {
            return Err(StorageError::InvalidState(format!(
                "Cannot start run in {:?} status",
                run.status
            )));
        }

        run.status = RunStatus::Running;
        run.start_time = Utc::now();
        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        let run = state
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Running {
            return Err(StorageError::InvalidState(format!(
                "Cannot complete run in {:?} status",
                run.status
            )));
        }

        run.status = status;
        run.end_time = Some(Utc::now());
        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let point = MetricPoint::new(step, value);

        state
            .metrics
            .entry(run_id.to_string())
            .or_default()
            .entry(key.to_string())
            .or_default()
            .push(point);

        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        // Compute SHA-256 for content-addressable storage
        let mut hasher = Sha256::new();
        hasher.update(data);
        let sha256 = format!("{:x}", hasher.finalize());

        // Store artifact data (deduplicated by hash)
        state.artifact_data.entry(sha256.clone()).or_insert_with(|| data.to_vec());

        // Create artifact reference
        let artifact = ArtifactRef {
            id: Self::generate_id(),
            run_id: run_id.to_string(),
            path: key.to_string(),
            size_bytes: data.len() as u64,
            sha256: sha256.clone(),
            created_at: Utc::now(),
        };

        state.artifacts.entry(run_id.to_string()).or_default().push(artifact);

        Ok(sha256)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        Ok(state.metrics.get(run_id).and_then(|m| m.get(key)).cloned().unwrap_or_default())
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        state
            .runs
            .get(run_id)
            .map(|r| r.status)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        state.span_ids.insert(run_id.to_string(), span_id.to_string());
        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        Ok(state.span_ids.get(run_id).cloned())
    }
}

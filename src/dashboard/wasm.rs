//! WASM Dashboard Bindings (ENT-004)
//!
//! Provides browser-compatible dashboard implementation using IndexedDB
//! for storage and wasm_bindgen for JavaScript interop.
//!
//! # Features
//!
//! - `IndexedDbStorage`: Persistent storage in browser IndexedDB
//! - `WasmRun`: WASM-compatible run wrapper
//! - Callback-based metric subscriptions
//!
//! # Usage
//!
//! ```javascript
//! import { WasmRun } from 'entrenar';
//!
//! const run = await WasmRun.new('experiment-1');
//! run.log_metric('loss', 0.5);
//! run.log_metric('loss', 0.4);
//!
//! const metrics = run.get_metrics_json();
//! console.log(JSON.parse(metrics));
//!
//! run.subscribe_metrics((key, value) => {
//!     console.log(`${key}: ${value}`);
//! });
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};

/// IndexedDB-backed storage for browser environments.
///
/// This is a simplified in-memory implementation that mimics
/// IndexedDB behavior. A full implementation would use web-sys
/// to interact with the actual IndexedDB API.
#[derive(Debug, Default)]
pub struct IndexedDbStorage {
    /// Experiments by ID
    experiments: HashMap<String, ExperimentData>,
    /// Runs by ID
    runs: HashMap<String, RunData>,
    /// Metrics by run_id -> key -> points
    metrics: HashMap<String, HashMap<String, Vec<MetricPoint>>>,
    /// Artifacts by hash
    artifacts: HashMap<String, Vec<u8>>,
    /// Next experiment ID counter
    next_exp_id: u64,
    /// Next run ID counter
    next_run_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentData {
    id: String,
    name: String,
    config: Option<serde_json::Value>,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunData {
    id: String,
    experiment_id: String,
    status: RunStatus,
    span_id: Option<String>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
}

impl IndexedDbStorage {
    /// Create a new IndexedDB storage instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all experiments.
    pub fn list_experiments(&self) -> Vec<String> {
        self.experiments.keys().cloned().collect()
    }

    /// Get all runs for an experiment.
    pub fn list_runs(&self, experiment_id: &str) -> Vec<String> {
        self.runs
            .values()
            .filter(|r| r.experiment_id == experiment_id)
            .map(|r| r.id.clone())
            .collect()
    }

    /// Get all metric keys for a run.
    pub fn list_metric_keys(&self, run_id: &str) -> Vec<String> {
        self.metrics
            .get(run_id)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }
}

impl ExperimentStorage for IndexedDbStorage {
    fn create_experiment(
        &mut self,
        name: &str,
        config: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = format!("exp-{}", self.next_exp_id);
        self.next_exp_id += 1;

        let experiment = ExperimentData {
            id: id.clone(),
            name: name.to_string(),
            config,
            created_at: Utc::now(),
        };

        self.experiments.insert(id.clone(), experiment);
        Ok(id)
    }

    fn create_run(&mut self, experiment_id: &str) -> Result<String> {
        if !self.experiments.contains_key(experiment_id) {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let id = format!("run-{}", self.next_run_id);
        self.next_run_id += 1;

        let run = RunData {
            id: id.clone(),
            experiment_id: experiment_id.to_string(),
            status: RunStatus::Pending,
            span_id: None,
            started_at: None,
            completed_at: None,
        };

        self.runs.insert(id.clone(), run);
        self.metrics.insert(id.clone(), HashMap::new());
        Ok(id)
    }

    fn start_run(&mut self, run_id: &str) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Pending {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Pending state"
            )));
        }

        run.status = RunStatus::Running;
        run.started_at = Some(Utc::now());
        Ok(())
    }

    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        if run.status != RunStatus::Running {
            return Err(StorageError::InvalidState(format!(
                "Run {run_id} is not in Running state"
            )));
        }

        run.status = status;
        run.completed_at = Some(Utc::now());
        Ok(())
    }

    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()> {
        let metrics = self
            .metrics
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        let points = metrics.entry(key.to_string()).or_default();
        points.push(MetricPoint::new(step, value));
        Ok(())
    }

    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String> {
        if !self.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        // Compute SHA-256 hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hex::encode(hasher.finalize());

        // Store artifact
        let artifact_key = format!("{run_id}/{key}/{hash}");
        self.artifacts.insert(artifact_key, data.to_vec());

        Ok(hash)
    }

    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>> {
        let metrics = self
            .metrics
            .get(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(metrics.get(key).cloned().unwrap_or_default())
    }

    fn get_run_status(&self, run_id: &str) -> Result<RunStatus> {
        let run = self
            .runs
            .get(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(run.status)
    }

    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()> {
        let run = self
            .runs
            .get_mut(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        run.span_id = Some(span_id.to_string());
        Ok(())
    }

    fn get_span_id(&self, run_id: &str) -> Result<Option<String>> {
        let run = self
            .runs
            .get(run_id)
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))?;

        Ok(run.span_id.clone())
    }
}

/// WASM-compatible run wrapper.
///
/// Provides a JavaScript-friendly API for training runs.
#[wasm_bindgen]
pub struct WasmRun {
    run_id: String,
    experiment_id: String,
    storage: Arc<Mutex<IndexedDbStorage>>,
    step_counters: HashMap<String, u64>,
    finished: bool,
}

#[wasm_bindgen]
impl WasmRun {
    /// Create a new run in a new experiment.
    #[wasm_bindgen(constructor)]
    pub fn new(experiment_name: &str) -> std::result::Result<WasmRun, JsValue> {
        let mut storage = IndexedDbStorage::new();

        let experiment_id = storage
            .create_experiment(experiment_name, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let run_id = storage
            .create_run(&experiment_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        storage
            .start_run(&run_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            run_id,
            experiment_id,
            storage: Arc::new(Mutex::new(storage)),
            step_counters: HashMap::new(),
            finished: false,
        })
    }

    /// Log a metric value, auto-incrementing the step.
    pub fn log_metric(&mut self, key: &str, value: f64) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Err(JsValue::from_str("Cannot log to finished run"));
        }

        let step = *self.step_counters.get(key).unwrap_or(&0);
        self.log_metric_at(key, step, value)?;
        self.step_counters.insert(key.to_string(), step + 1);
        Ok(())
    }

    /// Log a metric value at a specific step.
    pub fn log_metric_at(
        &mut self,
        key: &str,
        step: u64,
        value: f64,
    ) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Err(JsValue::from_str("Cannot log to finished run"));
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .log_metric(&self.run_id, key, step, value)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }

    /// Get all metrics as a JSON string.
    pub fn get_metrics_json(&self) -> std::result::Result<String, JsValue> {
        let storage = self
            .storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let keys = storage.list_metric_keys(&self.run_id);
        let mut metrics: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for key in keys {
            if let Ok(points) = storage.get_metrics(&self.run_id, &key) {
                let values: Vec<serde_json::Value> = points
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "step": p.step,
                            "value": p.value,
                            "timestamp": p.timestamp.to_rfc3339()
                        })
                    })
                    .collect();
                metrics.insert(key, values);
            }
        }

        serde_json::to_string(&metrics).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Subscribe to metric updates via a JavaScript callback.
    ///
    /// The callback receives (key: string, value: number) for each update.
    pub fn subscribe_metrics(&self, _callback: &js_sys::Function) {
        // In a full implementation, this would store the callback
        // and invoke it when metrics are logged.
        // For now, this is a placeholder showing the API.
    }

    /// Get the run ID.
    pub fn run_id(&self) -> String {
        self.run_id.clone()
    }

    /// Get the experiment ID.
    pub fn experiment_id(&self) -> String {
        self.experiment_id.clone()
    }

    /// Get current step for a metric key.
    pub fn current_step(&self, key: &str) -> u64 {
        *self.step_counters.get(key).unwrap_or(&0)
    }

    /// Check if the run is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Finish the run with success status.
    pub fn finish(&mut self) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Ok(());
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .complete_run(&self.run_id, RunStatus::Success)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.finished = true;
        Ok(())
    }

    /// Finish the run with failed status.
    pub fn fail(&mut self) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Ok(());
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .complete_run(&self.run_id, RunStatus::Failed)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.finished = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_db_storage_create_experiment() {
        let mut storage = IndexedDbStorage::new();
        let id = storage.create_experiment("test-exp", None).unwrap();

        assert!(id.starts_with("exp-"));
        assert_eq!(storage.list_experiments().len(), 1);
    }

    #[test]
    fn test_indexed_db_storage_create_run() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();

        assert!(run_id.starts_with("run-"));
        assert_eq!(storage.get_run_status(&run_id).unwrap(), RunStatus::Pending);
    }

    #[test]
    fn test_indexed_db_storage_run_lifecycle() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();

        storage.start_run(&run_id).unwrap();
        assert_eq!(storage.get_run_status(&run_id).unwrap(), RunStatus::Running);

        storage.complete_run(&run_id, RunStatus::Success).unwrap();
        assert_eq!(storage.get_run_status(&run_id).unwrap(), RunStatus::Success);
    }

    #[test]
    fn test_indexed_db_storage_log_metrics() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();
        storage.start_run(&run_id).unwrap();

        storage.log_metric(&run_id, "loss", 0, 0.5).unwrap();
        storage.log_metric(&run_id, "loss", 1, 0.4).unwrap();
        storage.log_metric(&run_id, "accuracy", 0, 0.8).unwrap();

        let loss_metrics = storage.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(loss_metrics.len(), 2);
        assert!((loss_metrics[0].value - 0.5).abs() < f64::EPSILON);

        let keys = storage.list_metric_keys(&run_id);
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_indexed_db_storage_log_artifact() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();
        storage.start_run(&run_id).unwrap();

        let data = b"test artifact data";
        let hash = storage.log_artifact(&run_id, "model.bin", data).unwrap();

        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_indexed_db_storage_span_id() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();

        storage.set_span_id(&run_id, "span-123").unwrap();
        let span_id = storage.get_span_id(&run_id).unwrap();

        assert_eq!(span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_indexed_db_storage_error_experiment_not_found() {
        let mut storage = IndexedDbStorage::new();
        let result = storage.create_run("nonexistent");

        assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
    }

    #[test]
    fn test_indexed_db_storage_error_run_not_found() {
        let storage = IndexedDbStorage::new();
        let result = storage.get_run_status("nonexistent");

        assert!(matches!(result, Err(StorageError::RunNotFound(_))));
    }

    #[test]
    fn test_indexed_db_storage_error_invalid_state_start() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();

        storage.start_run(&run_id).unwrap();
        let result = storage.start_run(&run_id); // Try to start again

        assert!(matches!(result, Err(StorageError::InvalidState(_))));
    }

    #[test]
    fn test_indexed_db_storage_error_invalid_state_complete() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();
        let run_id = storage.create_run(&exp_id).unwrap();

        // Try to complete without starting
        let result = storage.complete_run(&run_id, RunStatus::Success);

        assert!(matches!(result, Err(StorageError::InvalidState(_))));
    }

    #[test]
    fn test_indexed_db_storage_list_runs() {
        let mut storage = IndexedDbStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).unwrap();

        storage.create_run(&exp_id).unwrap();
        storage.create_run(&exp_id).unwrap();
        storage.create_run(&exp_id).unwrap();

        let runs = storage.list_runs(&exp_id);
        assert_eq!(runs.len(), 3);
    }

    // Note: WasmRun tests require wasm-bindgen-test for full coverage
    // These are basic structural tests
    #[test]
    fn test_indexed_db_storage_implements_trait() {
        fn assert_storage<S: ExperimentStorage>(_: &S) {}

        let storage = IndexedDbStorage::new();
        assert_storage(&storage);
    }
}

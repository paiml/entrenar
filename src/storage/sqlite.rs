//! SQLite Backend for Experiment Storage (MLOPS-001)
//!
//! Sovereign, local-first storage using SQLite with WAL mode.
//!
//! # Toyota Way: 平準化 (Heijunka)
//!
//! SQLite provides consistent, predictable performance without external dependencies.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::storage::{SqliteBackend, ExperimentStorage, RunStatus};
//!
//! let backend = SqliteBackend::open("./experiments.db")?;
//! let exp_id = backend.create_experiment("my-exp", None)?;
//! let run_id = backend.create_run(&exp_id)?;
//! backend.log_metric(&run_id, "loss", 0, 0.5)?;
//! ```

use crate::storage::{ExperimentStorage, MetricPoint, Result, RunStatus, StorageError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Parameter value types for log_param
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParameterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ParameterValue>),
    Dict(HashMap<String, ParameterValue>),
}

impl ParameterValue {
    /// Get type name for storage
    pub fn type_name(&self) -> &'static str {
        match self {
            ParameterValue::String(_) => "string",
            ParameterValue::Int(_) => "int",
            ParameterValue::Float(_) => "float",
            ParameterValue::Bool(_) => "bool",
            ParameterValue::List(_) => "list",
            ParameterValue::Dict(_) => "dict",
        }
    }

    /// Serialize to JSON string for storage
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Deserialize from JSON string
    pub fn from_json(s: &str) -> Option<Self> {
        serde_json::from_str(s).ok()
    }
}

/// Filter operations for parameter search
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
    StartsWith,
}

/// Parameter filter for searching runs
#[derive(Debug, Clone)]
pub struct ParamFilter {
    pub key: String,
    pub op: FilterOp,
    pub value: ParameterValue,
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Run metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub params: HashMap<String, ParameterValue>,
    pub tags: HashMap<String, String>,
}

/// Artifact reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub run_id: String,
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub created_at: DateTime<Utc>,
}

/// Internal storage for experiments, runs, metrics, and artifacts
#[derive(Debug, Default)]
struct SqliteState {
    experiments: HashMap<String, Experiment>,
    runs: HashMap<String, Run>,
    metrics: HashMap<String, HashMap<String, Vec<MetricPoint>>>, // run_id -> key -> points
    params: HashMap<String, HashMap<String, ParameterValue>>,    // run_id -> key -> value
    artifacts: HashMap<String, Vec<ArtifactRef>>,                // run_id -> artifacts
    artifact_data: HashMap<String, Vec<u8>>,                     // sha256 -> data (CAS)
    span_ids: HashMap<String, String>,                           // run_id -> span_id
}

/// SQLite backend for experiment storage
///
/// Currently uses an in-memory implementation with SQLite-compatible schema.
/// Will be upgraded to actual SQLite via rusqlite/sqlx.
#[derive(Debug)]
pub struct SqliteBackend {
    path: String,
    state: Arc<RwLock<SqliteState>>,
}

impl SqliteBackend {
    /// Open or create a SQLite database at the given path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database file (use ":memory:" for in-memory)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        Ok(Self {
            path: path_str,
            state: Arc::new(RwLock::new(SqliteState::default())),
        })
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
    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{ts:x}")
    }

    /// Log a parameter for a run
    pub fn log_param(&self, run_id: &str, key: &str, value: ParameterValue) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        state
            .params
            .entry(run_id.to_string())
            .or_default()
            .insert(key.to_string(), value);

        Ok(())
    }

    /// Log multiple parameters for a run
    pub fn log_params(&self, run_id: &str, params: HashMap<String, ParameterValue>) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire write lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let run_params = state.params.entry(run_id.to_string()).or_default();
        for (key, value) in params {
            run_params.insert(key, value);
        }

        Ok(())
    }

    /// Get parameters for a run
    pub fn get_params(&self, run_id: &str) -> Result<HashMap<String, ParameterValue>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        if !state.runs.contains_key(run_id) {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        Ok(state.params.get(run_id).cloned().unwrap_or_default())
    }

    /// Search runs by parameter filters
    pub fn search_runs_by_params(&self, filters: &[ParamFilter]) -> Result<Vec<Run>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        let mut results = Vec::new();

        for run in state.runs.values() {
            let run_params = state.params.get(&run.id);

            let matches = filters.iter().all(|filter| {
                if let Some(params) = run_params {
                    if let Some(value) = params.get(&filter.key) {
                        Self::param_matches(value, &filter.op, &filter.value)
                    } else {
                        false
                    }
                } else {
                    false
                }
            });

            if matches || filters.is_empty() {
                results.push(run.clone());
            }
        }

        Ok(results)
    }

    /// Check if a parameter value matches a filter
    fn param_matches(value: &ParameterValue, op: &FilterOp, filter_value: &ParameterValue) -> bool {
        match (value, filter_value, op) {
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Eq) => {
                (v - fv).abs() < f64::EPSILON
            }
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Ne) => {
                (v - fv).abs() >= f64::EPSILON
            }
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Gt) => v > fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Lt) => v < fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Gte) => v >= fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Lte) => v <= fv,

            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gt) => v > fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lt) => v < fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gte) => v >= fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lte) => v <= fv,

            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Contains) => {
                v.contains(fv.as_str())
            }
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::StartsWith) => {
                v.starts_with(fv.as_str())
            }

            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Ne) => v != fv,

            _ => false,
        }
    }

    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Result<Experiment> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        state
            .experiments
            .get(experiment_id)
            .cloned()
            .ok_or_else(|| StorageError::ExperimentNotFound(experiment_id.to_string()))
    }

    /// Get a run by ID
    pub fn get_run(&self, run_id: &str) -> Result<Run> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        state
            .runs
            .get(run_id)
            .cloned()
            .ok_or_else(|| StorageError::RunNotFound(run_id.to_string()))
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Result<Vec<Experiment>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        Ok(state.experiments.values().cloned().collect())
    }

    /// List runs for an experiment
    pub fn list_runs(&self, experiment_id: &str) -> Result<Vec<Run>> {
        let state = self
            .state
            .read()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire read lock: {e}")))?;

        if !state.experiments.contains_key(experiment_id) {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        Ok(state
            .runs
            .values()
            .filter(|r| r.experiment_id == experiment_id)
            .cloned()
            .collect())
    }

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
        state
            .artifact_data
            .entry(sha256.clone())
            .or_insert_with(|| data.to_vec());

        // Create artifact reference
        let artifact = ArtifactRef {
            id: Self::generate_id(),
            run_id: run_id.to_string(),
            path: key.to_string(),
            size_bytes: data.len() as u64,
            sha256: sha256.clone(),
            created_at: Utc::now(),
        };

        state
            .artifacts
            .entry(run_id.to_string())
            .or_default()
            .push(artifact);

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

        Ok(state
            .metrics
            .get(run_id)
            .and_then(|m| m.get(key))
            .cloned()
            .unwrap_or_default())
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

        state
            .span_ids
            .insert(run_id.to_string(), span_id.to_string());
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

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SqliteBackend Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_open_in_memory() {
        let backend = SqliteBackend::open_in_memory();
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().path(), ":memory:");
    }

    #[test]
    fn test_open_file_path() {
        let backend = SqliteBackend::open("/tmp/test.db");
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().path(), "/tmp/test.db");
    }

    // -------------------------------------------------------------------------
    // Experiment CRUD Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_experiment() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        assert!(!exp_id.is_empty());
    }

    #[test]
    fn test_create_experiment_with_config() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let config = serde_json::json!({"lr": 0.001, "epochs": 10});
        let exp_id = backend
            .create_experiment("test-exp", Some(config.clone()))
            .unwrap();

        let exp = backend.get_experiment(&exp_id).unwrap();
        assert_eq!(exp.name, "test-exp");
        assert_eq!(exp.config, Some(config));
    }

    #[test]
    fn test_get_experiment_not_found() {
        let backend = SqliteBackend::open_in_memory().unwrap();
        let result = backend.get_experiment("nonexistent");
        assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
    }

    #[test]
    fn test_list_experiments() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        backend.create_experiment("exp-1", None).unwrap();
        backend.create_experiment("exp-2", None).unwrap();

        let experiments = backend.list_experiments().unwrap();
        assert_eq!(experiments.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Run Lifecycle Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_run() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        assert!(!run_id.is_empty());
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Pending);
    }

    #[test]
    fn test_create_run_experiment_not_found() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let result = backend.create_run("nonexistent");
        assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
    }

    #[test]
    fn test_start_run() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.start_run(&run_id).unwrap();
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Running);
    }

    #[test]
    fn test_start_run_not_found() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let result = backend.start_run("nonexistent");
        assert!(matches!(result, Err(StorageError::RunNotFound(_))));
    }

    #[test]
    fn test_start_run_invalid_state() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.start_run(&run_id).unwrap();
        let result = backend.start_run(&run_id);
        assert!(matches!(result, Err(StorageError::InvalidState(_))));
    }

    #[test]
    fn test_complete_run_success() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.start_run(&run_id).unwrap();
        backend.complete_run(&run_id, RunStatus::Success).unwrap();
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Success);
    }

    #[test]
    fn test_complete_run_failed() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.start_run(&run_id).unwrap();
        backend.complete_run(&run_id, RunStatus::Failed).unwrap();
        assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Failed);
    }

    #[test]
    fn test_complete_run_invalid_state() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        // Try to complete without starting
        let result = backend.complete_run(&run_id, RunStatus::Success);
        assert!(matches!(result, Err(StorageError::InvalidState(_))));
    }

    #[test]
    fn test_list_runs() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();

        backend.create_run(&exp_id).unwrap();
        backend.create_run(&exp_id).unwrap();

        let runs = backend.list_runs(&exp_id).unwrap();
        assert_eq!(runs.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Metrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_log_metric() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.log_metric(&run_id, "loss", 0, 0.5).unwrap();
        backend.log_metric(&run_id, "loss", 1, 0.4).unwrap();
        backend.log_metric(&run_id, "loss", 2, 0.3).unwrap();

        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(metrics.len(), 3);
        assert!((metrics[0].value - 0.5).abs() < f64::EPSILON);
        assert!((metrics[1].value - 0.4).abs() < f64::EPSILON);
        assert!((metrics[2].value - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_metric_run_not_found() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let result = backend.log_metric("nonexistent", "loss", 0, 0.5);
        assert!(matches!(result, Err(StorageError::RunNotFound(_))));
    }

    #[test]
    fn test_get_metrics_empty() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert!(metrics.is_empty());
    }

    // -------------------------------------------------------------------------
    // Artifact Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_log_artifact() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let data = b"model weights data";
        let sha256 = backend.log_artifact(&run_id, "model.bin", data).unwrap();

        assert!(!sha256.is_empty());
        assert_eq!(sha256.len(), 64); // SHA-256 hex length
    }

    #[test]
    fn test_artifact_deduplication() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let data = b"same data";
        let sha1 = backend.log_artifact(&run_id, "file1.bin", data).unwrap();
        let sha2 = backend.log_artifact(&run_id, "file2.bin", data).unwrap();

        // Same data should produce same hash
        assert_eq!(sha1, sha2);
    }

    #[test]
    fn test_get_artifact_data() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let data = b"test artifact data";
        let sha256 = backend.log_artifact(&run_id, "file.bin", data).unwrap();

        let retrieved = backend.get_artifact_data(&sha256).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_list_artifacts() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend
            .log_artifact(&run_id, "model.bin", b"model")
            .unwrap();
        backend
            .log_artifact(&run_id, "config.json", b"config")
            .unwrap();

        let artifacts = backend.list_artifacts(&run_id).unwrap();
        assert_eq!(artifacts.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Parameter Tests (MLOPS-003)
    // -------------------------------------------------------------------------

    #[test]
    fn test_log_param_string() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend
            .log_param(
                &run_id,
                "optimizer",
                ParameterValue::String("adam".to_string()),
            )
            .unwrap();

        let params = backend.get_params(&run_id).unwrap();
        assert_eq!(
            params.get("optimizer"),
            Some(&ParameterValue::String("adam".to_string()))
        );
    }

    #[test]
    fn test_log_param_float() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend
            .log_param(&run_id, "lr", ParameterValue::Float(0.001))
            .unwrap();

        let params = backend.get_params(&run_id).unwrap();
        assert_eq!(params.get("lr"), Some(&ParameterValue::Float(0.001)));
    }

    #[test]
    fn test_log_param_int() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend
            .log_param(&run_id, "epochs", ParameterValue::Int(100))
            .unwrap();

        let params = backend.get_params(&run_id).unwrap();
        assert_eq!(params.get("epochs"), Some(&ParameterValue::Int(100)));
    }

    #[test]
    fn test_log_param_bool() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend
            .log_param(&run_id, "use_cuda", ParameterValue::Bool(true))
            .unwrap();

        let params = backend.get_params(&run_id).unwrap();
        assert_eq!(params.get("use_cuda"), Some(&ParameterValue::Bool(true)));
    }

    #[test]
    fn test_log_params_batch() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let mut params = HashMap::new();
        params.insert("lr".to_string(), ParameterValue::Float(0.001));
        params.insert("epochs".to_string(), ParameterValue::Int(100));
        params.insert(
            "optimizer".to_string(),
            ParameterValue::String("adam".to_string()),
        );

        backend.log_params(&run_id, params).unwrap();

        let retrieved = backend.get_params(&run_id).unwrap();
        assert_eq!(retrieved.len(), 3);
    }

    #[test]
    fn test_log_param_run_not_found() {
        let backend = SqliteBackend::open_in_memory().unwrap();
        let result = backend.log_param("nonexistent", "key", ParameterValue::Int(1));
        assert!(matches!(result, Err(StorageError::RunNotFound(_))));
    }

    // -------------------------------------------------------------------------
    // Parameter Search Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_search_runs_by_params_eq() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();

        let run1 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(&run1, "lr", ParameterValue::Float(0.001))
            .unwrap();

        let run2 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(&run2, "lr", ParameterValue::Float(0.01))
            .unwrap();

        let filters = vec![ParamFilter {
            key: "lr".to_string(),
            op: FilterOp::Eq,
            value: ParameterValue::Float(0.001),
        }];

        let results = backend.search_runs_by_params(&filters).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, run1);
    }

    #[test]
    fn test_search_runs_by_params_gt() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();

        let run1 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(&run1, "lr", ParameterValue::Float(0.001))
            .unwrap();

        let run2 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(&run2, "lr", ParameterValue::Float(0.01))
            .unwrap();

        let filters = vec![ParamFilter {
            key: "lr".to_string(),
            op: FilterOp::Gt,
            value: ParameterValue::Float(0.005),
        }];

        let results = backend.search_runs_by_params(&filters).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, run2);
    }

    #[test]
    fn test_search_runs_by_params_contains() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();

        let run1 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(
                &run1,
                "model",
                ParameterValue::String("llama-7b".to_string()),
            )
            .unwrap();

        let run2 = backend.create_run(&exp_id).unwrap();
        backend
            .log_param(
                &run2,
                "model",
                ParameterValue::String("gpt-3.5".to_string()),
            )
            .unwrap();

        let filters = vec![ParamFilter {
            key: "model".to_string(),
            op: FilterOp::Contains,
            value: ParameterValue::String("llama".to_string()),
        }];

        let results = backend.search_runs_by_params(&filters).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, run1);
    }

    #[test]
    fn test_search_runs_no_filters() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();

        backend.create_run(&exp_id).unwrap();
        backend.create_run(&exp_id).unwrap();

        let results = backend.search_runs_by_params(&[]).unwrap();
        assert_eq!(results.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Span ID Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_set_and_get_span_id() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        backend.set_span_id(&run_id, "span-123").unwrap();
        let span_id = backend.get_span_id(&run_id).unwrap();
        assert_eq!(span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_get_span_id_none() {
        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        let span_id = backend.get_span_id(&run_id).unwrap();
        assert_eq!(span_id, None);
    }

    // -------------------------------------------------------------------------
    // ParameterValue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parameter_value_type_name() {
        assert_eq!(
            ParameterValue::String("test".to_string()).type_name(),
            "string"
        );
        assert_eq!(ParameterValue::Int(42).type_name(), "int");
        assert_eq!(ParameterValue::Float(3.14).type_name(), "float");
        assert_eq!(ParameterValue::Bool(true).type_name(), "bool");
        assert_eq!(ParameterValue::List(vec![]).type_name(), "list");
        assert_eq!(ParameterValue::Dict(HashMap::new()).type_name(), "dict");
    }

    #[test]
    fn test_parameter_value_json_roundtrip() {
        let values = vec![
            ParameterValue::String("hello".to_string()),
            ParameterValue::Int(42),
            ParameterValue::Float(3.14),
            ParameterValue::Bool(true),
            ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]),
        ];

        for value in values {
            let json = value.to_json();
            let parsed = ParameterValue::from_json(&json).unwrap();
            assert_eq!(value, parsed);
        }
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let mut backend = SqliteBackend::open_in_memory().unwrap();
        let exp_id = backend.create_experiment("test-exp", None).unwrap();
        let run_id = backend.create_run(&exp_id).unwrap();

        // Clone state for threads
        let state = backend.state.clone();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let state = state.clone();
                let run_id = run_id.clone();
                thread::spawn(move || {
                    let mut s = state.write().unwrap();
                    let point = MetricPoint::new(i, i as f64 * 0.1);
                    s.metrics
                        .entry(run_id.to_string())
                        .or_default()
                        .entry("loss".to_string())
                        .or_default()
                        .push(point);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = backend.get_metrics(&run_id, "loss").unwrap();
        assert_eq!(metrics.len(), 10);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_experiment_name_preserved(name in "[a-zA-Z][a-zA-Z0-9_-]{0,50}") {
            let mut backend = SqliteBackend::open_in_memory().unwrap();
            let exp_id = backend.create_experiment(&name, None).unwrap();
            let exp = backend.get_experiment(&exp_id).unwrap();
            prop_assert_eq!(exp.name, name);
        }

        #[test]
        fn prop_metric_values_preserved(values in prop::collection::vec(-1e10f64..1e10f64, 1..100)) {
            let mut backend = SqliteBackend::open_in_memory().unwrap();
            let exp_id = backend.create_experiment("test", None).unwrap();
            let run_id = backend.create_run(&exp_id).unwrap();

            for (step, value) in values.iter().enumerate() {
                if !value.is_nan() && !value.is_infinite() {
                    backend.log_metric(&run_id, "metric", step as u64, *value).unwrap();
                }
            }

            let metrics = backend.get_metrics(&run_id, "metric").unwrap();
            for (i, metric) in metrics.iter().enumerate() {
                let original = values[i];
                if !original.is_nan() && !original.is_infinite() {
                    prop_assert!((metric.value - original).abs() < 1e-10);
                }
            }
        }

        #[test]
        fn prop_artifact_sha256_deterministic(data in prop::collection::vec(any::<u8>(), 1..1000)) {
            let mut backend = SqliteBackend::open_in_memory().unwrap();
            let exp_id = backend.create_experiment("test", None).unwrap();
            let run_id = backend.create_run(&exp_id).unwrap();

            let sha1 = backend.log_artifact(&run_id, "file1", &data).unwrap();
            let sha2 = backend.log_artifact(&run_id, "file2", &data).unwrap();

            prop_assert_eq!(sha1, sha2);
        }

        #[test]
        fn prop_parameter_int_roundtrip(value in any::<i64>()) {
            let param = ParameterValue::Int(value);
            let json = param.to_json();
            let parsed = ParameterValue::from_json(&json).unwrap();
            prop_assert_eq!(param, parsed);
        }

        #[test]
        fn prop_parameter_float_roundtrip(value in -1e15f64..1e15f64) {
            if !value.is_nan() && !value.is_infinite() {
                let param = ParameterValue::Float(value);
                let json = param.to_json();
                let parsed = ParameterValue::from_json(&json).unwrap();
                if let ParameterValue::Float(v) = parsed {
                    // Use relative tolerance for large values
                    let tol = if value.abs() > 1.0 {
                        value.abs() * 1e-14
                    } else {
                        1e-14
                    };
                    prop_assert!((v - value).abs() < tol, "Expected {} == {} within tolerance", v, value);
                } else {
                    prop_assert!(false, "Expected Float");
                }
            }
        }

        #[test]
        fn prop_run_status_transitions_valid(
            complete_success in any::<bool>()
        ) {
            let mut backend = SqliteBackend::open_in_memory().unwrap();
            let exp_id = backend.create_experiment("test", None).unwrap();
            let run_id = backend.create_run(&exp_id).unwrap();

            // Pending -> Running
            prop_assert!(backend.start_run(&run_id).is_ok());
            prop_assert_eq!(backend.get_run_status(&run_id).unwrap(), RunStatus::Running);

            // Running -> Success/Failed
            let final_status = if complete_success {
                RunStatus::Success
            } else {
                RunStatus::Failed
            };
            prop_assert!(backend.complete_run(&run_id, final_status).is_ok());
            prop_assert_eq!(backend.get_run_status(&run_id).unwrap(), final_status);
        }
    }
}

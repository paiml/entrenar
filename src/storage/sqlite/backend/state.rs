//! SQLite internal state storage.
//!
//! Contains the in-memory state structure for experiments, runs, metrics, and artifacts.

use super::super::types::{ArtifactRef, Experiment, ParameterValue, Run};
use crate::storage::MetricPoint;
use std::collections::HashMap;

/// Internal storage for experiments, runs, metrics, and artifacts
#[derive(Debug, Default)]
pub(crate) struct SqliteState {
    pub(crate) experiments: HashMap<String, Experiment>,
    pub(crate) runs: HashMap<String, Run>,
    pub(crate) metrics: HashMap<String, HashMap<String, Vec<MetricPoint>>>, // run_id -> key -> points
    pub(crate) params: HashMap<String, HashMap<String, ParameterValue>>, // run_id -> key -> value
    pub(crate) artifacts: HashMap<String, Vec<ArtifactRef>>,             // run_id -> artifacts
    pub(crate) artifact_data: HashMap<String, Vec<u8>>,                  // sha256 -> data (CAS)
    pub(crate) span_ids: HashMap<String, String>,                        // run_id -> span_id
}

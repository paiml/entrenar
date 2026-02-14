//! Query operations for SQLite Backend.
//!
//! Contains search, list, and parameter filtering methods.

use super::backend::SqliteBackend;
use super::types::{Experiment, FilterOp, ParamFilter, ParameterValue, Run};
use crate::storage::{Result, StorageError};
use std::collections::HashMap;

impl SqliteBackend {
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
            // Float comparisons
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

            // Int comparisons
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gt) => v > fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lt) => v < fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gte) => v >= fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lte) => v <= fv,

            // String comparisons
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Contains) => {
                v.contains(fv.as_str())
            }
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::StartsWith) => {
                v.starts_with(fv.as_str())
            }

            // Bool comparisons
            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Ne) => v != fv,

            // Unsupported ops for same-type pairs
            (ParameterValue::Float(..), ParameterValue::Float(..), FilterOp::Contains | FilterOp::StartsWith)
            | (ParameterValue::Int(..), ParameterValue::Int(..), FilterOp::Contains | FilterOp::StartsWith)
            | (ParameterValue::String(..), ParameterValue::String(..), FilterOp::Gt | FilterOp::Lt | FilterOp::Gte | FilterOp::Lte)
            | (ParameterValue::Bool(..), ParameterValue::Bool(..), FilterOp::Gt | FilterOp::Lt | FilterOp::Gte | FilterOp::Lte | FilterOp::Contains | FilterOp::StartsWith) => false,

            // List and Dict do not support any filter operations
            (ParameterValue::List(..), ParameterValue::List(..), _)
            | (ParameterValue::Dict(..), ParameterValue::Dict(..), _) => false,

            // Cross-type comparisons are not supported
            (ParameterValue::Float(..), ParameterValue::Int(..) | ParameterValue::String(..) | ParameterValue::Bool(..) | ParameterValue::List(..) | ParameterValue::Dict(..), _)
            | (ParameterValue::Int(..), ParameterValue::Float(..) | ParameterValue::String(..) | ParameterValue::Bool(..) | ParameterValue::List(..) | ParameterValue::Dict(..), _)
            | (ParameterValue::String(..), ParameterValue::Float(..) | ParameterValue::Int(..) | ParameterValue::Bool(..) | ParameterValue::List(..) | ParameterValue::Dict(..), _)
            | (ParameterValue::Bool(..), ParameterValue::Float(..) | ParameterValue::Int(..) | ParameterValue::String(..) | ParameterValue::List(..) | ParameterValue::Dict(..), _)
            | (ParameterValue::List(..), ParameterValue::Float(..) | ParameterValue::Int(..) | ParameterValue::String(..) | ParameterValue::Bool(..) | ParameterValue::Dict(..), _)
            | (ParameterValue::Dict(..), ParameterValue::Float(..) | ParameterValue::Int(..) | ParameterValue::String(..) | ParameterValue::Bool(..) | ParameterValue::List(..), _) => false,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_matches_all_filter_ops() {
        let float_val = ParameterValue::Float(5.0);
        let float_filter = ParameterValue::Float(5.0);
        let string_val = ParameterValue::String("hello world".to_string());
        let string_filter = ParameterValue::String("hello".to_string());

        for op in &[
            FilterOp::Eq,
            FilterOp::Ne,
            FilterOp::Gt,
            FilterOp::Lt,
            FilterOp::Gte,
            FilterOp::Lte,
            FilterOp::Contains,
            FilterOp::StartsWith,
        ] {
            match op {
                FilterOp::Eq => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Ne => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Gt => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Lt => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Gte => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Lte => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Contains => {
                    // Unsupported for Float, but supported for String
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                    assert!(SqliteBackend::param_matches(&string_val, op, &string_filter));
                }
                FilterOp::StartsWith => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                    assert!(SqliteBackend::param_matches(&string_val, op, &string_filter));
                }
            }
        }
    }

    #[test]
    fn test_param_matches_cross_type_returns_false() {
        let float_val = ParameterValue::Float(42.0);
        let int_filter = ParameterValue::Int(42);
        // Cross-type comparisons always return false
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Eq, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Ne, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Gt, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Lt, &int_filter));
    }

    #[test]
    fn test_param_matches_list_and_dict_always_false() {
        let list_val = ParameterValue::List(vec![ParameterValue::Int(1)]);
        let list_filter = ParameterValue::List(vec![ParameterValue::Int(1)]);
        let dict_val = ParameterValue::Dict(HashMap::from([
            ("k".to_string(), ParameterValue::Int(1)),
        ]));
        let dict_filter = ParameterValue::Dict(HashMap::from([
            ("k".to_string(), ParameterValue::Int(1)),
        ]));

        // List and Dict do not support any filter operations - verify with match
        for val_and_filter in &[
            (&list_val, &list_filter),
            (&dict_val, &dict_filter),
        ] {
            let (val, filter) = val_and_filter;
            for op in &[FilterOp::Eq, FilterOp::Ne, FilterOp::Gt, FilterOp::Lt] {
                let result = SqliteBackend::param_matches(val, op, filter);
                match (val, filter) {
                    (ParameterValue::List(..), ParameterValue::List(..), ..) => {
                        assert!(!result, "List should not match with any op");
                    }
                    (ParameterValue::Dict(..), ParameterValue::Dict(..), ..) => {
                        assert!(!result, "Dict should not match with any op");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

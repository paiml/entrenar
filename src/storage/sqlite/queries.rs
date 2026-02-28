//! Query operations for SQLite Backend.
//!
//! Contains search, list, and parameter filtering methods.

use super::backend::SqliteBackend;
use super::types::{Experiment, FilterOp, ParamFilter, ParameterValue, Run};
use crate::storage::{Result, RunStatus, StorageError};
use chrono::{DateTime, Utc};
use rusqlite::params;
use std::collections::HashMap;

/// Parse a RunStatus from its stored string
fn str_to_status(s: &str) -> RunStatus {
    match s {
        "pending" => RunStatus::Pending,
        "running" => RunStatus::Running,
        "completed" => RunStatus::Success,
        "failed" => RunStatus::Failed,
        "cancelled" => RunStatus::Cancelled,
        _ => RunStatus::Failed,
    }
}

/// Parse an RFC3339 timestamp string, falling back to now
fn parse_timestamp(s: &str) -> DateTime<Utc> {
    s.parse().unwrap_or_else(|_| Utc::now())
}

impl SqliteBackend {
    /// Log a parameter for a run
    pub fn log_param(&self, run_id: &str, key: &str, value: ParameterValue) -> Result<()> {
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

        let value_json = value.to_json();
        let type_name = value.type_name();

        conn.execute(
            "INSERT OR REPLACE INTO params (run_id, key, value, type) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, key, value_json, type_name],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log param: {e}")))?;

        Ok(())
    }

    /// Log multiple parameters for a run
    pub fn log_params(&self, run_id: &str, params_map: HashMap<String, ParameterValue>) -> Result<()> {
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

        for (key, value) in &params_map {
            let value_json = value.to_json();
            let type_name = value.type_name();

            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value, type) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![run_id, key, value_json, type_name],
            )
            .map_err(|e| StorageError::Backend(format!("Failed to log param: {e}")))?;
        }

        Ok(())
    }

    /// Get parameters for a run
    pub fn get_params(&self, run_id: &str) -> Result<HashMap<String, ParameterValue>> {
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
            .prepare("SELECT key, value FROM params WHERE run_id = ?1")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare params query: {e}")))?;

        let rows = stmt
            .query_map([run_id], |row| {
                let key: String = row.get(0)?;
                let value_json: String = row.get(1)?;
                Ok((key, value_json))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query params: {e}")))?;

        let mut result = HashMap::new();
        for row in rows {
            let (key, value_json) =
                row.map_err(|e| StorageError::Backend(format!("Failed to read param row: {e}")))?;
            if let Some(value) = ParameterValue::from_json(&value_json) {
                result.insert(key, value);
            }
        }

        Ok(result)
    }

    /// Search runs by parameter filters
    pub fn search_runs_by_params(&self, filters: &[ParamFilter]) -> Result<Vec<Run>> {
        if filters.is_empty() {
            // No filters — return all runs
            return self.list_all_runs();
        }

        // Get all runs, then filter in Rust (same semantics as the HashMap version).
        // The filter logic requires cross-type matching rules that are complex to
        // express in SQL, so we fetch candidate runs and apply filters in memory.
        let all_runs = self.list_all_runs()?;
        let mut results = Vec::new();

        for run in all_runs {
            let run_params = self.get_params(&run.id)?;

            let matches = filters.iter().all(|filter| {
                if let Some(value) = run_params.get(&filter.key) {
                    Self::param_matches(value, &filter.op, &filter.value)
                } else {
                    false
                }
            });

            if matches {
                results.push(run);
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

            // Everything else (unsupported ops, cross-type, List/Dict) → false
            _ => false,
        }
    }

    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Result<Experiment> {
        let conn = self.lock_conn()?;

        let row = conn
            .query_row(
                "SELECT id, name, description, config, tags, created_at, updated_at FROM experiments WHERE id = ?1",
                [experiment_id],
                |row| {
                    let id: String = row.get(0)?;
                    let name: String = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let config_str: Option<String> = row.get(3)?;
                    let tags_str: Option<String> = row.get(4)?;
                    let created_str: String = row.get(5)?;
                    let updated_str: String = row.get(6)?;
                    Ok((id, name, description, config_str, tags_str, created_str, updated_str))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::ExperimentNotFound(experiment_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get experiment: {e}")),
            })?;

        let (id, name, description, config_str, tags_str, created_str, updated_str) = row;
        let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
        let tags: HashMap<String, String> =
            tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();

        Ok(Experiment {
            id,
            name,
            description,
            config,
            tags,
            created_at: parse_timestamp(&created_str),
            updated_at: parse_timestamp(&updated_str),
        })
    }

    /// Get a run by ID
    pub fn get_run(&self, run_id: &str) -> Result<Run> {
        self.query_run_by_id(run_id)
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Result<Vec<Experiment>> {
        let conn = self.lock_conn()?;

        let mut stmt = conn
            .prepare("SELECT id, name, description, config, tags, created_at, updated_at FROM experiments ORDER BY created_at DESC")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let name: String = row.get(1)?;
                let description: Option<String> = row.get(2)?;
                let config_str: Option<String> = row.get(3)?;
                let tags_str: Option<String> = row.get(4)?;
                let created_str: String = row.get(5)?;
                let updated_str: String = row.get(6)?;
                Ok((id, name, description, config_str, tags_str, created_str, updated_str))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to list experiments: {e}")))?;

        let mut result = Vec::new();
        for row in rows {
            let (id, name, description, config_str, tags_str, created_str, updated_str) =
                row.map_err(|e| StorageError::Backend(format!("Failed to read row: {e}")))?;
            let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
            let tags: HashMap<String, String> =
                tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            result.push(Experiment {
                id,
                name,
                description,
                config,
                tags,
                created_at: parse_timestamp(&created_str),
                updated_at: parse_timestamp(&updated_str),
            });
        }

        Ok(result)
    }

    /// List runs for an experiment
    pub fn list_runs(&self, experiment_id: &str) -> Result<Vec<Run>> {
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

        let mut stmt = conn
            .prepare("SELECT id, experiment_id, status, start_time, end_time, tags FROM runs WHERE experiment_id = ?1 ORDER BY start_time")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        Self::collect_runs_from_stmt(&mut stmt, params![experiment_id])
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn query_run_by_id(&self, run_id: &str) -> Result<Run> {
        let conn = self.lock_conn()?;

        let row = conn
            .query_row(
                "SELECT id, experiment_id, status, start_time, end_time, tags FROM runs WHERE id = ?1",
                [run_id],
                |row| {
                    let id: String = row.get(0)?;
                    let experiment_id: String = row.get(1)?;
                    let status_str: String = row.get(2)?;
                    let start_str: Option<String> = row.get(3)?;
                    let end_str: Option<String> = row.get(4)?;
                    let tags_str: Option<String> = row.get(5)?;
                    Ok((id, experiment_id, status_str, start_str, end_str, tags_str))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run: {e}")),
            })?;

        Ok(Self::row_to_run(row))
    }

    fn list_all_runs(&self) -> Result<Vec<Run>> {
        let conn = self.lock_conn()?;

        let mut stmt = conn
            .prepare("SELECT id, experiment_id, status, start_time, end_time, tags FROM runs ORDER BY start_time")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        Self::collect_runs_from_stmt(&mut stmt, [])
    }

    fn collect_runs_from_stmt<P: rusqlite::Params>(
        stmt: &mut rusqlite::Statement<'_>,
        params: P,
    ) -> Result<Vec<Run>> {
        let rows = stmt
            .query_map(params, |row| {
                let id: String = row.get(0)?;
                let experiment_id: String = row.get(1)?;
                let status_str: String = row.get(2)?;
                let start_str: Option<String> = row.get(3)?;
                let end_str: Option<String> = row.get(4)?;
                let tags_str: Option<String> = row.get(5)?;
                Ok((id, experiment_id, status_str, start_str, end_str, tags_str))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query runs: {e}")))?;

        let mut result = Vec::new();
        for row in rows {
            let tuple =
                row.map_err(|e| StorageError::Backend(format!("Failed to read run row: {e}")))?;
            result.push(Self::row_to_run(tuple));
        }
        Ok(result)
    }

    fn row_to_run(
        row: (String, String, String, Option<String>, Option<String>, Option<String>),
    ) -> Run {
        let (id, experiment_id, status_str, start_str, end_str, tags_str) = row;
        let start_time = start_str.map_or_else(Utc::now, |s| parse_timestamp(&s));
        let end_time = end_str.map(|s| parse_timestamp(&s));
        let tags: HashMap<String, String> =
            tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();

        Run {
            id,
            experiment_id,
            status: str_to_status(&status_str),
            start_time,
            end_time,
            params: HashMap::new(),
            tags,
        }
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
        let dict_val =
            ParameterValue::Dict(HashMap::from([("k".to_string(), ParameterValue::Int(1))]));
        let dict_filter =
            ParameterValue::Dict(HashMap::from([("k".to_string(), ParameterValue::Int(1))]));

        // List and Dict do not support any filter operations - verify with match
        let test_cases: Vec<(&ParameterValue, &ParameterValue, &FilterOp)> = vec![
            (&list_val, &list_filter, &FilterOp::Eq),
            (&list_val, &list_filter, &FilterOp::Ne),
            (&dict_val, &dict_filter, &FilterOp::Eq),
            (&dict_val, &dict_filter, &FilterOp::Ne),
        ];

        for (val, filt, op) in &test_cases {
            let result = SqliteBackend::param_matches(val, op, filt);
            match (val, filt, op) {
                (ParameterValue::List(..), ParameterValue::List(..), _) => {
                    assert!(!result, "List should not match with any op");
                }
                (ParameterValue::Dict(..), ParameterValue::Dict(..), _) => {
                    assert!(!result, "Dict should not match with any op");
                }
                _ => unreachable!(),
            }
        }
    }
}

//! Additional coverage tests for filter operations and edge cases.

use std::collections::HashMap;

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::types::{FilterOp, ParamFilter, ParameterValue};
use crate::storage::{ExperimentStorage, RunStatus, StorageError};

// -------------------------------------------------------------------------
// Float filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_float_ne() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

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
        op: FilterOp::Ne,
        value: ParameterValue::Float(0.001),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_float_lt() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

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
        op: FilterOp::Lt,
        value: ParameterValue::Float(0.005),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

#[test]
fn test_filter_op_float_gte() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "lr", ParameterValue::Float(0.01))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Float(0.01),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_float_lte() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "lr", ParameterValue::Float(0.01))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Float(0.01),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

// -------------------------------------------------------------------------
// Int filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_int_eq() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_ne() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let run2 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run2, "epochs", ParameterValue::Int(200))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_int_gt() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let run2 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run2, "epochs", ParameterValue::Int(200))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Gt,
        value: ParameterValue::Int(150),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_int_lt() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Int(150),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_gte() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Gte,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_int_lte() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "epochs", ParameterValue::Int(100))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Lte,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

// -------------------------------------------------------------------------
// String filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_string_eq() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "model", ParameterValue::String("llama".to_string()))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_filter_op_string_ne() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "model", ParameterValue::String("llama".to_string()))
        .unwrap();

    let run2 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run2, "model", ParameterValue::String("gpt".to_string()))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

#[test]
fn test_filter_op_string_starts_with() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

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
        .log_param(&run2, "model", ParameterValue::String("gpt-3".to_string()))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "model".to_string(),
        op: FilterOp::StartsWith,
        value: ParameterValue::String("llama".to_string()),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

// -------------------------------------------------------------------------
// Bool filter operations
// -------------------------------------------------------------------------

#[test]
fn test_filter_op_bool_eq() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "use_cuda", ParameterValue::Bool(true))
        .unwrap();

    let run2 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run2, "use_cuda", ParameterValue::Bool(false))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "use_cuda".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Bool(true),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run1);
}

#[test]
fn test_filter_op_bool_ne() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "use_cuda", ParameterValue::Bool(true))
        .unwrap();

    let run2 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run2, "use_cuda", ParameterValue::Bool(false))
        .unwrap();

    let filters = vec![ParamFilter {
        key: "use_cuda".to_string(),
        op: FilterOp::Ne,
        value: ParameterValue::Bool(true),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, run2);
}

// -------------------------------------------------------------------------
// Edge cases
// -------------------------------------------------------------------------

#[test]
fn test_filter_type_mismatch() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "value", ParameterValue::Int(100))
        .unwrap();

    // Try to filter int with float - should not match
    let filters = vec![ParamFilter {
        key: "value".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Float(100.0),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_filter_missing_key() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    let run1 = backend.create_run(&exp_id).unwrap();
    backend
        .log_param(&run1, "lr", ParameterValue::Float(0.001))
        .unwrap();

    // Filter by key that doesn't exist
    let filters = vec![ParamFilter {
        key: "epochs".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Int(100),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_filter_run_without_params() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();

    // Create run without any params
    backend.create_run(&exp_id).unwrap();

    let filters = vec![ParamFilter {
        key: "lr".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::Float(0.001),
    }];

    let results = backend.search_runs_by_params(&filters).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_log_params_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let mut params = HashMap::new();
    params.insert("lr".to_string(), ParameterValue::Float(0.001));
    let result = backend.log_params("nonexistent", params);
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_get_params_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.get_params("nonexistent");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_list_runs_experiment_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.list_runs("nonexistent");
    assert!(matches!(result, Err(StorageError::ExperimentNotFound(_))));
}

#[test]
fn test_get_artifact_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.get_artifact_data("nonexistent_sha256");
    assert!(result.is_err());
}

#[test]
fn test_list_artifacts_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.list_artifacts("nonexistent");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_get_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.get_run("nonexistent");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_complete_run_not_found() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.complete_run("nonexistent", RunStatus::Success);
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_get_metrics_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.get_metrics("nonexistent", "loss");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_log_artifact_run_not_found() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.log_artifact("nonexistent", "file.bin", b"data");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_set_span_id_run_not_found() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.set_span_id("nonexistent", "span-123");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_get_span_id_run_not_found() {
    let backend = SqliteBackend::open_in_memory().unwrap();
    let result = backend.get_span_id("nonexistent");
    assert!(matches!(result, Err(StorageError::RunNotFound(_))));
}

#[test]
fn test_run_struct_fields() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();
    let run_id = backend.create_run(&exp_id).unwrap();

    let run = backend.get_run(&run_id).unwrap();
    assert_eq!(run.id, run_id);
    assert_eq!(run.experiment_id, exp_id);
    assert_eq!(run.status, RunStatus::Pending);
    assert!(run.end_time.is_none());
    assert!(run.params.is_empty());
    assert!(run.tags.is_empty());
}

#[test]
fn test_experiment_struct_fields() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("my-experiment", None).unwrap();

    let exp = backend.get_experiment(&exp_id).unwrap();
    assert_eq!(exp.id, exp_id);
    assert_eq!(exp.name, "my-experiment");
    assert!(exp.description.is_none());
    assert!(exp.config.is_none());
    assert!(exp.tags.is_empty());
}

#[test]
fn test_artifact_ref_fields() {
    let mut backend = SqliteBackend::open_in_memory().unwrap();
    let exp_id = backend.create_experiment("test", None).unwrap();
    let run_id = backend.create_run(&exp_id).unwrap();

    let data = b"test artifact content";
    let sha = backend.log_artifact(&run_id, "test.bin", data).unwrap();

    let artifacts = backend.list_artifacts(&run_id).unwrap();
    assert_eq!(artifacts.len(), 1);

    let artifact = &artifacts[0];
    assert_eq!(artifact.run_id, run_id);
    assert_eq!(artifact.path, "test.bin");
    assert_eq!(artifact.size_bytes, data.len() as u64);
    assert_eq!(artifact.sha256, sha);
}

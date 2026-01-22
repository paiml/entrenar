//! Tests for SQLite Backend.
//!
//! Contains all test modules: unit tests, property tests, and coverage tests.

use super::super::types::{FilterOp, ParamFilter, ParameterValue};
use super::SqliteBackend;
use crate::storage::{ExperimentStorage, MetricPoint, RunStatus, StorageError};
use std::collections::HashMap;

// =============================================================================
// SqliteBackend Basic Tests
// =============================================================================

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
                    .entry(run_id.clone())
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

// =============================================================================
// Property Tests
// =============================================================================

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

// =============================================================================
// Additional Coverage Tests
// =============================================================================

mod coverage_tests {
    use super::*;

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
}

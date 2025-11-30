//! Integration tests for Run struct with Renacer tracing (ENT-002)

use std::sync::{Arc, Mutex};

use entrenar::run::{Run, TracingConfig};
use entrenar::storage::{ExperimentStorage, InMemoryStorage, RunStatus};

fn setup_storage() -> (Arc<Mutex<InMemoryStorage>>, String) {
    let mut storage = InMemoryStorage::new();
    let exp_id = storage.create_experiment("test-experiment", None).unwrap();
    (Arc::new(Mutex::new(storage)), exp_id)
}

#[test]
fn test_run_creates_span() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::default();

    let run = Run::new(&exp_id, storage, config).unwrap();

    // Run::new() should create a valid span_id when tracing is enabled
    assert!(run.span_id().is_some());
    let span_id = run.span_id().unwrap();
    assert!(span_id.starts_with("span-"));
}

#[test]
fn test_run_no_span_when_disabled() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let run = Run::new(&exp_id, storage, config).unwrap();

    // No span when tracing is disabled
    assert!(run.span_id().is_none());
}

#[test]
fn test_run_stores_span_in_storage() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::default();

    let run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let span_id = run.span_id().unwrap().to_string();
    let run_id = run.run_id().to_string();

    // Span ID should be persisted in storage
    let stored_span = storage
        .lock()
        .unwrap()
        .get_span_id(&run_id)
        .unwrap()
        .unwrap();
    assert_eq!(stored_span, span_id);
}

#[test]
fn test_run_log_metric_auto_step() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();

    // Log metrics without explicit step
    run.log_metric("loss", 1.0).unwrap();
    run.log_metric("loss", 0.8).unwrap();
    run.log_metric("loss", 0.6).unwrap();

    // Steps should auto-increment: 0, 1, 2
    let metrics = storage
        .lock()
        .unwrap()
        .get_metrics(run.run_id(), "loss")
        .unwrap();
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].step, 0);
    assert_eq!(metrics[1].step, 1);
    assert_eq!(metrics[2].step, 2);
}

#[test]
fn test_run_log_metric_at_explicit_step() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();

    // Log at explicit steps
    run.log_metric_at("accuracy", 0, 0.5).unwrap();
    run.log_metric_at("accuracy", 100, 0.75).unwrap();
    run.log_metric_at("accuracy", 200, 0.9).unwrap();

    let metrics = storage
        .lock()
        .unwrap()
        .get_metrics(run.run_id(), "accuracy")
        .unwrap();
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].step, 0);
    assert_eq!(metrics[1].step, 100);
    assert_eq!(metrics[2].step, 200);
}

#[test]
fn test_run_multiple_metric_keys() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();

    // Log different metrics - each has independent step counter
    run.log_metric("loss", 1.0).unwrap();
    run.log_metric("accuracy", 0.5).unwrap();
    run.log_metric("loss", 0.8).unwrap();

    assert_eq!(run.current_step("loss"), 2);
    assert_eq!(run.current_step("accuracy"), 1);
}

#[test]
fn test_run_finish_success() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let run_id = run.run_id().to_string();

    run.finish(RunStatus::Success).unwrap();

    let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
    assert_eq!(status, RunStatus::Success);
}

#[test]
fn test_run_finish_failed() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let run_id = run.run_id().to_string();

    run.finish(RunStatus::Failed).unwrap();

    let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
    assert_eq!(status, RunStatus::Failed);
}

#[test]
fn test_run_finish_cancelled() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let run_id = run.run_id().to_string();

    run.finish(RunStatus::Cancelled).unwrap();

    let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
    assert_eq!(status, RunStatus::Cancelled);
}

#[test]
fn test_tracing_config_builder() {
    let config = TracingConfig::default()
        .with_otlp_export()
        .with_golden_trace_path("/tmp/golden");

    assert!(config.tracing_enabled);
    assert!(config.export_otlp);
    assert_eq!(
        config.golden_trace_path,
        Some(std::path::PathBuf::from("/tmp/golden"))
    );
}

#[test]
fn test_run_is_running_after_new() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::disabled();

    let run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let run_id = run.run_id().to_string();

    // Run::new() should start the run automatically
    let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
    assert_eq!(status, RunStatus::Running);
}

#[test]
fn test_run_accessors() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::default();

    let run = Run::new(&exp_id, storage, config).unwrap();

    assert!(!run.is_finished());
    assert!(run.run_id().starts_with("run-"));
    assert!(run.tracing_config().tracing_enabled);
}

#[test]
fn test_full_training_lifecycle() {
    let (storage, exp_id) = setup_storage();
    let config = TracingConfig::default();

    let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();
    let run_id = run.run_id().to_string();

    // Simulate training loop
    for epoch in 0..3 {
        let loss = 1.0 / (epoch as f64 + 1.0);
        let accuracy = 0.5 + (epoch as f64 * 0.15);

        run.log_metric("train_loss", loss).unwrap();
        run.log_metric("train_accuracy", accuracy).unwrap();
    }

    // Verify span exists
    assert!(run.span_id().is_some());

    // Complete the run
    run.finish(RunStatus::Success).unwrap();

    // Verify final state
    let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
    assert_eq!(status, RunStatus::Success);

    let loss_metrics = storage
        .lock()
        .unwrap()
        .get_metrics(&run_id, "train_loss")
        .unwrap();
    assert_eq!(loss_metrics.len(), 3);

    let acc_metrics = storage
        .lock()
        .unwrap()
        .get_metrics(&run_id, "train_accuracy")
        .unwrap();
    assert_eq!(acc_metrics.len(), 3);
}

#[cfg(feature = "monitor")]
mod trueno_integration_tests {
    use super::*;
    use entrenar::storage::TruenoBackend;

    #[test]
    fn test_run_with_trueno_backend() {
        let mut backend = TruenoBackend::new();
        let exp_id = backend.create_experiment("trueno-run-test", None).unwrap();
        let storage = Arc::new(Mutex::new(backend));

        let config = TracingConfig::default();
        let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();

        run.log_metric("loss", 0.5).unwrap();
        run.log_metric("loss", 0.4).unwrap();

        let run_id = run.run_id().to_string();
        run.finish(RunStatus::Success).unwrap();

        let status = storage.lock().unwrap().get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Success);
    }
}

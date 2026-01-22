//! Span ID tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::ExperimentStorage;

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

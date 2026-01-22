//! Artifact tests.

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::ExperimentStorage;

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

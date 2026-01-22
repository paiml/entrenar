//! Basic tests for SqliteBackend.

use crate::storage::sqlite::backend::SqliteBackend;

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

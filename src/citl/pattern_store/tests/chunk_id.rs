//! ChunkId tests.

use super::*;

#[test]
fn test_chunk_id_unique() {
    let id1 = ChunkId::new();
    let id2 = ChunkId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_chunk_id_display() {
    let id = ChunkId::new();
    let display = format!("{id}");
    assert!(!display.is_empty());
    assert!(display.contains('-')); // UUID format
}

#[test]
fn test_chunk_id_serialization() {
    let id = ChunkId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

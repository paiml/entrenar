//! Tests for KNN decision path.

use super::*;
use crate::monitor::inference::path::traits::DecisionPath;

#[test]
fn test_knn_path_new() {
    let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);

    assert_eq!(path.k(), 3);
    assert_eq!(path.neighbor_indices, vec![0, 1, 2]);
    assert_eq!(path.distances, vec![0.1, 0.2, 0.3]);
    assert_eq!(path.neighbor_labels, vec![0, 0, 1]);
    assert_eq!(path.prediction, 0.0);
    assert!(path.weighted_votes.is_none());
    // Votes should be computed
    assert!(!path.votes.is_empty());
}

#[test]
fn test_knn_path_with_weighted_votes() {
    let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0)
        .with_weighted_votes(vec![0.5, 0.3, 0.2]);

    assert!(path.weighted_votes.is_some());
    assert_eq!(path.weighted_votes.unwrap(), vec![0.5, 0.3, 0.2]);
}

#[test]
fn test_knn_path_k() {
    let path = KNNPath::new(
        vec![0, 1, 2, 3, 4],
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vec![0, 0, 1, 1, 2],
        1.0,
    );

    assert_eq!(path.k(), 5);
}

#[test]
fn test_knn_path_explain() {
    let path = KNNPath::new(vec![10, 20, 30], vec![0.5, 1.0, 1.5], vec![0, 1, 0], 0.0);

    let explanation = path.explain();
    assert!(explanation.contains("KNN Prediction: 0.0000 (k=3)"));
    assert!(explanation.contains("Nearest neighbors:"));
    assert!(explanation.contains("#1: idx=10, label=0, distance=0.5000"));
    assert!(explanation.contains("#2: idx=20, label=1, distance=1.0000"));
    assert!(explanation.contains("#3: idx=30, label=0, distance=1.5000"));
    assert!(explanation.contains("Vote distribution:"));
}

#[test]
fn test_knn_path_feature_contributions() {
    let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);

    // KNN doesn't have feature contributions
    assert!(path.feature_contributions().is_empty());
}

#[test]
fn test_knn_path_confidence() {
    // All same class - 100% confidence
    let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 0], 0.0);
    assert!((path.confidence() - 1.0).abs() < 1e-6);

    // Mixed classes - lower confidence
    let path2 = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 1, 2], 0.0);
    assert!((path2.confidence() - (1.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn test_knn_path_confidence_empty_votes() {
    let mut path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);
    path.votes.clear();

    assert_eq!(path.confidence(), 0.0);
}

#[test]
fn test_knn_path_serialization_roundtrip() {
    let path = KNNPath::new(vec![5, 10, 15], vec![0.25, 0.5, 0.75], vec![1, 1, 0], 1.0);

    let bytes = path.to_bytes();
    let restored = KNNPath::from_bytes(&bytes).unwrap();

    assert_eq!(restored.neighbor_indices, path.neighbor_indices);
    assert_eq!(restored.distances, path.distances);
    assert_eq!(restored.neighbor_labels, path.neighbor_labels);
    assert_eq!(restored.prediction, path.prediction);
}

#[test]
fn test_knn_path_serialization_with_weighted_votes() {
    let path = KNNPath::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0)
        .with_weighted_votes(vec![0.6, 0.3, 0.1]);

    let bytes = path.to_bytes();
    let restored = KNNPath::from_bytes(&bytes).unwrap();

    assert!(restored.weighted_votes.is_some());
    assert_eq!(restored.weighted_votes.unwrap(), vec![0.6, 0.3, 0.1]);
}

#[test]
fn test_knn_path_from_bytes_insufficient_data() {
    use crate::monitor::inference::path::traits::PathError;

    let bytes = vec![1, 0, 0]; // Too short

    let result = KNNPath::from_bytes(&bytes);
    assert!(result.is_err());
    match result {
        Err(PathError::InsufficientData { expected, actual }) => {
            assert_eq!(expected, 5);
            assert_eq!(actual, 3);
        }
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_knn_path_from_bytes_version_mismatch() {
    use crate::monitor::inference::path::traits::PathError;

    let bytes = vec![2, 0, 0, 0, 0]; // Version 2 instead of 1

    let result = KNNPath::from_bytes(&bytes);
    assert!(result.is_err());
    match result {
        Err(PathError::VersionMismatch { expected, actual }) => {
            assert_eq!(expected, 1);
            assert_eq!(actual, 2);
        }
        _ => panic!("Expected VersionMismatch error"),
    }
}

#[test]
fn test_knn_path_from_bytes_truncated_neighbor_indices() {
    use crate::monitor::inference::path::traits::PathError;

    // Version 1, k=3, but no neighbor data
    let bytes = vec![1, 3, 0, 0, 0];

    let result = KNNPath::from_bytes(&bytes);
    assert!(result.is_err());
    match result {
        Err(PathError::InsufficientData { .. }) => {}
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_knn_path_clone() {
    let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);

    let cloned = path.clone();
    assert_eq!(cloned.neighbor_indices, path.neighbor_indices);
    assert_eq!(cloned.distances, path.distances);
    assert_eq!(cloned.neighbor_labels, path.neighbor_labels);
}

#[test]
fn test_knn_path_debug() {
    let path = KNNPath::new(vec![0], vec![0.5], vec![1], 1.0);

    let debug_str = format!("{:?}", path);
    assert!(debug_str.contains("KNNPath"));
    assert!(debug_str.contains("neighbor_indices"));
}

#[test]
fn test_knn_path_serde_json() {
    let path = KNNPath::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3], vec![0, 1, 0], 0.0);

    let json = serde_json::to_string(&path).unwrap();
    let restored: KNNPath = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.neighbor_indices, path.neighbor_indices);
    assert_eq!(restored.prediction, path.prediction);
}

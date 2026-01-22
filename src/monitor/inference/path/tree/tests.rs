//! Tests for tree-based decision paths

use super::*;
use crate::monitor::inference::path::traits::{DecisionPath, PathError};

#[test]
fn test_tree_path_new() {
    let splits = vec![
        TreeSplit {
            feature_idx: 0,
            threshold: 35.0,
            went_left: true,
            n_samples: 1000,
        },
        TreeSplit {
            feature_idx: 1,
            threshold: 50000.0,
            went_left: false,
            n_samples: 600,
        },
    ];
    let leaf = LeafInfo {
        prediction: 1.0,
        n_samples: 250,
        class_distribution: Some(vec![0.08, 0.92]),
    };

    let path = TreePath::new(splits, leaf);
    assert_eq!(path.depth(), 2);
}

#[test]
fn test_tree_path_explain() {
    let splits = vec![TreeSplit {
        feature_idx: 0,
        threshold: 35.0,
        went_left: true,
        n_samples: 1000,
    }];
    let leaf = LeafInfo {
        prediction: 1.0,
        n_samples: 250,
        class_distribution: Some(vec![0.1, 0.9]),
    };

    let path = TreePath::new(splits, leaf);
    let explanation = path.explain();
    assert!(explanation.contains("Decision Path (depth=1)"));
    assert!(explanation.contains("feature[0]"));
    assert!(explanation.contains("LEAF"));
}

#[test]
fn test_tree_path_serialization_roundtrip() {
    let splits = vec![
        TreeSplit {
            feature_idx: 0,
            threshold: 35.0,
            went_left: true,
            n_samples: 1000,
        },
        TreeSplit {
            feature_idx: 1,
            threshold: 50000.0,
            went_left: false,
            n_samples: 600,
        },
    ];
    let leaf = LeafInfo {
        prediction: 0.92,
        n_samples: 250,
        class_distribution: Some(vec![0.08, 0.92]),
    };

    let path = TreePath::new(splits, leaf)
        .with_gini(vec![0.5, 0.3, 0.1])
        .with_contributions(vec![0.2, 0.5, 0.3]);

    let bytes = path.to_bytes();
    let restored = TreePath::from_bytes(&bytes).expect("Failed to deserialize");

    assert_eq!(path.splits.len(), restored.splits.len());
    assert_eq!(path.leaf.n_samples, restored.leaf.n_samples);
    assert!((path.leaf.prediction - restored.leaf.prediction).abs() < 1e-6);
    assert_eq!(path.gini_path.len(), restored.gini_path.len());
    assert_eq!(path.contributions.len(), restored.contributions.len());
}

#[test]
fn test_tree_path_confidence() {
    let leaf = LeafInfo {
        prediction: 1.0,
        n_samples: 100,
        class_distribution: Some(vec![0.1, 0.9]),
    };
    let path = TreePath::new(vec![], leaf);
    assert!((path.confidence() - 0.9).abs() < 1e-6);
}

#[test]
fn test_tree_path_insufficient_data_at_start() {
    let result = TreePath::from_bytes(&[1u8, 0, 0]);
    assert!(matches!(result, Err(PathError::InsufficientData { .. })));
}

#[test]
fn test_tree_path_invalid_version() {
    let result = TreePath::from_bytes(&[2u8, 0, 0, 0, 0]);
    assert!(matches!(result, Err(PathError::VersionMismatch { .. })));
}

#[test]
fn test_tree_path_insufficient_data_in_splits() {
    // Version 1, 1 split, but not enough data for the split
    let mut bytes = vec![1u8];
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_splits = 1
                                                  // Not enough data for split
    let result = TreePath::from_bytes(&bytes);
    assert!(matches!(result, Err(PathError::InsufficientData { .. })));
}

#[test]
fn test_tree_path_confidence_without_distribution() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf);
    let confidence = path.confidence();
    // 1.0 - 1.0 / (100 + 1) = approximately 0.99
    assert!(confidence > 0.98);
    assert!(confidence < 1.0);
}

#[test]
fn test_tree_path_explain_went_right() {
    let splits = vec![TreeSplit {
        feature_idx: 0,
        threshold: 35.0,
        went_left: false, // went right
        n_samples: 100,
    }];
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 50,
        class_distribution: None,
    };
    let path = TreePath::new(splits, leaf);
    let explanation = path.explain();
    assert!(explanation.contains("NO")); // went_left=false shows "NO"
    assert!(explanation.contains(">"));
}

#[test]
fn test_tree_path_serialization_without_class_distribution() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf);
    let bytes = path.to_bytes();
    let restored = TreePath::from_bytes(&bytes).expect("Failed to deserialize");
    assert!(restored.leaf.class_distribution.is_none());
}

#[test]
fn test_tree_split_clone() {
    let split = TreeSplit {
        feature_idx: 5,
        threshold: 2.5,
        went_left: true,
        n_samples: 500,
    };
    let cloned = split.clone();
    assert_eq!(split.feature_idx, cloned.feature_idx);
    assert_eq!(split.threshold, cloned.threshold);
    assert_eq!(split.went_left, cloned.went_left);
    assert_eq!(split.n_samples, cloned.n_samples);
}

#[test]
fn test_leaf_info_clone() {
    let leaf = LeafInfo {
        prediction: 0.75,
        n_samples: 200,
        class_distribution: Some(vec![0.25, 0.75]),
    };
    let cloned = leaf.clone();
    assert_eq!(leaf.prediction, cloned.prediction);
    assert_eq!(leaf.n_samples, cloned.n_samples);
    assert_eq!(leaf.class_distribution, cloned.class_distribution);
}

#[test]
fn test_tree_path_with_gini() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf).with_gini(vec![0.5, 0.3, 0.1]);
    assert_eq!(path.gini_path, vec![0.5, 0.3, 0.1]);
}

#[test]
fn test_tree_path_with_contributions() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf).with_contributions(vec![0.2, 0.5, 0.3]);
    assert_eq!(path.feature_contributions(), &[0.2, 0.5, 0.3]);
}

#[test]
fn test_tree_path_empty_splits() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf);
    assert_eq!(path.depth(), 0);
    let explanation = path.explain();
    assert!(explanation.contains("Decision Path (depth=0)"));
    assert!(explanation.contains("LEAF"));
}

#[test]
fn test_tree_path_feature_contributions_empty() {
    let leaf = LeafInfo {
        prediction: 0.5,
        n_samples: 100,
        class_distribution: None,
    };
    let path = TreePath::new(vec![], leaf);
    assert!(path.feature_contributions().is_empty());
}

//! Tests for NeuralPath

use super::*;

#[test]
fn test_neural_path_new() {
    let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.87, 0.92);
    assert_eq!(path.input_gradient.len(), 3);
    assert_eq!(path.prediction, 0.87);
    assert_eq!(path.confidence, 0.92);
}

#[test]
fn test_neural_path_top_salient() {
    let path = NeuralPath::new(vec![0.1, -0.5, 0.3], 0.0, 0.0);
    let top = path.top_salient_features(2);
    assert_eq!(top[0].0, 1); // -0.5 has highest absolute value
    assert_eq!(top[1].0, 2); // 0.3 is second
}

#[test]
fn test_neural_path_serialization_roundtrip() {
    let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.87, 0.92)
        .with_activations(vec![vec![0.5, 0.6], vec![0.7, 0.8]])
        .with_attention(vec![vec![0.1, 0.9]])
        .with_integrated_gradients(vec![0.15, -0.25, 0.35]);

    let bytes = path.to_bytes();
    let restored = NeuralPath::from_bytes(&bytes).expect("Failed to deserialize");

    assert_eq!(path.input_gradient.len(), restored.input_gradient.len());
    assert!((path.prediction - restored.prediction).abs() < 1e-6);
    assert!((path.confidence - restored.confidence).abs() < 1e-6);
    assert!(restored.activations.is_some());
    assert!(restored.attention_weights.is_some());
    assert!(restored.integrated_gradients.is_some());
}

#[test]
fn test_neural_path_feature_contributions() {
    let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.0, 0.0);
    assert_eq!(path.feature_contributions(), &[0.1, -0.2, 0.3]);

    let path_with_ig =
        NeuralPath::new(vec![0.1, -0.2, 0.3], 0.0, 0.0).with_integrated_gradients(vec![0.5, 0.5]);
    assert_eq!(path_with_ig.feature_contributions(), &[0.5, 0.5]);
}

#[test]
fn test_neural_path_invalid_version() {
    let result = NeuralPath::from_bytes(&[2u8, 0, 0, 0, 0]);
    assert!(matches!(result, Err(PathError::VersionMismatch { .. })));
}

#[test]
fn test_neural_path_insufficient_data() {
    let result = NeuralPath::from_bytes(&[1u8, 0, 0]);
    assert!(matches!(result, Err(PathError::InsufficientData { .. })));
}

#[test]
fn test_neural_path_explain_with_ig() {
    let path = NeuralPath::new(vec![0.1], 0.5, 0.9).with_integrated_gradients(vec![0.2, 0.3, 0.5]);
    let explanation = path.explain();
    assert!(explanation.contains("Integrated gradients"));
    assert!(explanation.contains("3 features"));
}

#[test]
fn test_neural_path_explain_with_attention() {
    let path = NeuralPath::new(vec![0.1], 0.5, 0.9).with_attention(vec![vec![0.5, 0.5]]);
    let explanation = path.explain();
    assert!(explanation.contains("Attention weights"));
}

#[test]
fn test_neural_path_serialization_minimal() {
    let path = NeuralPath::new(vec![0.1, 0.2], 0.5, 0.9);
    let bytes = path.to_bytes();
    let restored = NeuralPath::from_bytes(&bytes).expect("Failed to deserialize");
    assert!(restored.activations.is_none());
    assert!(restored.attention_weights.is_none());
    assert!(restored.integrated_gradients.is_none());
}

#[test]
fn test_neural_path_with_activations() {
    let path =
        NeuralPath::new(vec![0.1], 0.5, 0.9).with_activations(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert!(path.activations.is_some());
    let activations = path.activations.unwrap();
    assert_eq!(activations.len(), 2);
    assert_eq!(activations[0], vec![1.0, 2.0]);
    assert_eq!(activations[1], vec![3.0, 4.0]);
}

#[test]
fn test_neural_path_confidence_method() {
    let path = NeuralPath::new(vec![0.1], 0.5, 0.85);
    assert_eq!(path.confidence(), 0.85);
}

#[test]
fn test_neural_path_explain_basic() {
    let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.75, 0.90);
    let explanation = path.explain();
    assert!(explanation.contains("Neural Network Prediction"));
    assert!(explanation.contains("0.75"));
    assert!(explanation.contains("90.0%"));
    assert!(explanation.contains("Top salient input features"));
}

#[test]
fn test_neural_path_top_salient_features_empty() {
    let path = NeuralPath::new(vec![], 0.5, 0.9);
    let top = path.top_salient_features(5);
    assert!(top.is_empty());
}

#[test]
fn test_neural_path_top_salient_features_more_than_available() {
    let path = NeuralPath::new(vec![0.1, 0.2], 0.5, 0.9);
    let top = path.top_salient_features(10);
    assert_eq!(top.len(), 2);
}

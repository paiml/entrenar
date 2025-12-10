//! Counterfactual Explanations (ENT-104)
//!
//! "What would have changed the decision?"

use serde::{Deserialize, Serialize};

/// A single feature change in a counterfactual
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureChange {
    /// Feature index
    pub feature_idx: usize,
    /// Optional human-readable feature name
    pub feature_name: Option<String>,
    /// Original value
    pub original_value: f32,
    /// Counterfactual value (value that would flip the decision)
    pub counterfactual_value: f32,
    /// Change amount (counterfactual - original)
    pub delta: f32,
}

impl FeatureChange {
    /// Create a new feature change
    pub fn new(feature_idx: usize, original: f32, counterfactual: f32) -> Self {
        Self {
            feature_idx,
            feature_name: None,
            original_value: original,
            counterfactual_value: counterfactual,
            delta: counterfactual - original,
        }
    }

    /// Set the feature name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.feature_name = Some(name.into());
        self
    }

    /// Get the absolute change
    pub fn abs_delta(&self) -> f32 {
        self.delta.abs()
    }
}

/// Counterfactual explanation for a decision
///
/// Answers: "What minimal change would have flipped the decision?"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counterfactual {
    /// Original input that produced the decision
    pub original_input: Vec<f32>,
    /// Original decision/class
    pub original_decision: usize,
    /// Original confidence
    pub original_confidence: f32,
    /// Modified input that would flip the decision
    pub counterfactual_input: Vec<f32>,
    /// The alternative decision
    pub alternative_decision: usize,
    /// Alternative confidence
    pub alternative_confidence: f32,
    /// Which features changed and by how much
    pub changes: Vec<FeatureChange>,
    /// L1 distance (sparsity of changes)
    pub sparsity: f32,
    /// L2 distance (magnitude of changes)
    pub distance: f32,
}

impl Counterfactual {
    /// Create a new counterfactual
    pub fn new(
        original_input: Vec<f32>,
        original_decision: usize,
        original_confidence: f32,
        counterfactual_input: Vec<f32>,
        alternative_decision: usize,
        alternative_confidence: f32,
    ) -> Self {
        assert_eq!(
            original_input.len(),
            counterfactual_input.len(),
            "Input dimensions must match"
        );

        let mut changes = Vec::new();
        let mut l1 = 0.0f32;
        let mut l2 = 0.0f32;

        for i in 0..original_input.len() {
            let delta = counterfactual_input[i] - original_input[i];
            if delta.abs() > 1e-6 {
                changes.push(FeatureChange::new(
                    i,
                    original_input[i],
                    counterfactual_input[i],
                ));
                l1 += delta.abs();
                l2 += delta * delta;
            }
        }

        Self {
            original_input,
            original_decision,
            original_confidence,
            counterfactual_input,
            alternative_decision,
            alternative_confidence,
            changes,
            sparsity: l1,
            distance: l2.sqrt(),
        }
    }

    /// Generate natural language explanation
    pub fn explain(&self) -> String {
        let mut explanation = format!(
            "Original decision: {} (confidence: {:.1}%)\n",
            self.original_decision,
            self.original_confidence * 100.0
        );
        explanation.push_str(&format!(
            "Alternative decision: {} (confidence: {:.1}%)\n",
            self.alternative_decision,
            self.alternative_confidence * 100.0
        ));
        explanation.push_str(&format!(
            "\nThe decision would have been {} if:\n",
            self.alternative_decision
        ));

        // Sort changes by absolute delta (most impactful first)
        let mut sorted_changes = self.changes.clone();
        sorted_changes.sort_by(|a, b| {
            b.abs_delta()
                .partial_cmp(&a.abs_delta())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for change in sorted_changes.iter().take(5) {
            let sign = if change.delta >= 0.0 { "+" } else { "" };
            let default_name = format!("feature[{}]", change.feature_idx);
            let name = change.feature_name.as_deref().unwrap_or(&default_name);
            explanation.push_str(&format!(
                "  - {}: {:.4} → {:.4} ({}{:.4})\n",
                name, change.original_value, change.counterfactual_value, sign, change.delta
            ));
        }

        if self.changes.len() > 5 {
            explanation.push_str(&format!(
                "  ... and {} more changes\n",
                self.changes.len() - 5
            ));
        }

        explanation.push_str(&format!("\nSparsity (L1): {:.4}\n", self.sparsity));
        explanation.push_str(&format!("Distance (L2): {:.4}\n", self.distance));

        explanation
    }

    /// Number of features that changed
    pub fn n_changes(&self) -> usize {
        self.changes.len()
    }

    /// Check if this is a valid counterfactual (decision actually flipped)
    pub fn is_valid(&self) -> bool {
        self.original_decision != self.alternative_decision
    }

    /// Set feature names for all changes
    pub fn with_feature_names(mut self, names: &[String]) -> Self {
        for change in &mut self.changes {
            if change.feature_idx < names.len() {
                change.feature_name = Some(names[change.feature_idx].clone());
            }
        }
        self
    }

    /// Convert to binary format
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Original decision info
        bytes.extend_from_slice(&(self.original_decision as u32).to_le_bytes());
        bytes.extend_from_slice(&self.original_confidence.to_le_bytes());
        bytes.extend_from_slice(&(self.alternative_decision as u32).to_le_bytes());
        bytes.extend_from_slice(&self.alternative_confidence.to_le_bytes());

        // Original input
        bytes.extend_from_slice(&(self.original_input.len() as u32).to_le_bytes());
        for v in &self.original_input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Counterfactual input
        for v in &self.counterfactual_input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Changes (compact: only store changed indices and deltas)
        bytes.extend_from_slice(&(self.changes.len() as u32).to_le_bytes());
        for change in &self.changes {
            bytes.extend_from_slice(&(change.feature_idx as u32).to_le_bytes());
            bytes.extend_from_slice(&change.original_value.to_le_bytes());
            bytes.extend_from_slice(&change.counterfactual_value.to_le_bytes());
            bytes.extend_from_slice(&change.delta.to_le_bytes());

            // Feature name (length-prefixed)
            if let Some(name) = &change.feature_name {
                bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
                bytes.extend_from_slice(name.as_bytes());
            } else {
                bytes.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        // Metrics
        bytes.extend_from_slice(&self.sparsity.to_le_bytes());
        bytes.extend_from_slice(&self.distance.to_le_bytes());

        bytes
    }

    /// Reconstruct from binary format
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CounterfactualError> {
        if bytes.len() < 21 {
            return Err(CounterfactualError::InsufficientData {
                expected: 21,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(CounterfactualError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;

        let original_decision = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let original_confidence = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let alternative_decision = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let alternative_confidence = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let n_features = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        // Original input
        let mut original_input = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            if offset + 4 > bytes.len() {
                return Err(CounterfactualError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let v = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            original_input.push(v);
        }

        // Counterfactual input
        let mut counterfactual_input = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            if offset + 4 > bytes.len() {
                return Err(CounterfactualError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let v = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            counterfactual_input.push(v);
        }

        // Changes
        if offset + 4 > bytes.len() {
            return Err(CounterfactualError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_changes = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut changes = Vec::with_capacity(n_changes);
        for _ in 0..n_changes {
            if offset + 20 > bytes.len() {
                return Err(CounterfactualError::InsufficientData {
                    expected: offset + 20,
                    actual: bytes.len(),
                });
            }

            let feature_idx = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let original_value = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;

            let counterfactual_value = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;

            let delta = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;

            let name_len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let feature_name = if name_len > 0 {
                if offset + name_len > bytes.len() {
                    return Err(CounterfactualError::InsufficientData {
                        expected: offset + name_len,
                        actual: bytes.len(),
                    });
                }
                let name = String::from_utf8_lossy(&bytes[offset..offset + name_len]).to_string();
                offset += name_len;
                Some(name)
            } else {
                None
            };

            changes.push(FeatureChange {
                feature_idx,
                feature_name,
                original_value,
                counterfactual_value,
                delta,
            });
        }

        // Metrics
        if offset + 8 > bytes.len() {
            return Err(CounterfactualError::InsufficientData {
                expected: offset + 8,
                actual: bytes.len(),
            });
        }
        let sparsity = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let distance = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);

        Ok(Self {
            original_input,
            original_decision,
            original_confidence,
            counterfactual_input,
            alternative_decision,
            alternative_confidence,
            changes,
            sparsity,
            distance,
        })
    }
}

/// Error type for counterfactual operations
#[derive(Debug, Clone, PartialEq)]
pub enum CounterfactualError {
    /// Insufficient data
    InsufficientData { expected: usize, actual: usize },
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
}

impl std::fmt::Display for CounterfactualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterfactualError::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected {expected}, got {actual}")
            }
            CounterfactualError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for CounterfactualError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_change_new() {
        let change = FeatureChange::new(0, 1.0, 2.0);
        assert_eq!(change.feature_idx, 0);
        assert_eq!(change.original_value, 1.0);
        assert_eq!(change.counterfactual_value, 2.0);
        assert_eq!(change.delta, 1.0);
    }

    #[test]
    fn test_feature_change_with_name() {
        let change = FeatureChange::new(0, 1.0, 2.0).with_name("income");
        assert_eq!(change.feature_name, Some("income".to_string()));
    }

    #[test]
    fn test_feature_change_abs_delta() {
        let change_pos = FeatureChange::new(0, 1.0, 2.0);
        let change_neg = FeatureChange::new(0, 2.0, 1.0);
        assert_eq!(change_pos.abs_delta(), 1.0);
        assert_eq!(change_neg.abs_delta(), 1.0);
    }

    #[test]
    fn test_counterfactual_new() {
        let cf = Counterfactual::new(vec![1.0, 2.0, 3.0], 0, 0.9, vec![1.5, 2.0, 4.0], 1, 0.85);

        assert_eq!(cf.original_decision, 0);
        assert_eq!(cf.alternative_decision, 1);
        assert_eq!(cf.n_changes(), 2); // features 0 and 2 changed
        assert!(cf.is_valid());
    }

    #[test]
    fn test_counterfactual_metrics() {
        let cf = Counterfactual::new(vec![0.0, 0.0], 0, 0.9, vec![3.0, 4.0], 1, 0.85);

        // L1 = |3| + |4| = 7
        assert!((cf.sparsity - 7.0).abs() < 1e-6);
        // L2 = sqrt(9 + 16) = 5
        assert!((cf.distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_counterfactual_explain() {
        let cf = Counterfactual::new(vec![45000.0, 0.42], 0, 0.7, vec![52000.0, 0.35], 1, 0.8)
            .with_feature_names(&["income".to_string(), "debt_ratio".to_string()]);

        let explanation = cf.explain();
        assert!(explanation.contains("Original decision: 0"));
        assert!(explanation.contains("Alternative decision: 1"));
        assert!(explanation.contains("income"));
        assert!(explanation.contains("debt_ratio"));
    }

    #[test]
    fn test_counterfactual_serialization_roundtrip() {
        let cf = Counterfactual::new(vec![1.0, 2.0, 3.0], 0, 0.9, vec![1.5, 2.0, 4.0], 1, 0.85)
            .with_feature_names(&[
                "feature_a".to_string(),
                "feature_b".to_string(),
                "feature_c".to_string(),
            ]);

        let bytes = cf.to_bytes();
        let restored = Counterfactual::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(cf.original_decision, restored.original_decision);
        assert_eq!(cf.alternative_decision, restored.alternative_decision);
        assert!((cf.original_confidence - restored.original_confidence).abs() < 1e-6);
        assert_eq!(cf.original_input.len(), restored.original_input.len());
        assert_eq!(cf.changes.len(), restored.changes.len());
        assert!((cf.sparsity - restored.sparsity).abs() < 1e-6);
        assert!((cf.distance - restored.distance).abs() < 1e-6);
    }

    #[test]
    fn test_counterfactual_no_changes() {
        let cf = Counterfactual::new(
            vec![1.0, 2.0, 3.0],
            0,
            0.9,
            vec![1.0, 2.0, 3.0], // Same input
            0,                   // Same decision
            0.9,
        );

        assert_eq!(cf.n_changes(), 0);
        assert!(!cf.is_valid()); // Decision didn't flip
    }

    #[test]
    fn test_counterfactual_error_display() {
        let err = CounterfactualError::InsufficientData {
            expected: 100,
            actual: 50,
        };
        assert!(err.to_string().contains("expected 100"));

        let err = CounterfactualError::VersionMismatch {
            expected: 1,
            actual: 2,
        };
        assert!(err.to_string().contains("Version mismatch"));
    }

    #[test]
    fn test_counterfactual_insufficient_data() {
        let result = Counterfactual::from_bytes(&[0; 10]);
        assert!(matches!(
            result,
            Err(CounterfactualError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_counterfactual_version_mismatch() {
        let mut bytes = vec![2u8]; // Invalid version
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let result = Counterfactual::from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(CounterfactualError::VersionMismatch { .. })
        ));
    }
}

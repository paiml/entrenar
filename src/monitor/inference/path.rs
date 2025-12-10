//! Decision Path Types (ENT-102)
//!
//! Model-specific decision paths for explainability.

use serde::{Deserialize, Serialize};

/// Common interface for all decision paths
pub trait DecisionPath: Clone + Send + Sync + 'static {
    /// Human-readable explanation
    fn explain(&self) -> String;

    /// Feature importance scores (contribution of each feature)
    fn feature_contributions(&self) -> &[f32];

    /// Confidence in this decision (0.0 - 1.0)
    fn confidence(&self) -> f32;

    /// Compact binary representation
    fn to_bytes(&self) -> Vec<u8>;

    /// Reconstruct from binary representation
    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError>
    where
        Self: Sized;
}

/// Error type for path operations
#[derive(Debug, Clone, PartialEq)]
pub enum PathError {
    /// Invalid binary format
    InvalidFormat(String),
    /// Insufficient data
    InsufficientData { expected: usize, actual: usize },
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            PathError::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected {expected}, got {actual}")
            }
            PathError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PathError {}

// =============================================================================
// LinearPath - Decision path for linear models
// =============================================================================

/// Decision path for linear regression/logistic regression
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearPath {
    /// Per-feature contributions: coefficient[i] * input[i]
    pub contributions: Vec<f32>,
    /// Bias term contribution
    pub intercept: f32,
    /// Raw prediction before activation
    pub logit: f32,
    /// Final prediction
    pub prediction: f32,
    /// For classification: probability (sigmoid/softmax output)
    pub probability: Option<f32>,
}

impl LinearPath {
    /// Create a new linear path
    pub fn new(contributions: Vec<f32>, intercept: f32, logit: f32, prediction: f32) -> Self {
        Self {
            contributions,
            intercept,
            logit,
            prediction,
            probability: None,
        }
    }

    /// Set probability for classification
    pub fn with_probability(mut self, prob: f32) -> Self {
        self.probability = Some(prob);
        self
    }

    /// Get top k features by absolute contribution
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .contributions
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }
}

impl DecisionPath for LinearPath {
    fn explain(&self) -> String {
        let mut explanation = format!("Prediction: {:.4}", self.prediction);

        if let Some(prob) = self.probability {
            explanation.push_str(&format!(" (probability: {:.1}%)", prob * 100.0));
        }

        explanation.push_str("\nTop contributing features:");

        for (idx, contrib) in self.top_features(5) {
            let sign = if contrib >= 0.0 { "+" } else { "" };
            explanation.push_str(&format!("\n  - feature[{idx}]: {sign}{contrib:.4}"));
        }

        explanation.push_str(&format!("\nIntercept: {:.4}", self.intercept));
        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.contributions
    }

    fn confidence(&self) -> f32 {
        self.probability.unwrap_or_else(|| {
            // For regression, use inverse of prediction variance as confidence proxy
            1.0 / (1.0 + self.logit.abs())
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        // Version 1 format:
        // [0]: version (1)
        // [1..5]: n_features (u32 LE)
        // [5..9]: intercept (f32 LE)
        // [9..13]: logit (f32 LE)
        // [13..17]: prediction (f32 LE)
        // [17]: has_probability (1 byte)
        // [18..22]: probability if present (f32 LE)
        // [22+]: contributions (n_features * 4 bytes)

        let n_features = self.contributions.len() as u32;
        let has_prob = self.probability.is_some();

        let mut bytes = Vec::with_capacity(22 + self.contributions.len() * 4);
        bytes.push(1); // version
        bytes.extend_from_slice(&n_features.to_le_bytes());
        bytes.extend_from_slice(&self.intercept.to_le_bytes());
        bytes.extend_from_slice(&self.logit.to_le_bytes());
        bytes.extend_from_slice(&self.prediction.to_le_bytes());
        bytes.push(u8::from(has_prob));

        if let Some(prob) = self.probability {
            bytes.extend_from_slice(&prob.to_le_bytes());
        } else {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        for c in &self.contributions {
            bytes.extend_from_slice(&c.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 22 {
            return Err(PathError::InsufficientData {
                expected: 22,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let n_features = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let expected_len = 22 + n_features * 4;

        if bytes.len() < expected_len {
            return Err(PathError::InsufficientData {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let intercept = f32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let logit = f32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
        let prediction = f32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);
        let has_prob = bytes[17] != 0;
        let prob_value = f32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);

        let probability = if has_prob { Some(prob_value) } else { None };

        let mut contributions = Vec::with_capacity(n_features);
        for i in 0..n_features {
            let offset = 22 + i * 4;
            let c = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            contributions.push(c);
        }

        Ok(Self {
            contributions,
            intercept,
            logit,
            prediction,
            probability,
        })
    }
}

// =============================================================================
// TreePath - Decision path for tree-based models
// =============================================================================

/// A single split decision in a tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSplit {
    /// Feature index used for split
    pub feature_idx: usize,
    /// Threshold value
    pub threshold: f32,
    /// Direction taken (true = left, false = right)
    pub went_left: bool,
    /// Samples in node before split (from training)
    pub n_samples: usize,
}

/// Information about the leaf node reached
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeafInfo {
    /// Predicted class or value
    pub prediction: f32,
    /// Samples in training that reached this leaf
    pub n_samples: usize,
    /// Class distribution (for classification)
    pub class_distribution: Option<Vec<f32>>,
}

/// Decision path for tree-based models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreePath {
    /// Sequence of splits taken
    pub splits: Vec<TreeSplit>,
    /// Leaf node statistics
    pub leaf: LeafInfo,
    /// Gini impurity at each node (optional)
    pub gini_path: Vec<f32>,
    /// Feature contributions (computed from mean decrease)
    contributions: Vec<f32>,
}

impl TreePath {
    /// Create a new tree path
    pub fn new(splits: Vec<TreeSplit>, leaf: LeafInfo) -> Self {
        let gini_path = Vec::new();
        let contributions = Vec::new();
        Self {
            splits,
            leaf,
            gini_path,
            contributions,
        }
    }

    /// Set Gini impurity path
    pub fn with_gini(mut self, gini_path: Vec<f32>) -> Self {
        self.gini_path = gini_path;
        self
    }

    /// Set feature contributions
    pub fn with_contributions(mut self, contributions: Vec<f32>) -> Self {
        self.contributions = contributions;
        self
    }

    /// Get depth of the tree path
    pub fn depth(&self) -> usize {
        self.splits.len()
    }
}

impl DecisionPath for TreePath {
    fn explain(&self) -> String {
        let depth = self.depth();
        let mut explanation = format!("Decision Path (depth={depth}):\n");

        for (i, split) in self.splits.iter().enumerate() {
            let direction = if split.went_left { "YES" } else { "NO" };
            let comparison = if split.went_left { "<=" } else { ">" };
            let feature_idx = split.feature_idx;
            let threshold = split.threshold;
            let n_samples = split.n_samples;
            explanation.push_str(&format!(
                "  Node {i}: feature[{feature_idx}] {comparison} {threshold:.4}? {direction} (n={n_samples})\n"
            ));
        }

        let prediction = self.leaf.prediction;
        let n_samples = self.leaf.n_samples;
        explanation.push_str(&format!(
            "  LEAF -> prediction={prediction:.4}, n_samples={n_samples}\n"
        ));

        if let Some(dist) = &self.leaf.class_distribution {
            explanation.push_str("         class_distribution: [");
            explanation.push_str(
                &dist
                    .iter()
                    .map(|p| format!("{p:.2}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            explanation.push_str("]\n");
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.contributions
    }

    fn confidence(&self) -> f32 {
        if let Some(dist) = &self.leaf.class_distribution {
            // Max probability as confidence
            dist.iter().copied().fold(0.0f32, f32::max)
        } else {
            // For regression, use sample size as proxy
            1.0 - 1.0 / (self.leaf.n_samples as f32 + 1.0)
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        // Serialize using bincode-like format
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Number of splits
        let n_splits = self.splits.len() as u32;
        bytes.extend_from_slice(&n_splits.to_le_bytes());

        // Each split
        for split in &self.splits {
            bytes.extend_from_slice(&(split.feature_idx as u32).to_le_bytes());
            bytes.extend_from_slice(&split.threshold.to_le_bytes());
            bytes.push(u8::from(split.went_left));
            bytes.extend_from_slice(&(split.n_samples as u32).to_le_bytes());
        }

        // Leaf info
        bytes.extend_from_slice(&self.leaf.prediction.to_le_bytes());
        bytes.extend_from_slice(&(self.leaf.n_samples as u32).to_le_bytes());

        // Class distribution
        let has_dist = self.leaf.class_distribution.is_some();
        bytes.push(u8::from(has_dist));
        if let Some(dist) = &self.leaf.class_distribution {
            bytes.extend_from_slice(&(dist.len() as u32).to_le_bytes());
            for p in dist {
                bytes.extend_from_slice(&p.to_le_bytes());
            }
        }

        // Gini path
        bytes.extend_from_slice(&(self.gini_path.len() as u32).to_le_bytes());
        for g in &self.gini_path {
            bytes.extend_from_slice(&g.to_le_bytes());
        }

        // Contributions
        bytes.extend_from_slice(&(self.contributions.len() as u32).to_le_bytes());
        for c in &self.contributions {
            bytes.extend_from_slice(&c.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;
        let n_splits = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut splits = Vec::with_capacity(n_splits);
        for _ in 0..n_splits {
            if offset + 13 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 13,
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

            let threshold = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;

            let went_left = bytes[offset] != 0;
            offset += 1;

            let n_samples = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            splits.push(TreeSplit {
                feature_idx,
                threshold,
                went_left,
                n_samples,
            });
        }

        // Leaf info
        if offset + 9 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 9,
                actual: bytes.len(),
            });
        }

        let prediction = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let n_samples = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let has_dist = bytes[offset] != 0;
        offset += 1;

        let class_distribution = if has_dist {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_classes = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut dist = Vec::with_capacity(n_classes);
            for _ in 0..n_classes {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let p = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                offset += 4;
                dist.push(p);
            }
            Some(dist)
        } else {
            None
        };

        let leaf = LeafInfo {
            prediction,
            n_samples,
            class_distribution,
        };

        // Gini path
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_gini = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut gini_path = Vec::with_capacity(n_gini);
        for _ in 0..n_gini {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let g = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            gini_path.push(g);
        }

        // Contributions
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_contrib = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut contributions = Vec::with_capacity(n_contrib);
        for _ in 0..n_contrib {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let c = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            contributions.push(c);
        }

        Ok(Self {
            splits,
            leaf,
            gini_path,
            contributions,
        })
    }
}

// =============================================================================
// ForestPath - Decision path for ensemble models
// =============================================================================

/// Decision path for ensemble models (Random Forest, Gradient Boosting)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForestPath {
    /// Individual tree paths
    pub tree_paths: Vec<TreePath>,
    /// Per-tree predictions
    pub tree_predictions: Vec<f32>,
    /// Aggregated prediction
    pub ensemble_prediction: f32,
    /// Agreement ratio among trees (0.0 - 1.0)
    pub tree_agreement: f32,
    /// Feature importance (averaged across trees)
    pub feature_importance: Vec<f32>,
}

impl ForestPath {
    /// Create a new forest path
    pub fn new(tree_paths: Vec<TreePath>, tree_predictions: Vec<f32>) -> Self {
        let ensemble_prediction = if tree_predictions.is_empty() {
            0.0
        } else {
            tree_predictions.iter().sum::<f32>() / tree_predictions.len() as f32
        };

        let tree_agreement = Self::compute_agreement(&tree_predictions);
        let feature_importance = Vec::new();

        Self {
            tree_paths,
            tree_predictions,
            ensemble_prediction,
            tree_agreement,
            feature_importance,
        }
    }

    /// Compute agreement ratio among trees
    fn compute_agreement(predictions: &[f32]) -> f32 {
        if predictions.len() < 2 {
            return 1.0;
        }

        let mean = predictions.iter().sum::<f32>() / predictions.len() as f32;
        let variance =
            predictions.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / predictions.len() as f32;

        // Convert variance to agreement (higher variance = lower agreement)
        1.0 / (1.0 + variance)
    }

    /// Set feature importance
    pub fn with_feature_importance(mut self, importance: Vec<f32>) -> Self {
        self.feature_importance = importance;
        self
    }

    /// Number of trees in the ensemble
    pub fn n_trees(&self) -> usize {
        self.tree_paths.len()
    }
}

impl DecisionPath for ForestPath {
    fn explain(&self) -> String {
        let mut explanation = format!(
            "Ensemble Prediction: {:.4} (n_trees={}, agreement={:.1}%)\n",
            self.ensemble_prediction,
            self.n_trees(),
            self.tree_agreement * 100.0
        );

        explanation.push_str("\nTree predictions:\n");
        for (i, pred) in self.tree_predictions.iter().enumerate() {
            explanation.push_str(&format!("  Tree {i}: {pred:.4}\n"));
        }

        if !self.feature_importance.is_empty() {
            explanation.push_str("\nTop features by importance:\n");
            let mut indexed: Vec<(usize, f32)> = self
                .feature_importance
                .iter()
                .enumerate()
                .map(|(i, &imp)| (i, imp))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, imp) in indexed.iter().take(5) {
                explanation.push_str(&format!("  feature[{idx}]: {imp:.4}\n"));
            }
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.feature_importance
    }

    fn confidence(&self) -> f32 {
        self.tree_agreement
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Number of trees
        bytes.extend_from_slice(&(self.tree_paths.len() as u32).to_le_bytes());

        // Each tree path
        for tree_path in &self.tree_paths {
            let tree_bytes = tree_path.to_bytes();
            bytes.extend_from_slice(&(tree_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&tree_bytes);
        }

        // Tree predictions
        bytes.extend_from_slice(&(self.tree_predictions.len() as u32).to_le_bytes());
        for pred in &self.tree_predictions {
            bytes.extend_from_slice(&pred.to_le_bytes());
        }

        // Ensemble prediction
        bytes.extend_from_slice(&self.ensemble_prediction.to_le_bytes());
        bytes.extend_from_slice(&self.tree_agreement.to_le_bytes());

        // Feature importance
        bytes.extend_from_slice(&(self.feature_importance.len() as u32).to_le_bytes());
        for imp in &self.feature_importance {
            bytes.extend_from_slice(&imp.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;

        // Number of trees
        let n_trees = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut tree_paths = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let tree_len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + tree_len > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + tree_len,
                    actual: bytes.len(),
                });
            }

            let tree_path = TreePath::from_bytes(&bytes[offset..offset + tree_len])?;
            tree_paths.push(tree_path);
            offset += tree_len;
        }

        // Tree predictions
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_preds = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut tree_predictions = Vec::with_capacity(n_preds);
        for _ in 0..n_preds {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let pred = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            tree_predictions.push(pred);
        }

        // Ensemble prediction and agreement
        if offset + 8 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 8,
                actual: bytes.len(),
            });
        }
        let ensemble_prediction = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let tree_agreement = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        // Feature importance
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_imp = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut feature_importance = Vec::with_capacity(n_imp);
        for _ in 0..n_imp {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let imp = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            feature_importance.push(imp);
        }

        Ok(Self {
            tree_paths,
            tree_predictions,
            ensemble_prediction,
            tree_agreement,
            feature_importance,
        })
    }
}

// =============================================================================
// KNNPath - Decision path for K-Nearest Neighbors
// =============================================================================

/// Decision path for KNN
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KNNPath {
    /// Indices of k nearest neighbors
    pub neighbor_indices: Vec<usize>,
    /// Distances to neighbors
    pub distances: Vec<f32>,
    /// Labels of neighbors
    pub neighbor_labels: Vec<usize>,
    /// Vote distribution: (class, count)
    pub votes: Vec<(usize, usize)>,
    /// Weighted vote (if distance-weighted)
    pub weighted_votes: Option<Vec<f32>>,
    /// Final prediction
    pub prediction: f32,
}

impl KNNPath {
    /// Create a new KNN path
    pub fn new(
        neighbor_indices: Vec<usize>,
        distances: Vec<f32>,
        neighbor_labels: Vec<usize>,
        prediction: f32,
    ) -> Self {
        // Compute vote distribution
        let mut vote_map = std::collections::HashMap::new();
        for &label in &neighbor_labels {
            *vote_map.entry(label).or_insert(0usize) += 1;
        }
        let votes: Vec<(usize, usize)> = vote_map.into_iter().collect();

        Self {
            neighbor_indices,
            distances,
            neighbor_labels,
            votes,
            weighted_votes: None,
            prediction,
        }
    }

    /// Set weighted votes
    pub fn with_weighted_votes(mut self, weights: Vec<f32>) -> Self {
        self.weighted_votes = Some(weights);
        self
    }

    /// Number of neighbors
    pub fn k(&self) -> usize {
        self.neighbor_indices.len()
    }
}

impl DecisionPath for KNNPath {
    fn explain(&self) -> String {
        let prediction = self.prediction;
        let k = self.k();
        let mut explanation = format!("KNN Prediction: {prediction:.4} (k={k})\n");

        explanation.push_str("\nNearest neighbors:\n");
        for i in 0..self.k() {
            let rank = i + 1;
            let idx = self.neighbor_indices[i];
            let label = self.neighbor_labels[i];
            let distance = self.distances[i];
            explanation.push_str(&format!(
                "  #{rank}: idx={idx}, label={label}, distance={distance:.4}\n"
            ));
        }

        explanation.push_str("\nVote distribution:\n");
        for (class, count) in &self.votes {
            explanation.push_str(&format!("  class {class}: {count} votes\n"));
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        // KNN doesn't have per-feature contributions in the same sense
        // Return empty slice
        &[]
    }

    fn confidence(&self) -> f32 {
        // Confidence based on voting margin
        if self.votes.is_empty() {
            return 0.0;
        }

        let max_votes = self.votes.iter().map(|(_, c)| *c).max().unwrap_or(0);
        max_votes as f32 / self.k() as f32
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // K value
        let k = self.neighbor_indices.len() as u32;
        bytes.extend_from_slice(&k.to_le_bytes());

        // Neighbor indices
        for idx in &self.neighbor_indices {
            bytes.extend_from_slice(&(*idx as u32).to_le_bytes());
        }

        // Distances
        for d in &self.distances {
            bytes.extend_from_slice(&d.to_le_bytes());
        }

        // Labels
        for l in &self.neighbor_labels {
            bytes.extend_from_slice(&(*l as u32).to_le_bytes());
        }

        // Votes
        bytes.extend_from_slice(&(self.votes.len() as u32).to_le_bytes());
        for (class, count) in &self.votes {
            bytes.extend_from_slice(&(*class as u32).to_le_bytes());
            bytes.extend_from_slice(&(*count as u32).to_le_bytes());
        }

        // Weighted votes
        let has_weights = self.weighted_votes.is_some();
        bytes.push(u8::from(has_weights));
        if let Some(weights) = &self.weighted_votes {
            bytes.extend_from_slice(&(weights.len() as u32).to_le_bytes());
            for w in weights {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }

        // Prediction
        bytes.extend_from_slice(&self.prediction.to_le_bytes());

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;

        let k = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        // Neighbor indices
        let mut neighbor_indices = Vec::with_capacity(k);
        for _ in 0..k {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let idx = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;
            neighbor_indices.push(idx);
        }

        // Distances
        let mut distances = Vec::with_capacity(k);
        for _ in 0..k {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let d = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            distances.push(d);
        }

        // Labels
        let mut neighbor_labels = Vec::with_capacity(k);
        for _ in 0..k {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let l = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;
            neighbor_labels.push(l);
        }

        // Votes
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let n_votes = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut votes = Vec::with_capacity(n_votes);
        for _ in 0..n_votes {
            if offset + 8 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 8,
                    actual: bytes.len(),
                });
            }
            let class = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;
            let count = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;
            votes.push((class, count));
        }

        // Weighted votes
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_weights = bytes[offset] != 0;
        offset += 1;

        let weighted_votes = if has_weights {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_weights = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut weights = Vec::with_capacity(n_weights);
            for _ in 0..n_weights {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let w = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                offset += 4;
                weights.push(w);
            }
            Some(weights)
        } else {
            None
        };

        // Prediction
        if offset + 4 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 4,
                actual: bytes.len(),
            });
        }
        let prediction = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);

        Ok(Self {
            neighbor_indices,
            distances,
            neighbor_labels,
            votes,
            weighted_votes,
            prediction,
        })
    }
}

// =============================================================================
// NeuralPath - Decision path for neural networks
// =============================================================================

/// Decision path for neural networks (gradient-based)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralPath {
    /// Input gradient (saliency map)
    pub input_gradient: Vec<f32>,
    /// Layer activations (optional, feature-gated for memory)
    pub activations: Option<Vec<Vec<f32>>>,
    /// Attention weights (for transformers)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Integrated gradients attribution
    pub integrated_gradients: Option<Vec<f32>>,
    /// Final prediction
    pub prediction: f32,
    /// Confidence (softmax probability)
    pub confidence: f32,
}

impl NeuralPath {
    /// Create a new neural path
    pub fn new(input_gradient: Vec<f32>, prediction: f32, confidence: f32) -> Self {
        Self {
            input_gradient,
            activations: None,
            attention_weights: None,
            integrated_gradients: None,
            prediction,
            confidence,
        }
    }

    /// Set layer activations
    pub fn with_activations(mut self, activations: Vec<Vec<f32>>) -> Self {
        self.activations = Some(activations);
        self
    }

    /// Set attention weights
    pub fn with_attention(mut self, attention: Vec<Vec<f32>>) -> Self {
        self.attention_weights = Some(attention);
        self
    }

    /// Set integrated gradients
    pub fn with_integrated_gradients(mut self, ig: Vec<f32>) -> Self {
        self.integrated_gradients = Some(ig);
        self
    }

    /// Get top salient features by absolute gradient
    pub fn top_salient_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .input_gradient
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g))
            .collect();

        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }
}

impl DecisionPath for NeuralPath {
    fn explain(&self) -> String {
        let mut explanation = format!(
            "Neural Network Prediction: {:.4} (confidence: {:.1}%)\n",
            self.prediction,
            self.confidence * 100.0
        );

        explanation.push_str("\nTop salient input features (by gradient):\n");
        for (idx, grad) in self.top_salient_features(5) {
            let sign = if grad >= 0.0 { "+" } else { "" };
            explanation.push_str(&format!("  input[{idx}]: {sign}{grad:.6}\n"));
        }

        if let Some(ig) = &self.integrated_gradients {
            explanation.push_str("\nIntegrated gradients available (");
            let len = ig.len();
            explanation.push_str(&format!("{len} features)\n"));
        }

        if self.attention_weights.is_some() {
            explanation.push_str("\nAttention weights available\n");
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        self.integrated_gradients
            .as_deref()
            .unwrap_or(&self.input_gradient)
    }

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Input gradient
        bytes.extend_from_slice(&(self.input_gradient.len() as u32).to_le_bytes());
        for g in &self.input_gradient {
            bytes.extend_from_slice(&g.to_le_bytes());
        }

        // Prediction and confidence
        bytes.extend_from_slice(&self.prediction.to_le_bytes());
        bytes.extend_from_slice(&self.confidence.to_le_bytes());

        // Activations
        let has_activations = self.activations.is_some();
        bytes.push(u8::from(has_activations));
        if let Some(activations) = &self.activations {
            bytes.extend_from_slice(&(activations.len() as u32).to_le_bytes());
            for layer in activations {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Attention weights
        let has_attention = self.attention_weights.is_some();
        bytes.push(u8::from(has_attention));
        if let Some(attention) = &self.attention_weights {
            bytes.extend_from_slice(&(attention.len() as u32).to_le_bytes());
            for layer in attention {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Integrated gradients
        let has_ig = self.integrated_gradients.is_some();
        bytes.push(u8::from(has_ig));
        if let Some(ig) = &self.integrated_gradients {
            bytes.extend_from_slice(&(ig.len() as u32).to_le_bytes());
            for g in ig {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let mut offset = 1;

        // Input gradient
        let n_grad = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut input_gradient = Vec::with_capacity(n_grad);
        for _ in 0..n_grad {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let g = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
            input_gradient.push(g);
        }

        // Prediction and confidence
        if offset + 8 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 8,
                actual: bytes.len(),
            });
        }
        let prediction = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let confidence = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        // Activations
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_activations = bytes[offset] != 0;
        offset += 1;

        let activations = if has_activations {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_layers = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut layers = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let layer_len = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                offset += 4;

                let mut layer = Vec::with_capacity(layer_len);
                for _ in 0..layer_len {
                    if offset + 4 > bytes.len() {
                        return Err(PathError::InsufficientData {
                            expected: offset + 4,
                            actual: bytes.len(),
                        });
                    }
                    let a = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    offset += 4;
                    layer.push(a);
                }
                layers.push(layer);
            }
            Some(layers)
        } else {
            None
        };

        // Attention weights (similar pattern)
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_attention = bytes[offset] != 0;
        offset += 1;

        let attention_weights = if has_attention {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_layers = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut layers = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let layer_len = u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]) as usize;
                offset += 4;

                let mut layer = Vec::with_capacity(layer_len);
                for _ in 0..layer_len {
                    if offset + 4 > bytes.len() {
                        return Err(PathError::InsufficientData {
                            expected: offset + 4,
                            actual: bytes.len(),
                        });
                    }
                    let a = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    offset += 4;
                    layer.push(a);
                }
                layers.push(layer);
            }
            Some(layers)
        } else {
            None
        };

        // Integrated gradients
        if offset + 1 > bytes.len() {
            return Err(PathError::InsufficientData {
                expected: offset + 1,
                actual: bytes.len(),
            });
        }
        let has_ig = bytes[offset] != 0;
        offset += 1;

        let integrated_gradients = if has_ig {
            if offset + 4 > bytes.len() {
                return Err(PathError::InsufficientData {
                    expected: offset + 4,
                    actual: bytes.len(),
                });
            }
            let n_ig = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            let mut ig = Vec::with_capacity(n_ig);
            for _ in 0..n_ig {
                if offset + 4 > bytes.len() {
                    return Err(PathError::InsufficientData {
                        expected: offset + 4,
                        actual: bytes.len(),
                    });
                }
                let g = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                offset += 4;
                ig.push(g);
            }
            Some(ig)
        } else {
            None
        };

        Ok(Self {
            input_gradient,
            activations,
            attention_weights,
            integrated_gradients,
            prediction,
            confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // LinearPath tests
    // ==========================================================================

    #[test]
    fn test_linear_path_new() {
        let path = LinearPath::new(vec![0.5, -0.3, 0.2], 0.1, 0.5, 0.87);
        assert_eq!(path.contributions.len(), 3);
        assert_eq!(path.intercept, 0.1);
        assert_eq!(path.prediction, 0.87);
        assert!(path.probability.is_none());
    }

    #[test]
    fn test_linear_path_with_probability() {
        let path = LinearPath::new(vec![0.5], 0.0, 0.0, 0.87).with_probability(0.87);
        assert_eq!(path.probability, Some(0.87));
    }

    #[test]
    fn test_linear_path_top_features() {
        let path = LinearPath::new(vec![0.1, -0.5, 0.3, 0.2], 0.0, 0.0, 0.0);
        let top = path.top_features(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // -0.5 has highest absolute value
        assert_eq!(top[1].0, 2); // 0.3 is second
    }

    #[test]
    fn test_linear_path_explain() {
        let path = LinearPath::new(vec![0.42, 0.28, -0.15], 0.32, 0.87, 0.87);
        let explanation = path.explain();
        assert!(explanation.contains("Prediction: 0.87"));
        assert!(explanation.contains("feature[0]: +0.42"));
        assert!(explanation.contains("Intercept: 0.32"));
    }

    #[test]
    fn test_linear_path_serialization_roundtrip() {
        let path =
            LinearPath::new(vec![0.5, -0.3, 0.2, 0.1], 0.1, 0.5, 0.87).with_probability(0.87);

        let bytes = path.to_bytes();
        let restored = LinearPath::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(path.contributions.len(), restored.contributions.len());
        for (a, b) in path.contributions.iter().zip(restored.contributions.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((path.intercept - restored.intercept).abs() < 1e-6);
        assert!((path.logit - restored.logit).abs() < 1e-6);
        assert!((path.prediction - restored.prediction).abs() < 1e-6);
        assert_eq!(path.probability, restored.probability);
    }

    #[test]
    fn test_linear_path_confidence() {
        let path = LinearPath::new(vec![0.5], 0.0, 0.5, 0.5).with_probability(0.9);
        assert!((path.confidence() - 0.9).abs() < 1e-6);

        let path_no_prob = LinearPath::new(vec![0.5], 0.0, 0.5, 0.5);
        assert!(path_no_prob.confidence() > 0.0);
        assert!(path_no_prob.confidence() <= 1.0);
    }

    // ==========================================================================
    // TreePath tests
    // ==========================================================================

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

    // ==========================================================================
    // ForestPath tests
    // ==========================================================================

    #[test]
    fn test_forest_path_new() {
        let tree_paths = vec![
            TreePath::new(
                vec![],
                LeafInfo {
                    prediction: 0.8,
                    n_samples: 50,
                    class_distribution: None,
                },
            ),
            TreePath::new(
                vec![],
                LeafInfo {
                    prediction: 0.9,
                    n_samples: 50,
                    class_distribution: None,
                },
            ),
        ];
        let predictions = vec![0.8, 0.9];

        let path = ForestPath::new(tree_paths, predictions);
        assert_eq!(path.n_trees(), 2);
        assert!((path.ensemble_prediction - 0.85).abs() < 1e-6);
        assert!(path.tree_agreement > 0.0);
    }

    #[test]
    fn test_forest_path_agreement_identical() {
        let path = ForestPath::new(vec![], vec![0.5, 0.5, 0.5]);
        assert!((path.tree_agreement - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_forest_path_agreement_varied() {
        let path = ForestPath::new(vec![], vec![0.0, 1.0]);
        assert!(path.tree_agreement < 1.0);
    }

    #[test]
    fn test_forest_path_serialization_roundtrip() {
        let tree_paths = vec![TreePath::new(
            vec![TreeSplit {
                feature_idx: 0,
                threshold: 0.5,
                went_left: true,
                n_samples: 100,
            }],
            LeafInfo {
                prediction: 0.8,
                n_samples: 50,
                class_distribution: None,
            },
        )];

        let path = ForestPath::new(tree_paths, vec![0.8]).with_feature_importance(vec![0.3, 0.7]);

        let bytes = path.to_bytes();
        let restored = ForestPath::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(path.n_trees(), restored.n_trees());
        assert!((path.ensemble_prediction - restored.ensemble_prediction).abs() < 1e-6);
        assert_eq!(
            path.feature_importance.len(),
            restored.feature_importance.len()
        );
    }

    // ==========================================================================
    // KNNPath tests
    // ==========================================================================

    #[test]
    fn test_knn_path_new() {
        let path = KNNPath::new(vec![0, 5, 10], vec![0.1, 0.2, 0.3], vec![0, 1, 1], 1.0);
        assert_eq!(path.k(), 3);
        assert!(!path.votes.is_empty());
    }

    #[test]
    fn test_knn_path_confidence() {
        let path = KNNPath::new(
            vec![0, 1, 2, 3, 4],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![1, 1, 1, 0, 0],
            1.0,
        );
        // 3 votes for class 1, 2 for class 0
        assert!((path.confidence() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_knn_path_serialization_roundtrip() {
        let path = KNNPath::new(vec![0, 5, 10], vec![0.1, 0.2, 0.3], vec![0, 1, 1], 1.0)
            .with_weighted_votes(vec![0.5, 0.3, 0.2]);

        let bytes = path.to_bytes();
        let restored = KNNPath::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(path.k(), restored.k());
        assert!((path.prediction - restored.prediction).abs() < 1e-6);
        assert!(restored.weighted_votes.is_some());
    }

    // ==========================================================================
    // NeuralPath tests
    // ==========================================================================

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

        let path_with_ig = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.0, 0.0)
            .with_integrated_gradients(vec![0.5, 0.5]);
        assert_eq!(path_with_ig.feature_contributions(), &[0.5, 0.5]);
    }

    // ==========================================================================
    // PathError tests
    // ==========================================================================

    #[test]
    fn test_path_error_display() {
        let err = PathError::InvalidFormat("bad data".to_string());
        assert!(err.to_string().contains("Invalid format"));

        let err = PathError::InsufficientData {
            expected: 100,
            actual: 50,
        };
        assert!(err.to_string().contains("expected 100"));

        let err = PathError::VersionMismatch {
            expected: 1,
            actual: 2,
        };
        assert!(err.to_string().contains("Version mismatch"));
    }

    #[test]
    fn test_linear_path_invalid_version() {
        let mut bytes = vec![2u8]; // Invalid version
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&0.0f32.to_le_bytes());

        let result = LinearPath::from_bytes(&bytes);
        assert!(matches!(result, Err(PathError::VersionMismatch { .. })));
    }

    #[test]
    fn test_linear_path_insufficient_data() {
        let result = LinearPath::from_bytes(&[1u8, 0, 0, 0]);
        assert!(matches!(result, Err(PathError::InsufficientData { .. })));
    }
}

//! Core types for tree-based decision paths

use serde::{Deserialize, Serialize};

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

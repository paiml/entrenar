//! Neural network decision path (gradient-based)

mod path_impl;
mod serialization;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use super::traits::{DecisionPath, PathError};

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

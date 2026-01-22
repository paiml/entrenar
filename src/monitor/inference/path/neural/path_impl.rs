//! NeuralPath implementation methods

use super::NeuralPath;

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

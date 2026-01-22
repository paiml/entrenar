//! KNN Path structure and basic implementation.

use serde::{Deserialize, Serialize};

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

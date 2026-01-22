//! DecisionPath trait implementation for KNNPath.

use super::path::KNNPath;
use crate::monitor::inference::path::traits::{DecisionPath, PathError};

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

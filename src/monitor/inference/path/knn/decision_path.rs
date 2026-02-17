//! DecisionPath trait implementation for KNNPath.

use super::path::KNNPath;
use crate::monitor::inference::path::traits::{DecisionPath, PathError};

/// Stateful byte reader that tracks offset and validates bounds.
struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, PathError> {
        self.ensure_available(1)?;
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    fn read_bool(&mut self) -> Result<bool, PathError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u32(&mut self) -> Result<u32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = u32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    fn read_u32_as_usize(&mut self) -> Result<usize, PathError> {
        Ok(self.read_u32()? as usize)
    }

    fn read_f32(&mut self) -> Result<f32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = f32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    fn read_f32_vec(&mut self) -> Result<Vec<f32>, PathError> {
        let len = self.read_u32()? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f32()?);
        }
        Ok(vec)
    }

    fn read_optional<T>(
        &mut self,
        read_value: impl FnOnce(&mut Self) -> Result<T, PathError>,
    ) -> Result<Option<T>, PathError> {
        let present = self.read_bool()?;
        if present {
            Ok(Some(read_value(self)?))
        } else {
            Ok(None)
        }
    }

    fn ensure_available(&self, needed: usize) -> Result<(), PathError> {
        if self.offset + needed > self.data.len() {
            return Err(PathError::InsufficientData {
                expected: self.offset + needed,
                actual: self.data.len(),
            });
        }
        Ok(())
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

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let k = reader.read_u32_as_usize()?;

        // Neighbor indices
        let mut neighbor_indices = Vec::with_capacity(k);
        for _ in 0..k {
            neighbor_indices.push(reader.read_u32_as_usize()?);
        }

        // Distances
        let mut distances = Vec::with_capacity(k);
        for _ in 0..k {
            distances.push(reader.read_f32()?);
        }

        // Labels
        let mut neighbor_labels = Vec::with_capacity(k);
        for _ in 0..k {
            neighbor_labels.push(reader.read_u32_as_usize()?);
        }

        // Votes
        let n_votes = reader.read_u32_as_usize()?;
        let mut votes = Vec::with_capacity(n_votes);
        for _ in 0..n_votes {
            let class = reader.read_u32_as_usize()?;
            let count = reader.read_u32_as_usize()?;
            votes.push((class, count));
        }

        // Weighted votes
        let weighted_votes = reader.read_optional(ByteReader::read_f32_vec)?;

        // Prediction
        let prediction = reader.read_f32()?;

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

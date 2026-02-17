//! TreePath struct and core implementation

use super::types::{LeafInfo, TreeSplit};
use crate::monitor::inference::path::traits::{DecisionPath, PathError};
use serde::{Deserialize, Serialize};

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
    pub(crate) contributions: Vec<f32>,
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

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        // Splits
        let n_splits = reader.read_u32()? as usize;
        let mut splits = Vec::with_capacity(n_splits);
        for _ in 0..n_splits {
            let feature_idx = reader.read_u32()? as usize;
            let threshold = reader.read_f32()?;
            let went_left = reader.read_bool()?;
            let n_samples = reader.read_u32()? as usize;
            splits.push(TreeSplit {
                feature_idx,
                threshold,
                went_left,
                n_samples,
            });
        }

        // Leaf info
        let prediction = reader.read_f32()?;
        let n_samples = reader.read_u32()? as usize;
        let class_distribution = reader.read_optional(ByteReader::read_f32_vec)?;
        let leaf = LeafInfo {
            prediction,
            n_samples,
            class_distribution,
        };

        let gini_path = reader.read_f32_vec()?;
        let contributions = reader.read_f32_vec()?;

        Ok(Self {
            splits,
            leaf,
            gini_path,
            contributions,
        })
    }
}

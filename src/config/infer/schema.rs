//! Inferred schema representation

use std::collections::HashMap;

use super::stats::ColumnStats;
use super::types::FeatureType;

/// Inferred schema for a dataset
#[derive(Debug, Clone, Default)]
pub struct InferredSchema {
    /// Feature name -> inferred type
    pub features: HashMap<String, FeatureType>,
    /// Column statistics used for inference
    pub stats: HashMap<String, ColumnStats>,
}

impl InferredSchema {
    /// Get features of a specific type
    pub fn features_of_type(&self, feature_type: FeatureType) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| t == feature_type)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get target columns
    pub fn targets(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get input feature columns (non-targets)
    pub fn inputs(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                !matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

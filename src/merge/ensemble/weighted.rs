//! ENT-032: Weighted average merging

use super::{MergeError, Model};
use std::collections::HashMap;

/// Weighted average of multiple models
pub fn weighted_average_merge(models: &[Model], weights: &[f32]) -> Result<Model, MergeError> {
    // Normalize weights or use uniform
    let weights: Vec<f32> = if weights.is_empty() {
        vec![1.0 / models.len() as f32; models.len()]
    } else if weights.len() != models.len() {
        return Err(MergeError::InvalidConfig(format!(
            "Weights length {} doesn't match models length {}",
            weights.len(),
            models.len()
        )));
    } else {
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(MergeError::InvalidConfig(
                "Weights must sum to positive value".to_string(),
            ));
        }
        weights.iter().map(|w| w / sum).collect()
    };

    // Get reference model for parameter names
    let reference = &models[0];
    let mut merged = HashMap::new();

    for name in reference.keys() {
        let param_len = reference[name].len();

        // Weighted sum
        let mut weighted_sum = ndarray::Array1::<f32>::zeros(param_len);
        for (model, weight) in models.iter().zip(weights.iter()) {
            let param = model
                .get(name)
                .ok_or_else(|| MergeError::IncompatibleArchitectures(format!("Missing {name}")))?;
            if param.len() != param_len {
                return Err(MergeError::ShapeMismatch(name.clone()));
            }
            weighted_sum = weighted_sum + param.data() * *weight;
        }

        merged.insert(
            name.clone(),
            crate::autograd::Tensor::new(weighted_sum, false),
        );
    }

    Ok(merged)
}

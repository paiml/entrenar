//! SLERP (Spherical Linear intERPolation) merge algorithm
//!
//! SLERP blends two models using spherical interpolation, which is better
//! than linear interpolation for unit-normalized weight spaces.

use super::{validate_models, Model, MergeError};
use crate::autograd::Tensor;
use ndarray::Array1;
use std::collections::HashMap;

/// Configuration for SLERP merge
#[derive(Clone, Debug)]
pub struct SlerpConfig {
    /// Interpolation parameter: t ∈ [0, 1]
    /// t=0.0 -> 100% model1
    /// t=0.5 -> 50/50 blend
    /// t=1.0 -> 100% model2
    pub t: f32,
}

impl Default for SlerpConfig {
    fn default() -> Self {
        Self { t: 0.5 }
    }
}

impl SlerpConfig {
    pub fn new(t: f32) -> Result<Self, MergeError> {
        if !(0.0..=1.0).contains(&t) {
            return Err(MergeError::InvalidConfig(format!(
                "Interpolation parameter t must be in [0.0, 1.0], got {}",
                t
            )));
        }
        Ok(Self { t })
    }
}

/// SLERP merge: spherical linear interpolation between two models
///
/// # Arguments
/// * `model1` - First model (t=0)
/// * `model2` - Second model (t=1)
/// * `config` - SLERP configuration (interpolation parameter t)
///
/// # Returns
/// Interpolated model at parameter t
///
/// # Algorithm
/// For each parameter w1, w2:
/// 1. Compute angle: θ = arccos(w1·w2 / (|w1||w2|))
/// 2. If θ ≈ 0 (parallel vectors), use linear interpolation
/// 3. Otherwise: w = (sin((1-t)θ)/sinθ)*w1 + (sin(tθ)/sinθ)*w2
///
/// # Notes
/// - SLERP is rotation-invariant and maintains constant angular velocity
/// - Falls back to linear interpolation for nearly parallel weights
/// - Only works for 2 models (unlike TIES/DARE which support N models)
pub fn slerp_merge(
    model1: &Model,
    model2: &Model,
    config: &SlerpConfig,
) -> Result<Model, MergeError> {
    validate_models(&[model1.clone(), model2.clone()])?;

    let mut merged = HashMap::new();

    for (name, tensor1) in model1 {
        let tensor2 = &model2[name];
        let merged_tensor = slerp_tensor(tensor1, tensor2, config.t);
        merged.insert(name.clone(), merged_tensor);
    }

    Ok(merged)
}

/// Spherical linear interpolation between two tensors
fn slerp_tensor(tensor1: &Tensor, tensor2: &Tensor, t: f32) -> Tensor {
    let w1 = tensor1.data();
    let w2 = tensor2.data();

    // Compute dot product and norms
    let dot = w1.iter().zip(w2.iter()).map(|(a, b)| a * b).sum::<f32>();
    let norm1 = w1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2 = w2.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle zero vectors
    if norm1 < 1e-8 || norm2 < 1e-8 {
        return linear_interp_tensor(tensor1, tensor2, t);
    }

    // Compute cosine of angle
    let cos_theta = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);

    // If vectors are nearly parallel (cos_theta ≈ 1), use linear interpolation
    const EPSILON: f32 = 1e-4;
    if (cos_theta - 1.0).abs() < EPSILON {
        return linear_interp_tensor(tensor1, tensor2, t);
    }

    // Compute theta and sin(theta)
    let theta = cos_theta.acos();
    let sin_theta = theta.sin();

    // Spherical interpolation
    // w = (sin((1-t)θ)/sinθ)*w1 + (sin(tθ)/sinθ)*w2
    let coef1 = ((1.0 - t) * theta).sin() / sin_theta;
    let coef2 = (t * theta).sin() / sin_theta;

    let interpolated: Array1<f32> = w1
        .iter()
        .zip(w2.iter())
        .map(|(a, b)| coef1 * a + coef2 * b)
        .collect();

    Tensor::new(interpolated, false)
}

/// Linear interpolation fallback for parallel vectors
fn linear_interp_tensor(tensor1: &Tensor, tensor2: &Tensor, t: f32) -> Tensor {
    let w1 = tensor1.data();
    let w2 = tensor2.data();

    let interpolated: Array1<f32> = w1
        .iter()
        .zip(w2.iter())
        .map(|(a, b)| (1.0 - t) * a + t * b)
        .collect();

    Tensor::new(interpolated, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slerp_at_endpoints() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let t2 = Tensor::from_vec(vec![4.0, 5.0, 6.0], false);

        // t=0.0 should return tensor1
        let result = slerp_tensor(&t1, &t2, 0.0);
        for (a, b) in result.data().iter().zip(t1.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // t=1.0 should return tensor2
        let result = slerp_tensor(&t1, &t2, 1.0);
        for (a, b) in result.data().iter().zip(t2.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_slerp_midpoint() {
        let t1 = Tensor::from_vec(vec![1.0, 0.0], false);
        let t2 = Tensor::from_vec(vec![0.0, 1.0], false);

        // t=0.5 should be close to (1/√2, 1/√2) for perpendicular vectors
        let result = slerp_tensor(&t1, &t2, 0.5);
        let expected_val = 1.0 / 2.0f32.sqrt();

        assert!((result.data()[0] - expected_val).abs() < 1e-5);
        assert!((result.data()[1] - expected_val).abs() < 1e-5);
    }

    #[test]
    fn test_linear_interp_fallback_for_parallel() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let t2 = Tensor::from_vec(vec![2.0, 4.0, 6.0], false); // Parallel (2x t1)

        let result = slerp_tensor(&t1, &t2, 0.5);

        // Should fall back to linear interpolation
        let expected = [1.5, 3.0, 4.5]; // (1+2)/2, (2+4)/2, (3+6)/2
        for (a, e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_slerp_config_validation() {
        assert!(SlerpConfig::new(0.0).is_ok());
        assert!(SlerpConfig::new(0.5).is_ok());
        assert!(SlerpConfig::new(1.0).is_ok());
        assert!(SlerpConfig::new(-0.1).is_err());
        assert!(SlerpConfig::new(1.1).is_err());
    }

    #[test]
    fn test_slerp_merge() {
        let mut model1 = HashMap::new();
        model1.insert("w".to_string(), Tensor::from_vec(vec![1.0, 0.0], false));

        let mut model2 = HashMap::new();
        model2.insert("w".to_string(), Tensor::from_vec(vec![0.0, 1.0], false));

        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&model1, &model2, &config).unwrap();

        // Midpoint of perpendicular unit vectors
        let expected_val = 1.0 / 2.0f32.sqrt();
        assert!((merged["w"].data()[0] - expected_val).abs() < 1e-5);
        assert!((merged["w"].data()[1] - expected_val).abs() < 1e-5);
    }
}

//! Property tests for model merging algorithms

use super::*;
use crate::autograd::Tensor;
use std::collections::HashMap;

/// Helper to create a simple model with one parameter
fn create_model(name: &str, values: Vec<f32>) -> Model {
    let mut model = HashMap::new();
    model.insert(name.to_string(), Tensor::from_vec(values, false));
    model
}

/// Helper to create multiple models
fn create_models(values_per_model: Vec<Vec<f32>>) -> Vec<Model> {
    values_per_model
        .into_iter()
        .map(|values| create_model("w", values))
        .collect()
}

#[cfg(test)]
mod ties_properties {
    use super::*;
    use crate::merge::{ties_merge, TiesConfig};

    #[test]
    fn ties_is_permutation_invariant() {
        // Property: TIES merge result should be independent of model ordering
        let base = create_model("w", vec![0.0, 0.0, 0.0, 0.0]);
        let models = create_models(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, 3.0, 4.0],
            vec![1.0, -2.0, -3.0, 4.0],
        ]);

        let config = TiesConfig::new(0.5).unwrap();

        let result1 = ties_merge(&models, &base, &config).unwrap();

        // Permute models
        let permuted = vec![models[2].clone(), models[0].clone(), models[1].clone()];
        let result2 = ties_merge(&permuted, &base, &config).unwrap();

        // Results should be identical
        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "TIES should be permutation-invariant: {} != {}",
                a,
                b
            );
        }
    }

    #[test]
    fn ties_with_identical_models_has_same_deltas() {
        // Property: Merging identical models should preserve the delta direction
        // Note: Due to trimming, exact values may differ, but non-zero elements should align
        let base = create_model("w", vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let model = create_model("w", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let models = vec![model.clone(), model.clone()];

        // Use high density to preserve most values
        let config = TiesConfig::new(0.8).unwrap();
        let result = ties_merge(&models, &base, &config).unwrap();

        // Result should be close to the model (after trimming)
        // Since both models are identical, all votes agree, so kept values equal original
        let expected = model["w"].data();
        let actual = result["w"].data();

        // Check that non-zero elements have correct sign and magnitude is reasonable
        for (a, e) in actual.iter().zip(expected.iter()) {
            if a.abs() > 1e-6 {
                // Non-zero result should match expected sign
                assert!(a * e > 0.0, "Sign mismatch: {} vs {}", a, e);
            }
        }
    }

    #[test]
    fn ties_preserves_zero_deltas() {
        // Property: If all models equal base, output equals base
        let base = create_model("w", vec![5.0, 10.0]);
        let models = vec![base.clone(), base.clone()];

        let config = TiesConfig::default();
        let result = ties_merge(&models, &base, &config).unwrap();

        let expected = base["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod dare_properties {
    use super::*;
    use crate::merge::{dare_merge, DareConfig};

    #[test]
    fn dare_is_deterministic_with_seed() {
        // Property: Same seed produces same results
        let base = create_model("w", vec![0.0, 0.0]);
        let models = create_models(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let config = DareConfig::new(0.5).unwrap().with_seed(42);

        let result1 = dare_merge(&models, &base, &config).unwrap();
        let result2 = dare_merge(&models, &base, &config).unwrap();

        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn dare_with_zero_drop_is_average() {
        // Property: drop_prob=0 should equal simple average
        let base = create_model("w", vec![0.0, 0.0]);
        let models = create_models(vec![vec![2.0, 4.0], vec![4.0, 6.0]]);

        let config = DareConfig::new(0.0).unwrap();
        let result = dare_merge(&models, &base, &config).unwrap();

        // Expected: (2+4)/2 = 3.0, (4+6)/2 = 5.0
        assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
        assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn dare_preserves_zero_deltas() {
        // Property: If all models equal base, output equals base
        let base = create_model("w", vec![7.0, 14.0]);
        let models = vec![base.clone(), base.clone()];

        let config = DareConfig::default();
        let result = dare_merge(&models, &base, &config).unwrap();

        let expected = base["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod slerp_properties {
    use super::*;
    use crate::merge::{slerp_merge, SlerpConfig};

    #[test]
    fn slerp_at_t0_returns_model1() {
        // Property: t=0 should return first model exactly
        let model1 = create_model("w", vec![1.0, 2.0, 3.0]);
        let model2 = create_model("w", vec![4.0, 5.0, 6.0]);

        let config = SlerpConfig::new(0.0).unwrap();
        let result = slerp_merge(&model1, &model2, &config).unwrap();

        let expected = model1["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn slerp_at_t1_returns_model2() {
        // Property: t=1 should return second model exactly
        let model1 = create_model("w", vec![1.0, 2.0, 3.0]);
        let model2 = create_model("w", vec![4.0, 5.0, 6.0]);

        let config = SlerpConfig::new(1.0).unwrap();
        let result = slerp_merge(&model1, &model2, &config).unwrap();

        let expected = model2["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn slerp_is_continuous() {
        // Property: Small changes in t produce small changes in output
        let model1 = create_model("w", vec![1.0, 0.0]);
        let model2 = create_model("w", vec![0.0, 1.0]);

        let config1 = SlerpConfig::new(0.5).unwrap();
        let config2 = SlerpConfig::new(0.51).unwrap();

        let result1 = slerp_merge(&model1, &model2, &config1).unwrap();
        let result2 = slerp_merge(&model1, &model2, &config2).unwrap();

        // Results should be very close for nearby t values
        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 0.1); // Generous tolerance for continuity
        }
    }

    #[test]
    fn slerp_symmetric_models_at_midpoint() {
        // Property: For symmetric models, t=0.5 should be exactly midway
        let model1 = create_model("w", vec![1.0]);
        let model2 = create_model("w", vec![-1.0]);

        let config = SlerpConfig::new(0.5).unwrap();
        let result = slerp_merge(&model1, &model2, &config).unwrap();

        // For anti-parallel vectors, SLERP at t=0.5 should be near zero
        assert!(result["w"].data()[0].abs() < 0.1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::merge::{dare_merge, slerp_merge, ties_merge, DareConfig, SlerpConfig, TiesConfig};

    #[test]
    fn test_three_way_merge_comparison() {
        // Compare all three methods on same inputs
        let base = create_model("w", vec![0.0, 0.0, 0.0]);
        let models = create_models(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
        ]);

        // TIES
        let ties_config = TiesConfig::default();
        let ties_result = ties_merge(&models, &base, &ties_config).unwrap();

        // DARE with zero drop (equivalent to average)
        let dare_config = DareConfig::new(0.0).unwrap();
        let dare_result = dare_merge(&models, &base, &dare_config).unwrap();

        // SLERP at midpoint
        let slerp_config = SlerpConfig::new(0.5).unwrap();
        let slerp_result = slerp_merge(&models[0], &models[1], &slerp_config).unwrap();

        // All should produce reasonable results (no NaN/Inf)
        for val in ties_result["w"].data().iter() {
            assert!(val.is_finite());
        }
        for val in dare_result["w"].data().iter() {
            assert!(val.is_finite());
        }
        for val in slerp_result["w"].data().iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_incompatible_shapes_rejected() {
        let base = create_model("w", vec![0.0, 0.0]);
        let model1 = create_model("w", vec![1.0, 2.0]);
        let model2 = create_model("w", vec![3.0, 4.0, 5.0]); // Wrong shape!

        let models = vec![model1, model2];

        // TIES should reject
        let ties_result = ties_merge(&models, &base, &TiesConfig::default());
        assert!(ties_result.is_err());

        // DARE should reject
        let dare_result = dare_merge(&models, &base, &DareConfig::default());
        assert!(dare_result.is_err());
    }
}

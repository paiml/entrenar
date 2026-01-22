//! Tests for ensemble merging

use super::*;
use crate::autograd::Tensor;
use proptest::prelude::*;
use std::collections::HashMap;

fn make_model(values: Vec<f32>) -> Model {
    let mut m = HashMap::new();
    m.insert("w".to_string(), Tensor::from_vec(values, false));
    m
}

fn models_approx_equal(m1: &Model, m2: &Model, tol: f32) -> bool {
    for (name, t1) in m1 {
        if let Some(t2) = m2.get(name) {
            for (a, b) in t1.data().iter().zip(t2.data().iter()) {
                if (a - b).abs() > tol {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    true
}

// ============================================================
// Weighted Average Tests
// ============================================================

#[test]
fn test_uniform_average_two_models() {
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    // (2+4)/2 = 3, (4+6)/2 = 5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_uniform_average_three_models() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![2.0, 4.0]);
    let m3 = make_model(vec![3.0, 6.0]);

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2, m3], &config).unwrap();

    // (1+2+3)/3 = 2, (2+4+6)/3 = 4
    assert!((result["w"].data()[0] - 2.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average() {
    let m1 = make_model(vec![0.0]);
    let m2 = make_model(vec![10.0]);

    // 70% m1, 30% m2
    let config = EnsembleConfig::weighted_average(vec![0.7, 0.3]);
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    // 0*0.7 + 10*0.3 = 3.0
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_unnormalized() {
    let m1 = make_model(vec![0.0]);
    let m2 = make_model(vec![10.0]);

    // Unnormalized weights: 7, 3 -> normalized to 0.7, 0.3
    let config = EnsembleConfig::weighted_average(vec![7.0, 3.0]);
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_wrong_length() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig::weighted_average(vec![1.0]); // Wrong length!
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

// ============================================================
// Iterative SLERP Tests
// ============================================================

#[test]
fn test_iterative_slerp_two_models() {
    let m1 = make_model(vec![1.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0]);

    let config = EnsembleConfig::iterative_slerp(0.5);
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    // At t=0.5, perpendicular vectors should blend to (1/sqrt(2), 1/sqrt(2))
    let expected = 1.0 / 2.0f32.sqrt();
    assert!((result["w"].data()[0] - expected).abs() < 1e-4);
    assert!((result["w"].data()[1] - expected).abs() < 1e-4);
}

#[test]
fn test_iterative_slerp_three_models() {
    let m1 = make_model(vec![1.0, 0.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0, 0.0]);
    let m3 = make_model(vec![0.0, 0.0, 1.0]);

    let config = EnsembleConfig::iterative_slerp(0.5);
    let result = ensemble_merge(&[m1, m2, m3], &config).unwrap();

    // Result should be finite and have reasonable values
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_iterative_slerp_t0_returns_first() {
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![4.0, 5.0, 6.0]);
    let m3 = make_model(vec![7.0, 8.0, 9.0]);

    let config = EnsembleConfig::iterative_slerp(0.0);
    let result = ensemble_merge(&[m1.clone(), m2, m3], &config).unwrap();

    assert!(models_approx_equal(&result, &m1, 1e-5));
}

// ============================================================
// Hierarchical Tests
// ============================================================

#[test]
fn test_hierarchical_four_models() {
    let models: Vec<Model> = (0..4)
        .map(|i| make_model(vec![i as f32 * 2.0, i as f32 * 3.0]))
        .collect();

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage {
        weights: vec![0.5, 0.5],
    });
    let result = ensemble_merge(&models, &config).unwrap();

    // ((0+2)/2 + (4+6)/2) / 2 = (1 + 5) / 2 = 3
    // ((0+3)/2 + (6+9)/2) / 2 = (1.5 + 7.5) / 2 = 4.5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 4.5).abs() < 1e-5);
}

#[test]
fn test_hierarchical_with_slerp() {
    let m1 = make_model(vec![1.0, 0.0]);
    let m2 = make_model(vec![0.0, 1.0]);
    let m3 = make_model(vec![-1.0, 0.0]);
    let m4 = make_model(vec![0.0, -1.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::IterativeSlerp { t: 0.5 });
    let result = ensemble_merge(&[m1, m2, m3, m4], &config).unwrap();

    // Result should be finite
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

// ============================================================
// TIES/DARE via Ensemble
// ============================================================

#[test]
fn test_ensemble_ties() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, -3.0, 4.0]);
    let m2 = make_model(vec![1.0, -2.0, 3.0, 4.0]);

    let config = EnsembleConfig::ties(base, 0.5);
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    // Should produce valid output
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_ensemble_dare() {
    let base = make_model(vec![0.0, 0.0]);
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::dare(base, 0.0, Some(42));
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    // With drop_prob=0, should be average: (2+4)/2=3, (4+6)/2=5
    assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
    assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
}

// ============================================================
// Error Cases
// ============================================================

#[test]
fn test_insufficient_models() {
    let m = make_model(vec![1.0]);
    let config = EnsembleConfig::uniform_average();

    let result = ensemble_merge(&[m], &config);
    assert!(matches!(
        result,
        Err(MergeError::InsufficientModels { min: 2, got: 1 })
    ));
}

#[test]
fn test_ties_without_base() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig {
        base: None,
        strategy: EnsembleStrategy::Ties { density: 0.5 },
    };

    let result = ensemble_merge(&[m1, m2], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_dare_without_base() {
    let m1 = make_model(vec![1.0]);
    let m2 = make_model(vec![2.0]);

    let config = EnsembleConfig {
        base: None,
        strategy: EnsembleStrategy::Dare {
            drop_prob: 0.5,
            seed: None,
        },
    };

    let result = ensemble_merge(&[m1, m2], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

// ============================================================
// Property Tests
// ============================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_uniform_average_is_mean(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v3 in proptest::collection::vec(-10.0f32..10.0, 3..6)
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1.clone());
        let m2 = make_model(v2.clone());
        let m3 = make_model(v3.clone());

        let config = EnsembleConfig::uniform_average();
        let result = ensemble_merge(&[m1, m2, m3], &config).unwrap();

        for i in 0..len {
            let expected = (v1[i] + v2[i] + v3[i]) / 3.0;
            prop_assert!(
                (result["w"].data()[i] - expected).abs() < 1e-4,
                "Uniform average mismatch at {}: {} vs {}",
                i, result["w"].data()[i], expected
            );
        }
    }

    #[test]
    fn prop_weighted_average_sums_correctly(
        v1 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        v2 in proptest::collection::vec(-10.0f32..10.0, 3..6),
        w1 in 0.1f32..1.0,
        w2 in 0.1f32..1.0
    ) {
        let len = v1.len().min(v2.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();

        let m1 = make_model(v1.clone());
        let m2 = make_model(v2.clone());

        let config = EnsembleConfig::weighted_average(vec![w1, w2]);
        let result = ensemble_merge(&[m1, m2], &config).unwrap();

        let total = w1 + w2;
        for i in 0..len {
            let expected = (v1[i] * w1 + v2[i] * w2) / total;
            prop_assert!(
                (result["w"].data()[i] - expected).abs() < 1e-4,
                "Weighted average mismatch"
            );
        }
    }

    #[test]
    fn prop_uniform_average_permutation_invariant(
        v1 in proptest::collection::vec(-5.0f32..5.0, 3..5),
        v2 in proptest::collection::vec(-5.0f32..5.0, 3..5),
        v3 in proptest::collection::vec(-5.0f32..5.0, 3..5)
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = EnsembleConfig::uniform_average();

        let r1 = ensemble_merge(&[m1.clone(), m2.clone(), m3.clone()], &config).unwrap();
        let r2 = ensemble_merge(&[m2.clone(), m3.clone(), m1.clone()], &config).unwrap();
        let r3 = ensemble_merge(&[m3, m1, m2], &config).unwrap();

        prop_assert!(models_approx_equal(&r1, &r2, 1e-5));
        prop_assert!(models_approx_equal(&r2, &r3, 1e-5));
    }

    #[test]
    fn prop_iterative_slerp_produces_finite_output(
        v1 in proptest::collection::vec(0.1f32..10.0, 3..6),
        v2 in proptest::collection::vec(0.1f32..10.0, 3..6),
        v3 in proptest::collection::vec(0.1f32..10.0, 3..6),
        t in 0.1f32..0.9
    ) {
        let len = v1.len().min(v2.len()).min(v3.len());
        let v1: Vec<f32> = v1.into_iter().take(len).collect();
        let v2: Vec<f32> = v2.into_iter().take(len).collect();
        let v3: Vec<f32> = v3.into_iter().take(len).collect();

        let m1 = make_model(v1);
        let m2 = make_model(v2);
        let m3 = make_model(v3);

        let config = EnsembleConfig::iterative_slerp(t);
        let result = ensemble_merge(&[m1, m2, m3], &config).unwrap();

        for val in result["w"].data() {
            prop_assert!(val.is_finite(), "SLERP produced non-finite value");
        }
    }

    #[test]
    fn prop_hierarchical_balanced_four_models(
        values in proptest::collection::vec(
            proptest::collection::vec(-5.0f32..5.0, 3..5),
            4..=4
        )
    ) {
        let len = values.iter().map(std::vec::Vec::len).min().unwrap_or(3);
        let models: Vec<Model> = values
            .into_iter()
            .map(|v| make_model(v.into_iter().take(len).collect()))
            .collect();

        let config = EnsembleConfig::hierarchical(
            EnsembleStrategy::WeightedAverage { weights: vec![0.5, 0.5] }
        );
        let result = ensemble_merge(&models, &config).unwrap();

        for val in result["w"].data() {
            prop_assert!(val.is_finite());
        }
    }

    #[test]
    fn prop_identity_single_weight(
        values in proptest::collection::vec(-10.0f32..10.0, 3..6)
    ) {
        let m1 = make_model(values.clone());
        let m2 = make_model(vec![0.0; values.len()]);

        // Weight 1.0 for m1, 0.0 for m2 should return m1
        let config = EnsembleConfig::weighted_average(vec![1.0, 0.0]);
        let result = ensemble_merge(&[m1.clone(), m2], &config).unwrap();

        prop_assert!(models_approx_equal(&result, &m1, 1e-5));
    }
}

// ============================================================
// Additional Coverage Tests
// ============================================================

#[test]
fn test_weighted_average_zero_sum() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);

    let config = EnsembleConfig::weighted_average(vec![0.0, 0.0]);
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_weighted_average_negative_sum() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);

    let config = EnsembleConfig::weighted_average(vec![-1.0, 0.5]);
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_weighted_average_missing_param() {
    // Model 1 has param "w", model 2 has param "x"
    let mut m1 = HashMap::new();
    m1.insert("w".to_string(), Tensor::from_vec(vec![1.0], false));

    let mut m2 = HashMap::new();
    m2.insert("x".to_string(), Tensor::from_vec(vec![2.0], false));

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(
        result,
        Err(MergeError::IncompatibleArchitectures(_))
    ));
}

#[test]
fn test_weighted_average_shape_mismatch() {
    let m1 = make_model(vec![1.0, 2.0, 3.0]);
    let m2 = make_model(vec![4.0, 5.0]); // Different length

    let config = EnsembleConfig::uniform_average();
    let result = ensemble_merge(&[m1, m2], &config);

    assert!(matches!(result, Err(MergeError::ShapeMismatch(_))));
}

#[test]
fn test_ensemble_config_with_base() {
    let base = make_model(vec![0.0, 0.0]);
    let config = EnsembleConfig::uniform_average().with_base(base);

    assert!(config.base.is_some());
}

#[test]
fn test_ensemble_strategy_default() {
    let strategy = EnsembleStrategy::default();
    matches!(strategy, EnsembleStrategy::WeightedAverage { weights } if weights.is_empty());
}

#[test]
fn test_ensemble_config_default() {
    let config = EnsembleConfig::default();
    assert!(config.base.is_none());
    matches!(config.strategy, EnsembleStrategy::WeightedAverage { .. });
}

#[test]
fn test_hierarchical_with_ties() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = make_model(vec![5.0, 6.0, 7.0, 8.0]);
    let m3 = make_model(vec![9.0, 10.0, 11.0, 12.0]);
    let m4 = make_model(vec![13.0, 14.0, 15.0, 16.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::Ties { density: 0.5 }).with_base(base);

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).unwrap();
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_with_dare() {
    let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
    let m1 = make_model(vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = make_model(vec![5.0, 6.0, 7.0, 8.0]);
    let m3 = make_model(vec![9.0, 10.0, 11.0, 12.0]);
    let m4 = make_model(vec![13.0, 14.0, 15.0, 16.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Dare {
        drop_prob: 0.3,
        seed: Some(42),
    })
    .with_base(base);

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).unwrap();
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_three_models() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);

    let config =
        EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage { weights: vec![] });

    let result = ensemble_merge(&[m1, m2, m3], &config).unwrap();
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_nested_hierarchical_fallback() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    // Nested hierarchical - the inner hierarchical falls back to weighted average
    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Hierarchical {
        leaf_strategy: Box::new(EnsembleStrategy::WeightedAverage { weights: vec![] }),
    });

    let result = ensemble_merge(&[m1, m2, m3, m4], &config).unwrap();
    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_hierarchical_ties_without_base_error() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Ties { density: 0.5 });
    // No base provided

    let result = ensemble_merge(&[m1, m2, m3, m4], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_hierarchical_dare_without_base_error() {
    let m1 = make_model(vec![1.0, 2.0]);
    let m2 = make_model(vec![3.0, 4.0]);
    let m3 = make_model(vec![5.0, 6.0]);
    let m4 = make_model(vec![7.0, 8.0]);

    let config = EnsembleConfig::hierarchical(EnsembleStrategy::Dare {
        drop_prob: 0.3,
        seed: None,
    });
    // No base provided

    let result = ensemble_merge(&[m1, m2, m3, m4], &config);
    assert!(matches!(result, Err(MergeError::InvalidConfig(_))));
}

#[test]
fn test_merge_pair_with_weighted_avg_specific_weights() {
    let m1 = make_model(vec![0.0, 0.0]);
    let m2 = make_model(vec![10.0, 10.0]);

    // Using hierarchical will call merge_pair with specific weights
    let config = EnsembleConfig::hierarchical(EnsembleStrategy::WeightedAverage {
        weights: vec![0.3, 0.7],
    });

    let result = ensemble_merge(&[m1, m2], &config).unwrap();
    // 0*0.3 + 10*0.7 = 7
    assert!((result["w"].data()[0] - 7.0).abs() < 1e-5);
}

#[test]
fn test_ensemble_dare_without_seed() {
    let base = make_model(vec![0.0, 0.0]);
    let m1 = make_model(vec![2.0, 4.0]);
    let m2 = make_model(vec![4.0, 6.0]);

    let config = EnsembleConfig::dare(base, 0.3, None);
    let result = ensemble_merge(&[m1, m2], &config).unwrap();

    for val in result["w"].data() {
        assert!(val.is_finite());
    }
}

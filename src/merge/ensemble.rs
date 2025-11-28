//! ENT-032: Multi-model ensemble merging (>2 models)
//!
//! Provides unified interface for merging multiple models with various strategies:
//! - Weighted averaging
//! - Iterative SLERP (pairwise application)
//! - Hierarchical merging (tree-based)
//! - Layer-wise strategy selection

use super::{
    dare_merge, slerp_merge, ties_merge, DareConfig, MergeError, Model, SlerpConfig, TiesConfig,
};
use std::collections::HashMap;

/// Strategy for ensemble merging
#[derive(Clone, Debug)]
pub enum EnsembleStrategy {
    /// Simple weighted average (equivalent to DARE with drop_prob=0)
    WeightedAverage { weights: Vec<f32> },

    /// TIES merge with configurable density
    Ties { density: f32 },

    /// DARE merge with dropout
    Dare { drop_prob: f32, seed: Option<u64> },

    /// Iterative SLERP: merge models pairwise until one remains
    IterativeSlerp { t: f32 },

    /// Hierarchical: merge in tree structure for balanced combination
    Hierarchical {
        leaf_strategy: Box<EnsembleStrategy>,
    },
}

impl Default for EnsembleStrategy {
    fn default() -> Self {
        Self::WeightedAverage {
            weights: Vec::new(), // Will use uniform weights
        }
    }
}

/// Configuration for ensemble merging
#[derive(Clone, Debug, Default)]
pub struct EnsembleConfig {
    /// Base model for delta-based methods (TIES, DARE)
    /// If None, uses first model as base for delta methods
    pub base: Option<Model>,

    /// Merging strategy
    pub strategy: EnsembleStrategy,
}

impl EnsembleConfig {
    /// Create config for weighted averaging
    pub fn weighted_average(weights: Vec<f32>) -> Self {
        Self {
            base: None,
            strategy: EnsembleStrategy::WeightedAverage { weights },
        }
    }

    /// Create config for uniform averaging
    pub fn uniform_average() -> Self {
        Self {
            base: None,
            strategy: EnsembleStrategy::WeightedAverage {
                weights: Vec::new(),
            },
        }
    }

    /// Create config for TIES merging
    pub fn ties(base: Model, density: f32) -> Self {
        Self {
            base: Some(base),
            strategy: EnsembleStrategy::Ties { density },
        }
    }

    /// Create config for DARE merging
    pub fn dare(base: Model, drop_prob: f32, seed: Option<u64>) -> Self {
        Self {
            base: Some(base),
            strategy: EnsembleStrategy::Dare { drop_prob, seed },
        }
    }

    /// Create config for iterative SLERP
    pub fn iterative_slerp(t: f32) -> Self {
        Self {
            base: None,
            strategy: EnsembleStrategy::IterativeSlerp { t },
        }
    }

    /// Create config for hierarchical merging
    pub fn hierarchical(leaf_strategy: EnsembleStrategy) -> Self {
        Self {
            base: None,
            strategy: EnsembleStrategy::Hierarchical {
                leaf_strategy: Box::new(leaf_strategy),
            },
        }
    }

    /// Set base model for delta-based methods
    pub fn with_base(mut self, base: Model) -> Self {
        self.base = Some(base);
        self
    }
}

/// Merge multiple models using the specified strategy
///
/// # Arguments
/// * `models` - Models to merge (must have at least 2)
/// * `config` - Ensemble configuration
///
/// # Returns
/// Merged model combining all inputs
pub fn ensemble_merge(models: &[Model], config: &EnsembleConfig) -> Result<Model, MergeError> {
    if models.len() < 2 {
        return Err(MergeError::InsufficientModels {
            min: 2,
            got: models.len(),
        });
    }

    match &config.strategy {
        EnsembleStrategy::WeightedAverage { weights } => weighted_average_merge(models, weights),
        EnsembleStrategy::Ties { density } => {
            let base = config
                .base
                .as_ref()
                .ok_or_else(|| MergeError::InvalidConfig("TIES requires base model".to_string()))?;
            let ties_config = TiesConfig::new(*density)?;
            ties_merge(models, base, &ties_config)
        }
        EnsembleStrategy::Dare { drop_prob, seed } => {
            let base = config
                .base
                .as_ref()
                .ok_or_else(|| MergeError::InvalidConfig("DARE requires base model".to_string()))?;
            let mut dare_config = DareConfig::new(*drop_prob)?;
            if let Some(s) = seed {
                dare_config = dare_config.with_seed(*s);
            }
            dare_merge(models, base, &dare_config)
        }
        EnsembleStrategy::IterativeSlerp { t } => iterative_slerp_merge(models, *t),
        EnsembleStrategy::Hierarchical { leaf_strategy } => {
            hierarchical_merge(models, leaf_strategy, config.base.as_ref())
        }
    }
}

/// Weighted average of multiple models
fn weighted_average_merge(models: &[Model], weights: &[f32]) -> Result<Model, MergeError> {
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
            let param = model.get(name).ok_or_else(|| {
                MergeError::IncompatibleArchitectures(format!("Missing {}", name))
            })?;
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

/// Iterative SLERP: merge models pairwise until one remains
///
/// For N models: ((m1 ⊕ m2) ⊕ m3) ⊕ ... ⊕ mN
/// where ⊕ is SLERP at parameter t
fn iterative_slerp_merge(models: &[Model], t: f32) -> Result<Model, MergeError> {
    let config = SlerpConfig::new(t)?;

    let mut current = models[0].clone();
    for model in models.iter().skip(1) {
        current = slerp_merge(&current, model, &config)?;
    }

    Ok(current)
}

/// Hierarchical merge: tree-based for balanced combination
///
/// For 4 models: (m1 ⊕ m2) ⊕ (m3 ⊕ m4)
/// More balanced than iterative for large N
fn hierarchical_merge(
    models: &[Model],
    leaf_strategy: &EnsembleStrategy,
    base: Option<&Model>,
) -> Result<Model, MergeError> {
    if models.len() == 1 {
        return Ok(models[0].clone());
    }

    if models.len() == 2 {
        return merge_pair(&models[0], &models[1], leaf_strategy, base);
    }

    // Split and recurse
    let mid = models.len() / 2;
    let left = hierarchical_merge(&models[..mid], leaf_strategy, base)?;
    let right = hierarchical_merge(&models[mid..], leaf_strategy, base)?;

    merge_pair(&left, &right, leaf_strategy, base)
}

/// Merge two models using specified strategy
fn merge_pair(
    m1: &Model,
    m2: &Model,
    strategy: &EnsembleStrategy,
    base: Option<&Model>,
) -> Result<Model, MergeError> {
    match strategy {
        EnsembleStrategy::WeightedAverage { weights } => {
            let w = if weights.len() == 2 {
                weights.clone()
            } else {
                vec![0.5, 0.5]
            };
            weighted_average_merge(&[m1.clone(), m2.clone()], &w)
        }
        EnsembleStrategy::IterativeSlerp { t } => {
            let config = SlerpConfig::new(*t)?;
            slerp_merge(m1, m2, &config)
        }
        EnsembleStrategy::Ties { density } => {
            let base =
                base.ok_or_else(|| MergeError::InvalidConfig("TIES requires base".to_string()))?;
            let config = TiesConfig::new(*density)?;
            ties_merge(&[m1.clone(), m2.clone()], base, &config)
        }
        EnsembleStrategy::Dare { drop_prob, seed } => {
            let base =
                base.ok_or_else(|| MergeError::InvalidConfig("DARE requires base".to_string()))?;
            let mut config = DareConfig::new(*drop_prob)?;
            if let Some(s) = seed {
                config = config.with_seed(*s);
            }
            dare_merge(&[m1.clone(), m2.clone()], base, &config)
        }
        EnsembleStrategy::Hierarchical { .. } => {
            // For hierarchical, default to weighted average at leaves
            weighted_average_merge(&[m1.clone(), m2.clone()], &[0.5, 0.5])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tensor;
    use proptest::prelude::*;

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

        // At t=0.5, perpendicular vectors should blend to (1/√2, 1/√2)
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
        for val in result["w"].data().iter() {
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
        for val in result["w"].data().iter() {
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
        for val in result["w"].data().iter() {
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

            for val in result["w"].data().iter() {
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
            let len = values.iter().map(|v| v.len()).min().unwrap_or(3);
            let models: Vec<Model> = values
                .into_iter()
                .map(|v| make_model(v.into_iter().take(len).collect()))
                .collect();

            let config = EnsembleConfig::hierarchical(
                EnsembleStrategy::WeightedAverage { weights: vec![0.5, 0.5] }
            );
            let result = ensemble_merge(&models, &config).unwrap();

            for val in result["w"].data().iter() {
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
}

//! ENT-031: Merge commutativity property tests
//!
//! Tests mathematical properties of merge algorithms:
//! - Commutativity: order-independence
//! - Permutation invariance: reordering models
//! - Identity: merging with self
//! - Boundary conditions: endpoint behavior

#[cfg(test)]
mod tests {
    use crate::autograd::Tensor;
    use crate::merge::{
        dare_merge, slerp_merge, ties_merge, DareConfig, Model, SlerpConfig, TiesConfig,
    };
    use proptest::prelude::*;
    use std::collections::HashMap;

    /// Create a model with single parameter
    fn make_model(values: Vec<f32>) -> Model {
        let mut m = HashMap::new();
        m.insert("w".to_string(), Tensor::from_vec(values, false));
        m
    }

    /// Create a model with multiple parameters
    fn make_multi_param_model(params: Vec<(&str, Vec<f32>)>) -> Model {
        let mut m = HashMap::new();
        for (name, values) in params {
            m.insert(name.to_string(), Tensor::from_vec(values, false));
        }
        m
    }

    /// Compare two models for approximate equality
    fn models_approx_equal(m1: &Model, m2: &Model, tolerance: f32) -> bool {
        if m1.len() != m2.len() {
            return false;
        }
        for (name, t1) in m1 {
            if let Some(t2) = m2.get(name) {
                let d1 = t1.data();
                let d2 = t2.data();
                if d1.len() != d2.len() {
                    return false;
                }
                for (a, b) in d1.iter().zip(d2.iter()) {
                    if (a - b).abs() > tolerance {
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
    // SLERP Commutativity Tests
    // ============================================================

    #[test]
    fn slerp_commutativity_basic() {
        // slerp(A, B, t) = slerp(B, A, 1-t)
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_model(vec![4.0, 5.0, 6.0]);

        let t = 0.3;
        let c1 = SlerpConfig::new(t).unwrap();
        let c2 = SlerpConfig::new(1.0 - t).unwrap();

        let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
        let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-4),
            "SLERP should be commutative: slerp(A,B,t) = slerp(B,A,1-t)"
        );
    }

    #[test]
    fn slerp_self_merge_identity() {
        // slerp(A, A, t) = A for any t
        let m = make_model(vec![1.0, 2.0, 3.0, 4.0]);

        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let config = SlerpConfig::new(t).unwrap();
            let result = slerp_merge(&m, &m, &config).unwrap();
            assert!(
                models_approx_equal(&result, &m, 1e-5),
                "slerp(A, A, {}) should equal A",
                t
            );
        }
    }

    #[test]
    fn slerp_midpoint_symmetry() {
        // slerp(A, B, 0.5) should equal slerp(B, A, 0.5)
        let m1 = make_model(vec![1.0, 0.0, 0.0]);
        let m2 = make_model(vec![0.0, 1.0, 0.0]);

        let config = SlerpConfig::new(0.5).unwrap();

        let r1 = slerp_merge(&m1, &m2, &config).unwrap();
        let r2 = slerp_merge(&m2, &m1, &config).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-5),
            "SLERP at t=0.5 should be symmetric"
        );
    }

    // ============================================================
    // DARE Commutativity Tests
    // ============================================================

    #[test]
    fn dare_zero_drop_is_commutative() {
        // With drop_prob=0, DARE becomes simple averaging which is commutative
        let base = make_model(vec![0.0, 0.0]);
        let m1 = make_model(vec![2.0, 4.0]);
        let m2 = make_model(vec![4.0, 6.0]);

        let config = DareConfig::new(0.0).unwrap();

        let r1 = dare_merge(&[m1.clone(), m2.clone()], &base, &config).unwrap();
        let r2 = dare_merge(&[m2, m1], &base, &config).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-5),
            "DARE with drop_prob=0 should be commutative"
        );
    }

    #[test]
    fn dare_self_merge_identity() {
        // Merging identical models with drop_prob=0 should preserve values
        let base = make_model(vec![0.0, 0.0, 0.0]);
        let m = make_model(vec![1.0, 2.0, 3.0]);

        let config = DareConfig::new(0.0).unwrap();
        let result = dare_merge(&[m.clone(), m.clone()], &base, &config).unwrap();

        assert!(
            models_approx_equal(&result, &m, 1e-5),
            "DARE of identical models should preserve values"
        );
    }

    #[test]
    fn dare_permutation_invariance_with_zero_drop() {
        // Order of models shouldn't matter for averaging
        let base = make_model(vec![0.0, 0.0, 0.0]);
        let models = vec![
            make_model(vec![1.0, 2.0, 3.0]),
            make_model(vec![4.0, 5.0, 6.0]),
            make_model(vec![7.0, 8.0, 9.0]),
        ];

        let config = DareConfig::new(0.0).unwrap();

        let r1 = dare_merge(&models, &base, &config).unwrap();

        // Permute: [2, 0, 1]
        let permuted = vec![models[2].clone(), models[0].clone(), models[1].clone()];
        let r2 = dare_merge(&permuted, &base, &config).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-5),
            "DARE with drop_prob=0 should be permutation-invariant"
        );
    }

    // ============================================================
    // TIES Commutativity Tests
    // ============================================================

    #[test]
    fn ties_permutation_invariance() {
        // TIES result should be independent of model ordering
        let base = make_model(vec![0.0, 0.0, 0.0, 0.0]);
        let models = vec![
            make_model(vec![1.0, 2.0, -3.0, 4.0]),
            make_model(vec![1.0, -2.0, 3.0, 4.0]),
            make_model(vec![-1.0, 2.0, 3.0, 4.0]),
        ];

        let config = TiesConfig::new(0.5).unwrap();

        let r1 = ties_merge(&models, &base, &config).unwrap();

        // All permutations should yield same result
        let perms = [
            vec![0, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ];

        for perm in perms {
            let permuted: Vec<Model> = perm.iter().map(|&i| models[i].clone()).collect();
            let r = ties_merge(&permuted, &base, &config).unwrap();
            assert!(
                models_approx_equal(&r1, &r, 1e-5),
                "TIES should be permutation-invariant"
            );
        }
    }

    #[test]
    fn ties_self_merge_preserves_direction() {
        // Merging identical models should preserve sign and direction
        let base = make_model(vec![0.0, 0.0, 0.0]);
        let m = make_model(vec![1.0, -2.0, 3.0]);

        let config = TiesConfig::new(0.8).unwrap(); // High density to keep most values

        let result = ties_merge(&[m.clone(), m.clone()], &base, &config).unwrap();

        // Signs should match original
        let orig = m["w"].data();
        let merged = result["w"].data();
        for (o, r) in orig.iter().zip(merged.iter()) {
            if r.abs() > 1e-6 {
                assert!(
                    o.signum() == r.signum(),
                    "TIES of identical models should preserve sign: {} vs {}",
                    o,
                    r
                );
            }
        }
    }

    // ============================================================
    // Property Tests with Proptest
    // ============================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        // SLERP Properties

        #[test]
        fn prop_slerp_commutativity(
            v1 in proptest::collection::vec(1.0f32..10.0, 3..8),
            v2 in proptest::collection::vec(1.0f32..10.0, 3..8),
            t in 0.01f32..0.99
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let m1 = make_model(v1);
            let m2 = make_model(v2);

            let c1 = SlerpConfig::new(t).unwrap();
            let c2 = SlerpConfig::new(1.0 - t).unwrap();

            let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
            let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-3),
                "slerp(A,B,t) should equal slerp(B,A,1-t)"
            );
        }

        #[test]
        fn prop_slerp_identity(
            values in proptest::collection::vec(-10.0f32..10.0, 3..8),
            t in 0.0f32..=1.0
        ) {
            let m = make_model(values);
            let config = SlerpConfig::new(t).unwrap();

            let result = slerp_merge(&m, &m, &config).unwrap();

            prop_assert!(
                models_approx_equal(&result, &m, 1e-4),
                "slerp(A, A, t) should equal A"
            );
        }

        #[test]
        fn prop_slerp_boundary_t0(
            v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
            v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let m1 = make_model(v1);
            let m2 = make_model(v2);
            let config = SlerpConfig::new(0.0).unwrap();

            let result = slerp_merge(&m1, &m2, &config).unwrap();

            prop_assert!(
                models_approx_equal(&result, &m1, 1e-5),
                "slerp(A, B, 0) should equal A"
            );
        }

        #[test]
        fn prop_slerp_boundary_t1(
            v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
            v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let m1 = make_model(v1);
            let m2 = make_model(v2);
            let config = SlerpConfig::new(1.0).unwrap();

            let result = slerp_merge(&m1, &m2, &config).unwrap();

            prop_assert!(
                models_approx_equal(&result, &m2, 1e-5),
                "slerp(A, B, 1) should equal B"
            );
        }

        #[test]
        fn prop_slerp_midpoint_symmetric(
            v1 in proptest::collection::vec(1.0f32..10.0, 3..6),
            v2 in proptest::collection::vec(1.0f32..10.0, 3..6)
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let m1 = make_model(v1);
            let m2 = make_model(v2);
            let config = SlerpConfig::new(0.5).unwrap();

            let r1 = slerp_merge(&m1, &m2, &config).unwrap();
            let r2 = slerp_merge(&m2, &m1, &config).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-4),
                "slerp at t=0.5 should be symmetric"
            );
        }

        // DARE Properties

        #[test]
        fn prop_dare_zero_drop_commutative(
            v1 in proptest::collection::vec(-10.0f32..10.0, 3..8),
            v2 in proptest::collection::vec(-10.0f32..10.0, 3..8)
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let base = make_model(vec![0.0; len]);
            let m1 = make_model(v1);
            let m2 = make_model(v2);

            let config = DareConfig::new(0.0).unwrap();

            let r1 = dare_merge(&[m1.clone(), m2.clone()], &base, &config).unwrap();
            let r2 = dare_merge(&[m2, m1], &base, &config).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-5),
                "DARE(drop=0) should be commutative"
            );
        }

        #[test]
        fn prop_dare_zero_drop_permutation_invariant(
            v1 in proptest::collection::vec(-5.0f32..5.0, 3..6),
            v2 in proptest::collection::vec(-5.0f32..5.0, 3..6),
            v3 in proptest::collection::vec(-5.0f32..5.0, 3..6)
        ) {
            let len = v1.len().min(v2.len()).min(v3.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();
            let v3: Vec<f32> = v3.into_iter().take(len).collect();

            let base = make_model(vec![0.0; len]);
            let m1 = make_model(v1);
            let m2 = make_model(v2);
            let m3 = make_model(v3);

            let config = DareConfig::new(0.0).unwrap();

            let r1 = dare_merge(&[m1.clone(), m2.clone(), m3.clone()], &base, &config).unwrap();
            let r2 = dare_merge(&[m3.clone(), m1.clone(), m2.clone()], &base, &config).unwrap();
            let r3 = dare_merge(&[m2, m3, m1], &base, &config).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-5) && models_approx_equal(&r2, &r3, 1e-5),
                "DARE(drop=0) should be permutation-invariant"
            );
        }

        #[test]
        fn prop_dare_identity_merge(
            values in proptest::collection::vec(-10.0f32..10.0, 3..8)
        ) {
            let base = make_model(vec![0.0; values.len()]);
            let m = make_model(values);

            let config = DareConfig::new(0.0).unwrap();
            let result = dare_merge(&[m.clone(), m.clone()], &base, &config).unwrap();

            prop_assert!(
                models_approx_equal(&result, &m, 1e-5),
                "DARE of identical models should preserve values"
            );
        }

        // TIES Properties

        #[test]
        fn prop_ties_permutation_invariant_2_models(
            v1 in proptest::collection::vec(-10.0f32..10.0, 4..8),
            v2 in proptest::collection::vec(-10.0f32..10.0, 4..8),
            density in 0.3f32..0.8
        ) {
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let base = make_model(vec![0.0; len]);
            let m1 = make_model(v1);
            let m2 = make_model(v2);

            let config = TiesConfig::new(density).unwrap();

            let r1 = ties_merge(&[m1.clone(), m2.clone()], &base, &config).unwrap();
            let r2 = ties_merge(&[m2, m1], &base, &config).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-5),
                "TIES should be permutation-invariant for 2 models"
            );
        }

        #[test]
        fn prop_ties_permutation_invariant_3_models(
            v1 in proptest::collection::vec(-5.0f32..5.0, 4..6),
            v2 in proptest::collection::vec(-5.0f32..5.0, 4..6),
            v3 in proptest::collection::vec(-5.0f32..5.0, 4..6),
            density in 0.4f32..0.7
        ) {
            let len = v1.len().min(v2.len()).min(v3.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();
            let v3: Vec<f32> = v3.into_iter().take(len).collect();

            let base = make_model(vec![0.0; len]);
            let m1 = make_model(v1);
            let m2 = make_model(v2);
            let m3 = make_model(v3);

            let config = TiesConfig::new(density).unwrap();

            let r1 = ties_merge(&[m1.clone(), m2.clone(), m3.clone()], &base, &config).unwrap();
            let r2 = ties_merge(&[m2.clone(), m3.clone(), m1.clone()], &base, &config).unwrap();
            let r3 = ties_merge(&[m3, m1, m2], &base, &config).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-5) && models_approx_equal(&r2, &r3, 1e-5),
                "TIES should be permutation-invariant for 3 models"
            );
        }

        #[test]
        fn prop_ties_identity_preserves_sign(
            values in proptest::collection::vec(-10.0f32..10.0, 4..8)
                .prop_filter("non-zero values", |v| v.iter().all(|x| x.abs() > 0.1)),
            density in 0.7f32..0.95
        ) {
            let base = make_model(vec![0.0; values.len()]);
            let m = make_model(values.clone());

            let config = TiesConfig::new(density).unwrap();
            let result = ties_merge(&[m.clone(), m.clone()], &base, &config).unwrap();

            // Signs of non-zero results should match original
            let merged = result["w"].data();
            for (i, (orig, res)) in values.iter().zip(merged.iter()).enumerate() {
                if res.abs() > 1e-6 {
                    prop_assert!(
                        orig.signum() == res.signum(),
                        "TIES identity should preserve sign at index {}: {} vs {}",
                        i, orig, res
                    );
                }
            }
        }

        // Multi-parameter model tests

        #[test]
        fn prop_slerp_multi_param_commutativity(
            v1a in proptest::collection::vec(1.0f32..5.0, 3..5),
            v1b in proptest::collection::vec(1.0f32..5.0, 3..5),
            v2a in proptest::collection::vec(1.0f32..5.0, 3..5),
            v2b in proptest::collection::vec(1.0f32..5.0, 3..5),
            t in 0.1f32..0.9
        ) {
            let len_a = v1a.len().min(v2a.len());
            let len_b = v1b.len().min(v2b.len());

            let v1a: Vec<f32> = v1a.into_iter().take(len_a).collect();
            let v2a: Vec<f32> = v2a.into_iter().take(len_a).collect();
            let v1b: Vec<f32> = v1b.into_iter().take(len_b).collect();
            let v2b: Vec<f32> = v2b.into_iter().take(len_b).collect();

            let m1 = make_multi_param_model(vec![("a", v1a), ("b", v1b)]);
            let m2 = make_multi_param_model(vec![("a", v2a), ("b", v2b)]);

            let c1 = SlerpConfig::new(t).unwrap();
            let c2 = SlerpConfig::new(1.0 - t).unwrap();

            let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
            let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

            prop_assert!(
                models_approx_equal(&r1, &r2, 1e-3),
                "Multi-param SLERP should be commutative"
            );
        }
    }

    // ============================================================
    // Edge Case Tests
    // ============================================================

    #[test]
    fn slerp_parallel_vectors_commutativity() {
        // Parallel vectors should also be commutative (linear interp fallback)
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_model(vec![2.0, 4.0, 6.0]); // Parallel to m1

        let t = 0.3;
        let c1 = SlerpConfig::new(t).unwrap();
        let c2 = SlerpConfig::new(1.0 - t).unwrap();

        let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
        let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-4),
            "SLERP with parallel vectors should be commutative"
        );
    }

    #[test]
    fn slerp_antiparallel_vectors_commutativity() {
        // Anti-parallel vectors
        let m1 = make_model(vec![1.0, 2.0, 3.0]);
        let m2 = make_model(vec![-1.0, -2.0, -3.0]);

        let t = 0.4;
        let c1 = SlerpConfig::new(t).unwrap();
        let c2 = SlerpConfig::new(1.0 - t).unwrap();

        let r1 = slerp_merge(&m1, &m2, &c1).unwrap();
        let r2 = slerp_merge(&m2, &m1, &c2).unwrap();

        assert!(
            models_approx_equal(&r1, &r2, 1e-4),
            "SLERP with anti-parallel vectors should be commutative"
        );
    }

    #[test]
    fn dare_with_base_equals_models() {
        // When base equals models, result should equal base
        let base = make_model(vec![5.0, 10.0, 15.0]);
        let config = DareConfig::new(0.5).unwrap().with_seed(42);

        let result = dare_merge(&[base.clone(), base.clone()], &base, &config).unwrap();

        assert!(
            models_approx_equal(&result, &base, 1e-5),
            "DARE when models equal base should return base"
        );
    }

    #[test]
    fn ties_with_base_equals_models() {
        // When base equals models, result should equal base
        let base = make_model(vec![5.0, 10.0, 15.0]);
        let config = TiesConfig::new(0.5).unwrap();

        let result = ties_merge(&[base.clone(), base.clone()], &base, &config).unwrap();

        assert!(
            models_approx_equal(&result, &base, 1e-5),
            "TIES when models equal base should return base"
        );
    }

    #[test]
    fn all_methods_handle_single_element() {
        // Single element vectors
        let base = make_model(vec![0.0]);
        let m1 = make_model(vec![5.0]);
        let m2 = make_model(vec![10.0]);

        // SLERP
        let slerp_r = slerp_merge(&m1, &m2, &SlerpConfig::new(0.5).unwrap()).unwrap();
        assert!(slerp_r["w"].data()[0].is_finite());

        // DARE
        let dare_r = dare_merge(
            &[m1.clone(), m2.clone()],
            &base,
            &DareConfig::new(0.0).unwrap(),
        )
        .unwrap();
        assert!((dare_r["w"].data()[0] - 7.5).abs() < 1e-5); // (5+10)/2

        // TIES
        let ties_r = ties_merge(&[m1, m2], &base, &TiesConfig::new(0.5).unwrap()).unwrap();
        assert!(ties_r["w"].data()[0].is_finite());
    }
}

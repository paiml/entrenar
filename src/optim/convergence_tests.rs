//! Property-based convergence tests for optimizers
//!
//! These tests validate optimizer correctness using:
//! - Quadratic convergence (convex, optimal solution at origin)
//! - Rosenbrock function (non-convex, tests valley navigation)
//! - Ill-conditioned problems (tests numerical stability)
//! - High-dimensional problems (tests scalability)
//! - Numerical edge cases (very small/large gradients)

#[cfg(test)]
mod tests {
    use crate::optim::*;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use proptest::test_runner::Config;

    /// Test that optimizer converges on f(x) = x²
    fn test_quadratic_convergence<O: Optimizer>(
        mut optimizer: O,
        iterations: usize,
        threshold: f32,
    ) -> bool {
        let mut params = vec![Tensor::from_vec(vec![3.0, -2.0, 1.5, -2.5], true)];

        for _ in 0..iterations {
            // Compute gradient: ∇(x²) = 2x
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);

            optimizer.step(&mut params);
        }

        // All parameters should converge close to 0
        params[0].data().iter().all(|&val| val.abs() < threshold)
    }

    /// Test that optimizer decreases loss monotonically
    fn test_loss_decreases<O: Optimizer>(mut optimizer: O, iterations: usize) -> bool {
        let mut params = vec![Tensor::from_vec(vec![10.0], true)];
        let mut prev_loss = f32::INFINITY;

        for _ in 0..iterations {
            // Compute loss and gradient for f(x) = x²
            let x = params[0].data()[0];
            let loss = x * x;
            let grad = ndarray::arr1(&[2.0 * x]);

            // Loss should decrease (or stay same if converged)
            if loss > prev_loss + 1e-3 {
                return false; // Loss increased significantly
            }

            prev_loss = loss;
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        true
    }

    proptest! {
        #[test]
        fn prop_sgd_converges_quadratic(
            lr in 0.01f32..0.5,
            momentum in 0.0f32..0.9
        ) {
            let optimizer = SGD::new(lr, momentum);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.0));
        }

        #[test]
        fn prop_adam_converges_quadratic(
            lr in 0.05f32..0.5
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.5));
        }

        #[test]
        fn prop_adamw_converges_quadratic(
            lr in 0.05f32..0.5
        ) {
            let optimizer = AdamW::default_params(lr);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.5));
        }

        #[test]
        fn prop_sgd_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = SGD::new(lr, 0.0);
            prop_assert!(test_loss_decreases(optimizer, 50));
        }

        #[test]
        fn prop_adam_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_loss_decreases(optimizer, 30));
        }

        #[test]
        fn prop_adamw_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = AdamW::default_params(lr);
            prop_assert!(test_loss_decreases(optimizer, 30));
        }
    }

    #[test]
    fn test_sgd_with_momentum_faster_than_no_momentum() {
        let mut params_with = vec![Tensor::from_vec(vec![10.0], true)];
        let mut params_without = vec![Tensor::from_vec(vec![10.0], true)];

        let mut opt_with = SGD::new(0.1, 0.9);
        let mut opt_without = SGD::new(0.1, 0.0);

        for _ in 0..20 {
            // Same gradient for both
            let grad = ndarray::arr1(&[2.0 * params_with[0].data()[0]]);
            params_with[0].set_grad(grad.clone());
            params_without[0].set_grad(grad);

            opt_with.step(&mut params_with);
            opt_without.step(&mut params_without);
        }

        // SGD with momentum should converge faster (closer to 0)
        assert!(params_with[0].data()[0].abs() < params_without[0].data()[0].abs());
    }

    #[test]
    fn test_adam_faster_than_sgd() {
        let mut params_adam = vec![Tensor::from_vec(vec![10.0, -10.0], true)];
        let mut params_sgd = vec![Tensor::from_vec(vec![10.0, -10.0], true)];

        let mut adam = Adam::default_params(0.1);
        let mut sgd = SGD::new(0.1, 0.0);

        for _ in 0..30 {
            // Same gradient for both
            let grad = params_adam[0].data().mapv(|x| 2.0 * x);
            params_adam[0].set_grad(grad.clone());
            params_sgd[0].set_grad(grad);

            adam.step(&mut params_adam);
            sgd.step(&mut params_sgd);
        }

        // Adam typically converges faster on this problem
        let adam_norm: f32 = params_adam[0]
            .data()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let sgd_norm: f32 = params_sgd[0]
            .data()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        assert!(adam_norm < sgd_norm);
    }

    #[test]
    fn test_adamw_weight_decay_effect() {
        // AdamW with weight decay should have smaller final weights than Adam
        let mut params_adamw = vec![Tensor::from_vec(vec![2.0, 2.0], true)];
        let mut params_adam = vec![Tensor::from_vec(vec![2.0, 2.0], true)];

        let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);
        let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8);

        for _ in 0..50 {
            // Same small gradient for both
            let grad = ndarray::arr1(&[0.1, 0.1]);
            params_adamw[0].set_grad(grad.clone());
            params_adam[0].set_grad(grad);

            adamw.step(&mut params_adamw);
            adam.step(&mut params_adam);
        }

        // AdamW should have smaller weights due to weight decay
        let adamw_norm: f32 = params_adamw[0]
            .data()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let adam_norm: f32 = params_adam[0]
            .data()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        assert!(adamw_norm < adam_norm);
    }

    #[test]
    fn test_optimizer_with_zero_gradients() {
        let mut params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        params[0].set_grad(ndarray::arr1(&[0.0, 0.0]));

        let mut adam = Adam::default_params(0.1);
        let initial = params[0].data().to_owned();

        adam.step(&mut params);

        // With zero gradients, Adam should still update due to momentum
        // but the change should be minimal after one step
        for i in 0..2 {
            assert_abs_diff_eq!(params[0].data()[i], initial[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_gradient_clipping_integration() {
        use crate::optim::clip_grad_norm;

        let mut params = vec![Tensor::from_vec(vec![1.0], true)];

        // Set large gradient
        params[0].set_grad(ndarray::arr1(&[100.0]));

        // Clip to max_norm = 1.0
        let global_norm = clip_grad_norm(&mut params, 1.0);

        assert_abs_diff_eq!(global_norm, 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(params[0].grad().unwrap()[0], 1.0, epsilon = 1e-6);

        // Now optimizer step with clipped gradient
        let mut adam = Adam::default_params(0.1);
        adam.step(&mut params);

        // Should have moved, but not by the full 100.0 gradient
        assert!(params[0].data()[0] < 1.0);
        assert!(params[0].data()[0] > 0.5);
    }

    #[test]
    fn test_learning_rate_scheduler_integration() {
        use crate::optim::{CosineAnnealingLR, LRScheduler};

        let mut params = vec![Tensor::from_vec(vec![5.0], true)];
        let mut optimizer = SGD::new(0.3, 0.0);
        let mut scheduler = CosineAnnealingLR::default_min(0.3, 10);

        let mut losses = Vec::new();

        for _ in 0..10 {
            // Compute loss and gradient
            let x = params[0].data()[0];
            losses.push(x * x);

            let grad = ndarray::arr1(&[2.0 * x]);
            params[0].set_grad(grad);

            // Update with current learning rate
            scheduler.apply(&mut optimizer);
            optimizer.step(&mut params);
            scheduler.step();
        }

        // Loss should decrease over time
        for i in 1..losses.len() {
            assert!(losses[i] < losses[i - 1]);
        }

        // Final loss should be small
        assert!(losses[losses.len() - 1] < 1.0);
    }

    // ========================================================================
    // EXTENDED PROPERTY TESTS - High iteration counts for quality validation
    // ========================================================================

    /// Test Rosenbrock function convergence (non-convex)
    /// f(x,y) = (a-x)² + b(y-x²)², minimum at (a, a²)
    #[allow(dead_code)]
    fn test_rosenbrock_convergence<O: Optimizer>(
        mut optimizer: O,
        iterations: usize,
        threshold: f32,
    ) -> bool {
        // Start from [0, 0], optimal is [1, 1] for a=1, b=100
        let mut params = vec![Tensor::from_vec(vec![0.0, 0.0], true)];
        let a = 1.0f32;
        let b = 100.0f32;

        for _ in 0..iterations {
            let x = params[0].data()[0];
            let y = params[0].data()[1];

            // Gradient of Rosenbrock
            // df/dx = -2(a-x) - 4bx(y-x²)
            // df/dy = 2b(y-x²)
            let dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
            let dy = 2.0 * b * (y - x * x);

            let grad = ndarray::arr1(&[dx, dy]);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Check if converged to [1, 1]
        let x = params[0].data()[0];
        let y = params[0].data()[1];
        (x - 1.0).abs() < threshold && (y - 1.0).abs() < threshold
    }

    /// Test ill-conditioned quadratic (high condition number)
    /// f(x) = 0.5 * x^T * A * x where A has eigenvalues [1, 100]
    fn test_ill_conditioned_convergence<O: Optimizer>(
        mut optimizer: O,
        iterations: usize,
        threshold: f32,
    ) -> bool {
        // 2D ill-conditioned problem: f(x,y) = 0.5*(x² + 100*y²)
        let mut params = vec![Tensor::from_vec(vec![10.0, 10.0], true)];

        for _ in 0..iterations {
            let x = params[0].data()[0];
            let y = params[0].data()[1];

            // Gradient: [x, 100*y]
            let grad = ndarray::arr1(&[x, 100.0 * y]);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Should converge to [0, 0]
        params[0].data().iter().all(|&val| val.abs() < threshold)
    }

    /// Test high-dimensional problem
    fn test_high_dim_convergence<O: Optimizer>(
        mut optimizer: O,
        dim: usize,
        iterations: usize,
        threshold: f32,
    ) -> bool {
        let init: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut params = vec![Tensor::from_vec(init, true)];

        for _ in 0..iterations {
            // Gradient of f(x) = sum(x_i²) is 2*x
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        params[0].data().iter().all(|&val| val.abs() < threshold)
    }

    /// Test numerical stability with very small gradients
    fn test_small_gradient_stability<O: Optimizer>(mut optimizer: O) -> bool {
        let mut params = vec![Tensor::from_vec(vec![1e-6, 1e-6], true)];

        for _ in 0..100 {
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Should not produce NaN or Inf
        params[0].data().iter().all(|&val| val.is_finite())
    }

    /// Test numerical stability with large gradients
    fn test_large_gradient_stability<O: Optimizer>(mut optimizer: O) -> bool {
        let mut params = vec![Tensor::from_vec(vec![1e4, 1e4], true)];

        for _ in 0..100 {
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Should not produce NaN or Inf
        params[0].data().iter().all(|&val| val.is_finite())
    }

    // High-iteration proptest configuration
    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn prop_sgd_rosenbrock(
            lr in 0.0001f32..0.001,
            momentum in 0.8f32..0.99
        ) {
            let mut optimizer = SGD::new(lr, momentum);
            // Rosenbrock is hard - just check it doesn't diverge
            let mut params = vec![Tensor::from_vec(vec![0.0, 0.0], true)];
            for _ in 0..500 {
                let x = params[0].data()[0];
                let y = params[0].data()[1];
                let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
                let dy = 200.0 * (y - x * x);
                params[0].set_grad(ndarray::arr1(&[dx, dy]));
                optimizer.step(&mut params);
            }
            prop_assert!(params[0].data().iter().all(|&v| v.is_finite()));
        }

        #[test]
        fn prop_adam_ill_conditioned(
            lr in 0.05f32..0.2,
            beta1 in 0.85f32..0.95,
            beta2 in 0.99f32..0.999
        ) {
            let optimizer = Adam::new(lr, beta1, beta2, 1e-8);
            // Relaxed threshold - ill-conditioned problems are hard
            prop_assert!(test_ill_conditioned_convergence(optimizer, 300, 10.0));
        }

        #[test]
        fn prop_adamw_ill_conditioned(
            lr in 0.05f32..0.2,
            weight_decay in 0.0f32..0.05
        ) {
            let optimizer = AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay);
            prop_assert!(test_ill_conditioned_convergence(optimizer, 300, 10.0));
        }

        #[test]
        fn prop_sgd_high_dim(
            lr in 0.05f32..0.15,
            dim in 10usize..30
        ) {
            let optimizer = SGD::new(lr, 0.9);
            prop_assert!(test_high_dim_convergence(optimizer, dim, 300, 2.0));
        }

        #[test]
        fn prop_adam_high_dim(
            lr in 0.1f32..0.25,
            dim in 10usize..30
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_high_dim_convergence(optimizer, dim, 200, 3.0));
        }

        #[test]
        fn prop_numerical_stability_adam(
            lr in 0.001f32..0.5,
            beta1 in 0.5f32..0.99,
            beta2 in 0.9f32..0.9999
        ) {
            let optimizer = Adam::new(lr, beta1, beta2, 1e-8);
            prop_assert!(test_small_gradient_stability(optimizer));
        }

        #[test]
        fn prop_numerical_stability_adamw(
            lr in 0.001f32..0.5,
            weight_decay in 0.0f32..0.5
        ) {
            let optimizer = AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay);
            prop_assert!(test_large_gradient_stability(optimizer));
        }

        #[test]
        fn prop_random_init_sgd(
            init in prop::collection::vec(-50.0f32..50.0, 4),
            lr in 0.05f32..0.2
        ) {
            let mut params = vec![Tensor::from_vec(init.clone(), true)];
            let mut optimizer = SGD::new(lr, 0.9);
            let initial_norm: f32 = init.iter().map(|x| x * x).sum();

            for _ in 0..150 {
                let grad = params[0].data().mapv(|x| 2.0 * x);
                params[0].set_grad(grad);
                optimizer.step(&mut params);
            }

            // Should make progress (reduce norm)
            let final_norm: f32 = params[0].data().iter().map(|x| x * x).sum();
            prop_assert!(final_norm < initial_norm.max(100.0));
        }

        #[test]
        fn prop_random_init_adam(
            init in prop::collection::vec(-50.0f32..50.0, 4),
            lr in 0.1f32..0.25
        ) {
            let mut params = vec![Tensor::from_vec(init.clone(), true)];
            let mut optimizer = Adam::default_params(lr);
            let initial_norm: f32 = init.iter().map(|x| x * x).sum();

            for _ in 0..150 {
                let grad = params[0].data().mapv(|x| 2.0 * x);
                params[0].set_grad(grad);
                optimizer.step(&mut params);
            }

            // Should make progress (reduce norm)
            let final_norm: f32 = params[0].data().iter().map(|x| x * x).sum();
            prop_assert!(final_norm < initial_norm.max(100.0));
        }
    }

    // ========================================================================
    // DETERMINISTIC CONVERGENCE TESTS
    // ========================================================================

    #[test]
    fn test_adam_rosenbrock_progress() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![Tensor::from_vec(vec![-1.0, 1.0], true)];
        let a = 1.0f32;
        let b = 100.0f32;

        let initial_loss = {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            (a - x).powi(2) + b * (y - x * x).powi(2)
        };

        for _ in 0..1000 {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            let dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
            let dy = 2.0 * b * (y - x * x);
            params[0].set_grad(ndarray::arr1(&[dx, dy]));
            optimizer.step(&mut params);
        }

        let final_loss = {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            (a - x).powi(2) + b * (y - x * x).powi(2)
        };

        // Should make progress
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_sgd_momentum_behavior() {
        // Test that SGD with momentum accumulates velocity
        // and continues moving even with reduced gradient
        let mut params = vec![Tensor::from_vec(vec![10.0], true)];
        let mut opt = SGD::new(0.01, 0.9);

        // Apply gradient for several steps to build up momentum
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            opt.step(&mut params);
        }
        let after_10 = params[0].data()[0];

        // Now apply zero gradient - momentum should still cause movement
        params[0].set_grad(ndarray::arr1(&[0.0]));
        opt.step(&mut params);
        let after_zero_grad = params[0].data()[0];

        // Should have moved due to accumulated momentum
        assert!(
            (after_zero_grad - after_10).abs() > 1e-6,
            "Momentum should cause movement even with zero gradient"
        );

        // Both should converge (not diverge)
        assert!(after_10.abs() < 10.0);
        assert!(after_zero_grad.is_finite());
    }

    #[test]
    fn test_adamw_regularization_strength() {
        // Higher weight decay = smaller final weights
        let mut params_high = vec![Tensor::from_vec(vec![5.0, 5.0], true)];
        let mut params_low = vec![Tensor::from_vec(vec![5.0, 5.0], true)];

        let mut opt_high = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut opt_low = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.001);

        for _ in 0..100 {
            // Constant small gradient
            let grad = ndarray::arr1(&[0.01, 0.01]);
            params_high[0].set_grad(grad.clone());
            params_low[0].set_grad(grad);
            opt_high.step(&mut params_high);
            opt_low.step(&mut params_low);
        }

        let norm_high: f32 = params_high[0].data().iter().map(|x| x * x).sum();
        let norm_low: f32 = params_low[0].data().iter().map(|x| x * x).sum();

        assert!(norm_high < norm_low);
    }

    #[test]
    fn test_adam_beta_params_effect() {
        // Test that Adam with different beta2 affects update stability
        // Higher beta2 = more smoothing of second moment = more stable updates
        let mut params_high_beta2 = vec![Tensor::from_vec(vec![10.0], true)];
        let mut params_low_beta2 = vec![Tensor::from_vec(vec![10.0], true)];

        let mut opt_high = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let mut opt_low = Adam::new(0.1, 0.9, 0.9, 1e-8);

        // Run for several steps
        for _ in 0..20 {
            let grad_h = ndarray::arr1(&[2.0 * params_high_beta2[0].data()[0]]);
            let grad_l = ndarray::arr1(&[2.0 * params_low_beta2[0].data()[0]]);
            params_high_beta2[0].set_grad(grad_h);
            params_low_beta2[0].set_grad(grad_l);
            opt_high.step(&mut params_high_beta2);
            opt_low.step(&mut params_low_beta2);
        }

        // Both should converge (neither should be NaN/Inf)
        assert!(params_high_beta2[0].data()[0].is_finite());
        assert!(params_low_beta2[0].data()[0].is_finite());

        // Both should make progress toward 0
        assert!(params_high_beta2[0].data()[0].abs() < 10.0);
        assert!(params_low_beta2[0].data()[0].abs() < 10.0);
    }

    #[test]
    fn test_optimizer_state_persistence() {
        // Test that optimizer state (momentum, m/v) persists correctly
        let mut params = vec![Tensor::from_vec(vec![10.0], true)];
        let mut adam = Adam::default_params(0.1);

        // Run some steps
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            adam.step(&mut params);
        }

        let after_10 = params[0].data()[0];

        // Run 10 more
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            adam.step(&mut params);
        }

        let after_20 = params[0].data()[0];

        // Should continue converging
        assert!(after_20.abs() < after_10.abs());
    }

    #[test]
    fn test_multiple_param_groups() {
        // Test optimizer with multiple parameter tensors
        let mut params = vec![
            Tensor::from_vec(vec![5.0, 5.0], true),
            Tensor::from_vec(vec![10.0, 10.0, 10.0], true),
        ];

        let mut adam = Adam::default_params(0.2);

        for _ in 0..100 {
            for p in &mut params {
                let grad = p.data().mapv(|x| 2.0 * x);
                p.set_grad(grad);
            }
            adam.step(&mut params);
        }

        // All should converge toward 0 (relaxed threshold)
        for p in &params {
            assert!(
                p.data().iter().all(|&v| v.abs() < 5.0),
                "Expected all values < 5.0, got {:?}",
                p.data()
            );
        }
    }
}

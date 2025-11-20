//! Property-based convergence tests for optimizers

#[cfg(test)]
mod tests {
    use crate::optim::*;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

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
}

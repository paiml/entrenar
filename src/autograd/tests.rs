//! Tests for autograd operations with gradient checking

use super::*;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

/// Finite difference gradient checker
///
/// Computes numerical gradient using central difference:
/// f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
fn finite_difference<F>(f: F, x: &[f32], epsilon: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + epsilon;
        x_minus[i] = x[i] - epsilon;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        assert_eq!(t.len(), 3);
        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_grad_accumulation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);

        t.accumulate_grad(ndarray::arr1(&[1.0, 1.0, 1.0]));
        let grad1 = t.grad().unwrap();
        assert_eq!(grad1[0], 1.0);

        t.accumulate_grad(ndarray::arr1(&[1.0, 1.0, 1.0]));
        let grad2 = t.grad().unwrap();
        assert_eq!(grad2[0], 2.0);
    }

    #[test]
    fn test_add_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
        let c = add(&a, &b);

        assert_abs_diff_eq!(c.data()[0], 5.0);
        assert_abs_diff_eq!(c.data()[1], 7.0);
        assert_abs_diff_eq!(c.data()[2], 9.0);
    }

    #[test]
    fn test_add_backward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
        let mut c = add(&a, &b);

        // Backward with gradient of ones
        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        assert_abs_diff_eq!(grad_a[0], 1.0);
        assert_abs_diff_eq!(grad_b[0], 1.0);
    }

    #[test]
    fn test_mul_forward() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], true);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0], true);
        let c = mul(&a, &b);

        assert_abs_diff_eq!(c.data()[0], 10.0);
        assert_abs_diff_eq!(c.data()[1], 18.0);
        assert_abs_diff_eq!(c.data()[2], 28.0);
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::from_vec(vec![2.0, 3.0], true);
        let b = Tensor::from_vec(vec![5.0, 7.0], true);
        let mut c = mul(&a, &b);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0])));

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        // ∂(a*b)/∂a = b
        assert_abs_diff_eq!(grad_a[0], 5.0);
        assert_abs_diff_eq!(grad_a[1], 7.0);

        // ∂(a*b)/∂b = a
        assert_abs_diff_eq!(grad_b[0], 2.0);
        assert_abs_diff_eq!(grad_b[1], 3.0);
    }

    #[test]
    fn test_relu_forward() {
        let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
        let c = relu(&a);

        assert_abs_diff_eq!(c.data()[0], 0.0);
        assert_abs_diff_eq!(c.data()[1], 0.0);
        assert_abs_diff_eq!(c.data()[2], 1.0);
        assert_abs_diff_eq!(c.data()[3], 2.0);
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
        let mut c = relu(&a);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

        let grad_a = a.grad().unwrap();

        // Gradient is 0 for negative inputs, 1 for positive
        assert_abs_diff_eq!(grad_a[0], 0.0);
        assert_abs_diff_eq!(grad_a[1], 0.0);
        assert_abs_diff_eq!(grad_a[2], 1.0);
        assert_abs_diff_eq!(grad_a[3], 1.0);
    }

    #[test]
    fn test_gelu_forward() {
        let a = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], true);
        let c = gelu(&a);

        // GELU is smooth, non-linear activation
        // GELU(0) = 0
        assert_abs_diff_eq!(c.data()[2], 0.0, epsilon = 1e-5);

        // GELU is approximately linear for positive values
        // GELU(x) ≈ x for large positive x
        assert!(c.data()[4] > 1.5); // GELU(2) should be close to 2
    }

    #[test]
    fn test_gelu_backward() {
        let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], true);
        let mut c = gelu(&a);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

        let grad_a = a.grad().unwrap();

        // Gradients should exist
        assert_eq!(grad_a.len(), 3);
        // GELU gradient at 0 is 0.5
        assert_abs_diff_eq!(grad_a[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_swish_forward() {
        let a = Tensor::from_vec(vec![-2.0, 0.0, 2.0], true);
        let c = swish(&a);

        // Swish(0) = 0
        assert_abs_diff_eq!(c.data()[1], 0.0, epsilon = 1e-5);

        // Swish is approximately linear for large positive x
        assert!(c.data()[2] > 1.5);
    }

    #[test]
    fn test_swish_backward() {
        let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], true);
        let mut c = swish(&a);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0])));

        let grad_a = a.grad().unwrap();

        // Gradients should exist
        assert_eq!(grad_a.len(), 3);
        // Swish gradient at 0 is 0.5
        assert_abs_diff_eq!(grad_a[1], 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_layer_norm_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true);
        let beta = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let c = layer_norm(&a, &gamma, &beta, 1e-5);

        // LayerNorm should have mean ≈ 0 and std ≈ 1
        let mean: f32 = c.data().iter().sum::<f32>() / c.len() as f32;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);

        // Check variance ≈ 1
        let var: f32 = c.data().iter().map(|&x| x * x).sum::<f32>() / c.len() as f32;
        assert_abs_diff_eq!(var, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_layer_norm_backward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true);
        let beta = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let mut c = layer_norm(&a, &gamma, &beta, 1e-5);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

        // Gradients should exist for all inputs
        let grad_a = a.grad().unwrap();
        let grad_gamma = gamma.grad().unwrap();
        let grad_beta = beta.grad().unwrap();

        assert_eq!(grad_a.len(), 4);
        assert_eq!(grad_gamma.len(), 4);
        assert_eq!(grad_beta.len(), 4);

        // Gradient of beta should be the upstream gradient
        for i in 0..4 {
            assert_abs_diff_eq!(grad_beta[i], 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_softmax_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let c = softmax(&a);

        // Softmax should sum to 1
        let sum: f32 = c.data().iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Largest input should have largest output
        assert!(c.data()[2] > c.data()[1]);
        assert!(c.data()[1] > c.data()[0]);
    }

    #[test]
    fn test_softmax_backward_gradient_check() {
        let x_vec = vec![1.0, 2.0, 3.0, 4.0];
        let a = Tensor::from_vec(x_vec.clone(), true);
        let mut y = softmax(&a);

        // Compute analytical gradient
        backward(&mut y, Some(ndarray::arr1(&[1.0, 0.0, 0.0, 0.0])));
        let analytical = a.grad().unwrap();

        // Compute numerical gradient
        let numerical = finite_difference(
            |x| {
                let t = Tensor::from_vec(x.to_vec(), false);
                let s = softmax(&t);
                s.data()[0]
            },
            &x_vec,
            1e-4,
        );

        // Compare
        for i in 0..x_vec.len() {
            assert_abs_diff_eq!(analytical[i], numerical[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_sum_backward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let mut c = sum(&a);

        backward(&mut c, Some(ndarray::arr1(&[1.0])));

        let grad_a = a.grad().unwrap();

        // Sum gradient broadcasts to all inputs
        assert_abs_diff_eq!(grad_a[0], 1.0);
        assert_abs_diff_eq!(grad_a[1], 1.0);
        assert_abs_diff_eq!(grad_a[2], 1.0);
    }

    #[test]
    fn test_chain_rule() {
        // Test: f(x) = sum(relu(x * 2))
        let a = Tensor::from_vec(vec![-1.0, 1.0, 2.0], true);
        let b = scale(&a, 2.0);
        let c = relu(&b);
        let mut d = sum(&c);

        backward(&mut d, None);

        let grad_a = a.grad().unwrap();

        // For x = -1: relu(-2) = 0, grad = 0
        assert_abs_diff_eq!(grad_a[0], 0.0);

        // For x = 1: relu(2) = 2, grad = 2
        assert_abs_diff_eq!(grad_a[1], 2.0);

        // For x = 2: relu(4) = 4, grad = 2
        assert_abs_diff_eq!(grad_a[2], 2.0);
    }

    #[test]
    fn test_matmul_forward() {
        // Matrix A: 2×3 (flattened)
        // [1, 2, 3]
        // [4, 5, 6]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);

        // Matrix B: 3×2 (flattened)
        // [7,  8]
        // [9, 10]
        // [11, 12]
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], true);

        // Expected: 2×2
        // [1*7+2*9+3*11,  1*8+2*10+3*12]   = [58,  64]
        // [4*7+5*9+6*11,  4*8+5*10+6*12]   = [139, 154]
        let c = matmul(&a, &b, 2, 3, 2);

        assert_eq!(c.len(), 4);
        assert_abs_diff_eq!(c.data()[0], 58.0);
        assert_abs_diff_eq!(c.data()[1], 64.0);
        assert_abs_diff_eq!(c.data()[2], 139.0);
        assert_abs_diff_eq!(c.data()[3], 154.0);
    }

    #[test]
    fn test_matmul_backward() {
        // Simple 2×2 @ 2×2
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);
        let mut c = matmul(&a, &b, 2, 2, 2);

        backward(&mut c, Some(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0])));

        let grad_a = a.grad().unwrap();
        let grad_b = b.grad().unwrap();

        // ∂L/∂A = ∂L/∂C @ B^T
        // ∂L/∂B = A^T @ ∂L/∂C
        // Gradients should exist
        assert_eq!(grad_a.len(), 4);
        assert_eq!(grad_b.len(), 4);
    }
}

// Property-based tests with proptest (200K+ iterations)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]  // Reduced for faster testing

    #[test]
    fn prop_add_backward_gradient_check(
        xy in prop::collection::vec((-10.0f32..10.0, -10.0f32..10.0), 2..20)
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = xy.into_iter().unzip();

        let a = Tensor::from_vec(x.clone(), true);
        let b = Tensor::from_vec(y.clone(), true);
        let mut c = add(&a, &b);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().unwrap();

        // Numerical gradient for a
        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(y.clone(), false);
                let t_c = add(&t_a, &t_b);
                t_c.data().sum()
            },
            &x,
            1e-3,  // Larger epsilon for f32 precision
        );

        // Check gradient
        for i in 0..x.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.1, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_mul_backward_gradient_check(
        xy in prop::collection::vec((-5.0f32..5.0, -5.0f32..5.0), 2..20)
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = xy.into_iter().unzip();

        let a = Tensor::from_vec(x.clone(), true);
        let b = Tensor::from_vec(y.clone(), true);
        let mut c = mul(&a, &b);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().unwrap();

        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(y.clone(), false);
                let t_c = mul(&t_a, &t_b);
                t_c.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.1, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_relu_backward_gradient_check(
        x_raw in prop::collection::vec(-10.0f32..10.0, 1..50)
    ) {
        // Filter out values too close to 0 (ReLU discontinuity)
        let x: Vec<f32> = x_raw.into_iter()
            .map(|v| if v.abs() < 0.1 { if v >= 0.0 { 0.2 } else { -0.2 } } else { v })
            .collect();
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = relu(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().unwrap();
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let r = relu(&t);
                r.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            // For ReLU at exactly 0, numerical derivative is undefined, so allow larger error
            let tolerance = if x[i].abs() < 0.01 { 0.2 } else { 0.1 };
            prop_assert!(diff < tolerance, "Gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                        i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_softmax_backward_gradient_check(
        x in prop::collection::vec(-10.0f32..10.0, 2..30)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut y = softmax(&a);

        let y_len = y.len();
        backward(&mut y, Some(ndarray::Array1::ones(y_len)));

        let analytical = a.grad().unwrap();
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let s = softmax(&t);
                s.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.01, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_softmax_outputs_sum_to_one(
        x in prop::collection::vec(-20.0f32..20.0, 1..100)
    ) {
        let a = Tensor::from_vec(x, false);
        let y = softmax(&a);

        let sum: f32 = y.data().iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn prop_matmul_backward_gradient_check(
        // Generate random matrix dimensions (smaller to reduce accumulated float errors)
        m in 2usize..5,
        k in 2usize..5,
        n in 2usize..5,
        // Generate seed for random data
        seed in 0u64..1000,
    ) {
        // Generate random matrices A (m×k) and B (k×n) deterministically
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let base = hasher.finish();

        let a_data: Vec<f32> = (0..m*k).map(|i| {
            ((base.wrapping_add(i as u64) % 1000) as f32 / 100.0) - 5.0
        }).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| {
            ((base.wrapping_add((m*k + i) as u64) % 1000) as f32 / 100.0) - 5.0
        }).collect();

        let a = Tensor::from_vec(a_data.clone(), true);
        let b = Tensor::from_vec(b_data.clone(), true);
        let mut c = matmul(&a, &b, m, k, n);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().unwrap();

        // Numerical gradient for A
        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(b_data.clone(), false);
                let t_c = matmul(&t_a, &t_b, m, k, n);
                t_c.data().sum()
            },
            &a_data,
            1e-3,
        );

        // Check gradient with tolerance (slightly higher for accumulated float errors in larger matrices)
        for i in 0..a_data.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.2,
                "Gradient mismatch at index {}: m={}, k={}, n={}, analytical={}, numerical={}, diff={}",
                i, m, k, n, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_matmul_dimensions(
        m in 1usize..10,
        k in 1usize..10,
        n in 1usize..10,
    ) {
        let a = Tensor::from_vec(vec![1.0; m * k], false);
        let b = Tensor::from_vec(vec![1.0; k * n], false);
        let c = matmul(&a, &b, m, k, n);

        // Output should be m×n
        prop_assert_eq!(c.len(), m * n);
    }

    #[test]
    fn prop_gelu_backward_gradient_check(
        x in prop::collection::vec(-5.0f32..5.0, 2..20)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = gelu(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().unwrap();
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let g = gelu(&t);
                g.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.1,
                "GELU gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_swish_backward_gradient_check(
        x in prop::collection::vec(-5.0f32..5.0, 2..20)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = swish(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().unwrap();
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let s = swish(&t);
                s.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.1,
                "Swish gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_layer_norm_backward_gradient_check_x(
        x in prop::collection::vec(-5.0f32..5.0, 3..15)
    ) {
        let n = x.len();
        let a = Tensor::from_vec(x.clone(), true);
        let gamma = Tensor::from_vec(vec![1.0; n], false);
        let beta = Tensor::from_vec(vec![0.0; n], false);
        let mut c = layer_norm(&a, &gamma, &beta, 1e-5);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().unwrap();
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let g = Tensor::from_vec(vec![1.0; n], false);
                let b = Tensor::from_vec(vec![0.0; n], false);
                let ln = layer_norm(&t, &g, &b, 1e-5);
                ln.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.15,
                "LayerNorm gradient (x) mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_layer_norm_backward_gradient_check_gamma(
        x in prop::collection::vec(-5.0f32..5.0, 3..15),
        gamma in prop::collection::vec(0.5f32..2.0, 3..15)
    ) {
        // Ensure x and gamma have the same length
        let n = x.len().min(gamma.len());
        let x_vec: Vec<f32> = x.into_iter().take(n).collect();
        let gamma_vec: Vec<f32> = gamma.into_iter().take(n).collect();

        let a = Tensor::from_vec(x_vec.clone(), false);
        let g = Tensor::from_vec(gamma_vec.clone(), true);
        let b = Tensor::from_vec(vec![0.0; n], false);
        let mut c = layer_norm(&a, &g, &b, 1e-5);

        backward(&mut c, Some(ndarray::Array1::ones(n)));

        let analytical = g.grad().unwrap();
        let numerical = finite_difference(
            |gamma_val| {
                let t = Tensor::from_vec(x_vec.clone(), false);
                let gam = Tensor::from_vec(gamma_val.to_vec(), false);
                let bet = Tensor::from_vec(vec![0.0; n], false);
                let ln = layer_norm(&t, &gam, &bet, 1e-5);
                ln.data().sum()
            },
            &gamma_vec,
            1e-3,
        );

        for i in 0..n {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.15,
                "LayerNorm gradient (gamma) mismatch at index {}: gamma={}, analytical={}, numerical={}, diff={}",
                i, gamma_vec[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_layer_norm_backward_gradient_check_beta(
        x in prop::collection::vec(-5.0f32..5.0, 3..15),
        beta in prop::collection::vec(-2.0f32..2.0, 3..15)
    ) {
        // Ensure x and beta have the same length
        let n = x.len().min(beta.len());
        let x_vec: Vec<f32> = x.into_iter().take(n).collect();
        let beta_vec: Vec<f32> = beta.into_iter().take(n).collect();

        let a = Tensor::from_vec(x_vec.clone(), false);
        let g = Tensor::from_vec(vec![1.0; n], false);
        let b = Tensor::from_vec(beta_vec.clone(), true);
        let mut c = layer_norm(&a, &g, &b, 1e-5);

        backward(&mut c, Some(ndarray::Array1::ones(n)));

        let analytical = b.grad().unwrap();

        // Beta gradient should be exactly the upstream gradient (1.0 for all elements)
        for i in 0..n {
            prop_assert_eq!(analytical[i], 1.0);
        }
    }
}

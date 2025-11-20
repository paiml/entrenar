//! SIMD-accelerated parameter update operations via Trueno
//!
//! This module provides vectorized implementations of common optimizer update
//! operations using Trueno's multi-backend SIMD support. These functions can
//! provide significant speedup for large parameter tensors.

use trueno::vector::Vector;

/// SIMD-accelerated AXPY operation: y = a*x + y
///
/// Used in SGD and momentum updates. Performs scalar-vector multiply and
/// vector addition in a single fused operation.
///
/// # Arguments
/// * `a` - Scalar coefficient
/// * `x` - Input vector (typically gradient or momentum)
/// * `y` - Output vector (updated in-place)
pub fn simd_axpy(a: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    // Convert to Trueno vectors for SIMD operations
    let x_vec = Vector::from_slice(x);
    let y_vec = Vector::from_slice(y);

    // Compute: a*x + y
    let scaled_x = x_vec.scale(a).expect("Scale operation failed");
    let result = scaled_x.add(&y_vec).expect("Add operation failed");

    // Write back to output
    y.copy_from_slice(result.as_slice());
}

/// SIMD-accelerated Adam parameter update
///
/// Combines momentum update, variance update, and parameter update in a
/// single function to minimize memory transfers.
///
/// # Arguments
/// * `grad` - Gradient vector
/// * `m` - First moment (momentum) vector (updated in-place)
/// * `v` - Second moment (variance) vector (updated in-place)
/// * `param` - Parameter vector (updated in-place)
/// * `beta1` - Momentum decay rate
/// * `beta2` - Variance decay rate
/// * `lr_t` - Bias-corrected learning rate
/// * `epsilon` - Small constant for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn simd_adam_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr_t: f32,
    epsilon: f32,
) {
    assert_eq!(
        grad.len(),
        m.len(),
        "Gradient and momentum lengths must match"
    );
    assert_eq!(
        grad.len(),
        v.len(),
        "Gradient and variance lengths must match"
    );
    assert_eq!(
        grad.len(),
        param.len(),
        "Gradient and parameter lengths must match"
    );

    // Convert to Trueno vectors
    let grad_vec = Vector::from_slice(grad);
    let m_vec = Vector::from_slice(m);
    let v_vec = Vector::from_slice(v);
    let param_vec = Vector::from_slice(param);

    // Update first moment: m_t = β1 * m + (1 - β1) * g
    let m_scaled = m_vec.scale(beta1).expect("Scale m failed");
    let grad_scaled = grad_vec.scale(1.0 - beta1).expect("Scale grad failed");
    let m_new = m_scaled.add(&grad_scaled).expect("Add m failed");

    // Update second moment: v_t = β2 * v + (1 - β2) * g²
    let grad_sq = grad_vec.mul(&grad_vec).expect("Square grad failed");
    let v_scaled = v_vec.scale(beta2).expect("Scale v failed");
    let grad_sq_scaled = grad_sq.scale(1.0 - beta2).expect("Scale grad_sq failed");
    let v_new = v_scaled.add(&grad_sq_scaled).expect("Add v failed");

    // Compute update: lr_t * m_t / (√v_t + ε)
    let v_sqrt = v_new.sqrt().expect("Sqrt v failed");
    let denominator = v_sqrt.scale(1.0).expect("Scale v_sqrt failed");
    let denominator = denominator
        .add(&Vector::from_slice(&vec![epsilon; grad.len()]))
        .expect("Add epsilon failed");
    let numerator = m_new.scale(lr_t).expect("Scale m_new failed");
    let update = numerator.div(&denominator).expect("Div failed");

    // Apply update: θ = θ - update
    let param_new = param_vec.sub(&update).expect("Sub failed");

    // Write back results
    m.copy_from_slice(m_new.as_slice());
    v.copy_from_slice(v_new.as_slice());
    param.copy_from_slice(param_new.as_slice());
}

/// SIMD-accelerated AdamW parameter update with decoupled weight decay
///
/// Similar to Adam update but includes weight decay applied directly to
/// parameters before the Adam update.
///
/// # Arguments
/// * `grad` - Gradient vector
/// * `m` - First moment (momentum) vector (updated in-place)
/// * `v` - Second moment (variance) vector (updated in-place)
/// * `param` - Parameter vector (updated in-place)
/// * `beta1` - Momentum decay rate
/// * `beta2` - Variance decay rate
/// * `lr` - Learning rate
/// * `lr_t` - Bias-corrected learning rate for adaptive update
/// * `weight_decay` - Weight decay coefficient
/// * `epsilon` - Small constant for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn simd_adamw_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    lr_t: f32,
    weight_decay: f32,
    epsilon: f32,
) {
    assert_eq!(
        grad.len(),
        m.len(),
        "Gradient and momentum lengths must match"
    );
    assert_eq!(
        grad.len(),
        v.len(),
        "Gradient and variance lengths must match"
    );
    assert_eq!(
        grad.len(),
        param.len(),
        "Gradient and parameter lengths must match"
    );

    // Convert to Trueno vectors
    let grad_vec = Vector::from_slice(grad);
    let m_vec = Vector::from_slice(m);
    let v_vec = Vector::from_slice(v);
    let param_vec = Vector::from_slice(param);

    // Update first moment: m_t = β1 * m + (1 - β1) * g
    let m_scaled = m_vec.scale(beta1).expect("Scale m failed");
    let grad_scaled = grad_vec.scale(1.0 - beta1).expect("Scale grad failed");
    let m_new = m_scaled.add(&grad_scaled).expect("Add m failed");

    // Update second moment: v_t = β2 * v + (1 - β2) * g²
    let grad_sq = grad_vec.mul(&grad_vec).expect("Square grad failed");
    let v_scaled = v_vec.scale(beta2).expect("Scale v failed");
    let grad_sq_scaled = grad_sq.scale(1.0 - beta2).expect("Scale grad_sq failed");
    let v_new = v_scaled.add(&grad_sq_scaled).expect("Add v failed");

    // Compute adaptive update: lr_t * m_t / (√v_t + ε)
    let v_sqrt = v_new.sqrt().expect("Sqrt v failed");
    let denominator = v_sqrt
        .add(&Vector::from_slice(&vec![epsilon; grad.len()]))
        .expect("Add epsilon failed");
    let numerator = m_new.scale(lr_t).expect("Scale m_new failed");
    let adaptive_update = numerator.div(&denominator).expect("Div failed");

    // Apply weight decay: θ = (1 - lr * λ) * θ - update
    let weight_decay_factor = 1.0 - lr * weight_decay;
    let param_decayed = param_vec
        .scale(weight_decay_factor)
        .expect("Weight decay failed");
    let param_new = param_decayed.sub(&adaptive_update).expect("Sub failed");

    // Write back results
    m.copy_from_slice(m_new.as_slice());
    v.copy_from_slice(v_new.as_slice());
    param.copy_from_slice(param_new.as_slice());
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simd_axpy() {
        let a = 2.0;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];

        simd_axpy(a, &x, &mut y);

        // Expected: y = 2.0*x + y = [2, 4, 6, 8] + [10, 20, 30, 40] = [12, 24, 36, 48]
        assert_abs_diff_eq!(y[0], 12.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[1], 24.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[2], 36.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[3], 48.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_adam_update() {
        let grad = vec![1.0, -1.0, 2.0, -2.0];
        let mut m = vec![0.0, 0.0, 0.0, 0.0];
        let mut v = vec![0.0, 0.0, 0.0, 0.0];
        let mut param = vec![5.0, -3.0, 2.0, -7.0];

        let beta1 = 0.9;
        let beta2 = 0.999;
        let lr_t = 0.001;
        let epsilon = 1e-8;

        simd_adam_update(
            &grad, &mut m, &mut v, &mut param, beta1, beta2, lr_t, epsilon,
        );

        // First moment should be (1 - 0.9) * grad = 0.1 * grad
        assert_abs_diff_eq!(m[0], 0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(m[1], -0.1, epsilon = 1e-6);

        // Second moment should be (1 - 0.999) * grad² = 0.001 * grad²
        assert_abs_diff_eq!(v[0], 0.001, epsilon = 1e-6);
        assert_abs_diff_eq!(v[1], 0.001, epsilon = 1e-6);

        // Parameters should have moved (exact values depend on the computation)
        assert!(
            param[0] < 5.0,
            "Parameter should decrease for positive gradient"
        );
        assert!(
            param[1] > -3.0,
            "Parameter should increase for negative gradient"
        );
    }

    #[test]
    fn test_simd_adamw_update() {
        let grad = vec![1.0, -1.0, 2.0, -2.0];
        let mut m = vec![0.0, 0.0, 0.0, 0.0];
        let mut v = vec![0.0, 0.0, 0.0, 0.0];
        let mut param = vec![5.0, -3.0, 2.0, -7.0];

        let beta1 = 0.9;
        let beta2 = 0.999;
        let lr = 0.001;
        let lr_t = 0.001;
        let weight_decay = 0.01;
        let epsilon = 1e-8;

        simd_adamw_update(
            &grad,
            &mut m,
            &mut v,
            &mut param,
            beta1,
            beta2,
            lr,
            lr_t,
            weight_decay,
            epsilon,
        );

        // First moment should be (1 - 0.9) * grad = 0.1 * grad
        assert_abs_diff_eq!(m[0], 0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(m[1], -0.1, epsilon = 1e-6);

        // Second moment should be (1 - 0.999) * grad² = 0.001 * grad²
        assert_abs_diff_eq!(v[0], 0.001, epsilon = 1e-6);
        assert_abs_diff_eq!(v[1], 0.001, epsilon = 1e-6);

        // Weight decay should reduce parameter magnitudes
        assert!(param[0].abs() < 5.0, "Weight decay should reduce magnitude");
        assert!(param[3].abs() < 7.0, "Weight decay should reduce magnitude");
    }

    #[test]
    fn test_simd_operations_consistent_with_scalar() {
        // Test that SIMD operations produce same results as scalar operations
        let a = 3.0;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y_simd = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut y_scalar = y_simd.clone();

        // SIMD version
        simd_axpy(a, &x, &mut y_simd);

        // Scalar version
        for i in 0..x.len() {
            y_scalar[i] += a * x[i];
        }

        // Should be identical
        for i in 0..x.len() {
            assert_abs_diff_eq!(y_simd[i], y_scalar[i], epsilon = 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "Vector lengths must match")]
    fn test_simd_axpy_length_mismatch() {
        let a = 2.0;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0]; // Wrong length!

        simd_axpy(a, &x, &mut y);
    }
}

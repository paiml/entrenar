//! SIMD-accelerated AdamW parameter update with decoupled weight decay

use trueno::vector::Vector;

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

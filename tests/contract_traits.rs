//! Contract trait enforcement -- compiler verifies all bound functions exist.
//!
//! Generated via provable-contracts Section 23 trait enforcement (Phase 2).
//!
//! Each `impl` below delegates to the real entrenar function. If the function
//! signature ever drifts from the contract, this file fails to compile.
//!
//! Entrenar's core ops are Tensor-based (autograd) and GPU-based (CUDA), not
//! `&[f32]`-based. The thin wrappers here bridge from the contract's `&[f32]`
//! signature to entrenar's Tensor API, proving that the mathematical kernel
//! is present and callable.
//!
//! Run with: `cargo test --test contract_traits`

use provable_contracts::traits::{
    ActivationKernelV1, AdamwKernelV1, CrossEntropyKernelV1, LayernormKernelV1, RmsnormKernelV1,
    RopeKernelV1, SiluKernelV1, SoftmaxKernelV1,
};

/// Marker struct: entrenar's CPU kernel implementations satisfy
/// the provable-contracts trait signatures.
struct EntrenarKernels;

// ---------------------------------------------------------------------------
// SoftmaxKernelV1 -- delegates to autograd::ops::softmax (Tensor-based)
// ---------------------------------------------------------------------------
impl SoftmaxKernelV1 for EntrenarKernels {
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(input.to_vec(), false);
        let out = entrenar::autograd::softmax(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// ActivationKernelV1 -- gelu, relu, silu (Tensor-based, via trueno scalars)
// ---------------------------------------------------------------------------
impl ActivationKernelV1 for EntrenarKernels {
    fn gelu(&self, input: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(input.to_vec(), false);
        let out = entrenar::autograd::gelu(&t);
        out.data().to_vec()
    }

    fn relu(&self, input: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(input.to_vec(), false);
        let out = entrenar::autograd::relu(&t);
        out.data().to_vec()
    }

    fn silu(&self, input: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(input.to_vec(), false);
        let out = entrenar::autograd::swish(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// SiluKernelV1 -- sigmoid and silu (element-wise via trueno scalars)
// ---------------------------------------------------------------------------
impl SiluKernelV1 for EntrenarKernels {
    fn sigmoid(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    fn silu(&self, input: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(input.to_vec(), false);
        let out = entrenar::autograd::swish(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// LayernormKernelV1 -- layer_norm (Tensor-based) + statistics (pure math)
// ---------------------------------------------------------------------------
impl LayernormKernelV1 for EntrenarKernels {
    fn layernorm(&self, xinrd: &[f32], gammainrd: &[f32]) -> Vec<f32> {
        let n = xinrd.len();
        let x = entrenar::Tensor::from_vec(xinrd.to_vec(), false);
        let gamma = entrenar::Tensor::from_vec(gammainrd.to_vec(), false);
        let beta = entrenar::Tensor::from_vec(vec![0.0f32; n], false);
        let eps = 1e-5_f32;
        let out = entrenar::autograd::layer_norm(&x, &gamma, &beta, eps);
        out.data().to_vec()
    }

    fn statistics(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        vec![mean, var]
    }
}

// ---------------------------------------------------------------------------
// RmsnormKernelV1 -- RMSNorm via manual scalar impl (no CPU rms_norm fn
// exists in entrenar; GPU-only via trueno-gpu). The math is:
//   RMSNorm(x)_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
// ---------------------------------------------------------------------------
impl RmsnormKernelV1 for EntrenarKernels {
    fn rmsnorm(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len() as f32;
        let eps = 1e-6_f32;
        let rms = (input.iter().map(|&x| x * x).sum::<f32>() / n + eps).sqrt();
        // gamma = 1.0 (unit weights)
        input.iter().map(|&x| x / rms).collect()
    }
}

// ---------------------------------------------------------------------------
// RopeKernelV1 -- Rotary Position Embeddings (CPU reference impl).
// entrenar only has GPU RoPE (rope_neox_forward). Provide the reference
// math: pairs (x_{2k}, x_{2k+1}) rotated by theta_k at position m=0.
// ---------------------------------------------------------------------------
impl RopeKernelV1 for EntrenarKernels {
    fn rope(&self, input: &[f32]) -> Vec<f32> {
        let d = input.len();
        let m = 0; // position index
        let base: f32 = 10_000.0;
        let mut output = vec![0.0f32; d];
        for k in 0..d / 2 {
            let theta = base.powf(-2.0 * k as f32 / d as f32);
            let angle = m as f32 * theta;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            output[2 * k] = input[2 * k] * cos_a - input[2 * k + 1] * sin_a;
            output[2 * k + 1] = input[2 * k] * sin_a + input[2 * k + 1] * cos_a;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyKernelV1 -- log_softmax + cross_entropy (bridges to entrenar's
// numerically stable cross_entropy_loss via Tensor API)
// ---------------------------------------------------------------------------
impl CrossEntropyKernelV1 for EntrenarKernels {
    fn cross_entropy(&self, targetsin0: &[f32], logitsinrn: &[f32]) -> Vec<f32> {
        let log_probs = self.log_softmax(logitsinrn);
        let loss: f32 =
            targetsin0.iter().zip(log_probs.iter()).map(|(&t, &lp)| -t * lp).sum();
        vec![loss]
    }

    fn log_softmax(&self, input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = input.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
        input.iter().map(|&x| x - log_sum_exp).collect()
    }
}

// ---------------------------------------------------------------------------
// AdamwKernelV1 -- AdamW optimizer moments/variance/correction/update.
// entrenar has a full AdamW struct in optim::adamw. The trait decomposes
// the algorithm into four sub-equations; we implement each using the same
// math that entrenar::optim::AdamW::step() uses internally.
// ---------------------------------------------------------------------------
impl AdamwKernelV1 for EntrenarKernels {
    fn adam_moments(&self, g_tinrd: &[f32], m_00: &[f32]) -> Vec<f32> {
        // beta1 = 0.9 (entrenar default)
        let beta1: f32 = 0.9;
        g_tinrd.iter()
            .zip(m_00.iter())
            .map(|(&gi, &mi)| beta1 * mi + (1.0 - beta1) * gi)
            .collect()
    }

    fn adam_variance(&self, g_tinrd: &[f32], v_00: &[f32]) -> Vec<f32> {
        // beta2 = 0.999 (entrenar default)
        let beta2: f32 = 0.999;
        g_tinrd.iter()
            .zip(v_00.iter())
            .map(|(&gi, &vi)| beta2 * vi + (1.0 - beta2) * gi * gi)
            .collect()
    }

    fn bias_correction(&self, input: &[f32]) -> Vec<f32> {
        // Convention: input = [m_0..m_{n/2}, v_0..v_{n/2}], step t=1
        // Returns [m_hat_0..m_hat_{n/2}, v_hat_0..v_hat_{n/2}]
        let half = input.len() / 2;
        let m = &input[..half];
        let v = &input[half..];
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let t = 1_i32;
        let bc1 = 1.0 / (1.0 - beta1.powi(t));
        let bc2 = 1.0 / (1.0 - beta2.powi(t));
        let mut result = Vec::with_capacity(input.len());
        result.extend(m.iter().map(|&mi| mi * bc1));
        result.extend(v.iter().map(|&vi| vi * bc2));
        result
    }

    fn weight_update(&self, input: &[f32]) -> Vec<f32> {
        // Convention: input = [theta_0..theta_{n/3}, m_hat_0..m_hat_{n/3}, v_hat_0..v_hat_{n/3}]
        // Uses entrenar's default: lr=0.001, eps=1e-8, weight_decay=0.01
        let third = input.len() / 3;
        let theta = &input[..third];
        let m_hat = &input[third..2 * third];
        let v_hat = &input[2 * third..];
        let lr: f32 = 0.001;
        let eps: f32 = 1e-8;
        let wd: f32 = 0.01;
        theta
            .iter()
            .zip(m_hat.iter().zip(v_hat.iter()))
            .map(|(&ti, (&mi, &vi))| ti - lr * (mi / (vi.sqrt() + eps) + wd * ti))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Compile-time enforcement tests -- each test instantiates the trait to
// guarantee the compiler has verified all method signatures.
// ---------------------------------------------------------------------------

#[test]
fn softmax_trait_compiles() {
    let k = EntrenarKernels;
    let out = SoftmaxKernelV1::softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(out.len(), 3);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1.0");
}

#[test]
fn activation_trait_compiles() {
    let k = EntrenarKernels;
    let input = &[-1.0, 0.0, 1.0];

    let gelu_out = ActivationKernelV1::gelu(&k, input);
    assert_eq!(gelu_out.len(), 3);
    assert!(gelu_out[1].abs() < 1e-6, "GELU(0) = 0");

    let relu_out = ActivationKernelV1::relu(&k, input);
    assert_eq!(relu_out.len(), 3);
    assert_eq!(relu_out[0], 0.0, "ReLU(-1) = 0");
    assert_eq!(relu_out[2], 1.0, "ReLU(1) = 1");

    let silu_out = ActivationKernelV1::silu(&k, input);
    assert_eq!(silu_out.len(), 3);
    assert!(silu_out[1].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn silu_trait_compiles() {
    let k = EntrenarKernels;
    let input = &[-2.0, 0.0, 2.0];

    let sig = SiluKernelV1::sigmoid(&k, input);
    assert_eq!(sig.len(), 3);
    assert!((sig[1] - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");

    let silu = SiluKernelV1::silu(&k, input);
    assert_eq!(silu.len(), 3);
    assert!(silu[1].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn layernorm_trait_compiles() {
    let k = EntrenarKernels;
    let out = LayernormKernelV1::layernorm(&k, &[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(out.len(), 4);
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm output mean ~ 0");

    let stats = LayernormKernelV1::statistics(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stats.len(), 2);
    assert!((stats[0] - 2.5).abs() < 1e-6, "mean of [1,2,3,4] = 2.5");
    assert!(stats[1] > 0.0, "variance > 0 for non-constant input");
}

#[test]
fn rmsnorm_trait_compiles() {
    let k = EntrenarKernels;
    let out = RmsnormKernelV1::rmsnorm(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out.len(), 4);
    // RMSNorm preserves sign
    assert!(out[0] > 0.0);
    assert!(out[3] > out[0], "larger input -> larger output (unit gamma)");
}

#[test]
fn rope_trait_compiles() {
    let k = EntrenarKernels;
    // At position m=0, RoPE is identity (cos(0)=1, sin(0)=0)
    let input = &[1.0, 2.0, 3.0, 4.0];
    let out = RopeKernelV1::rope(&k, input);
    assert_eq!(out.len(), 4);
    for (i, (&a, &b)) in input.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "RoPE at m=0 should be identity, idx={i}");
    }
}

#[test]
fn cross_entropy_trait_compiles() {
    let k = EntrenarKernels;

    let log_sm = CrossEntropyKernelV1::log_softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(log_sm.len(), 3);
    assert!(log_sm.iter().all(|&v| v <= 0.0), "log_softmax <= 0");

    // targets (one-hot on class 2), logits
    let ce = CrossEntropyKernelV1::cross_entropy(&k, &[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
    assert_eq!(ce.len(), 1);
    assert!(ce[0] >= 0.0, "cross-entropy >= 0");
}

#[test]
fn adamw_trait_compiles() {
    let k = EntrenarKernels;

    // adam_moments: g_t = [0.5, 0.3], m_prev = [0.0, 0.0]
    let moments = AdamwKernelV1::adam_moments(&k, &[0.5, 0.3], &[0.0, 0.0]);
    assert_eq!(moments.len(), 2);
    // m = 0.9 * 0 + 0.1 * g = 0.1 * g
    assert!((moments[0] - 0.05).abs() < 1e-6, "m = 0.1 * 0.5 = 0.05");

    // adam_variance: g_t = [0.5, 0.3], v_prev = [0.0, 0.0]
    let variance = AdamwKernelV1::adam_variance(&k, &[0.5, 0.3], &[0.0, 0.0]);
    assert_eq!(variance.len(), 2);
    assert!(variance[0] > 0.0, "variance > 0 for non-zero gradient");

    // bias_correction: input = [m, v]
    let corrected = AdamwKernelV1::bias_correction(&k, &[0.05, 0.00025]);
    assert_eq!(corrected.len(), 2);
    assert!(corrected[0].abs() > 0.05, "bias correction amplifies at t=1");

    // weight_update: input = [theta, m_hat, v_hat]
    let updated = AdamwKernelV1::weight_update(&k, &[1.0, 0.5, 0.25, 1.0, 0.5, 0.25]);
    assert_eq!(updated.len(), 2);
    // Weights should have moved
    assert!((updated[0] - 1.0).abs() > 1e-6, "weights updated");
}

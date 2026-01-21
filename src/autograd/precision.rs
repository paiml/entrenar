//! Mixed-precision training utilities
//!
//! Provides support for training with reduced precision (fp16/bf16) while
//! maintaining numerical stability through loss scaling and master weights.
//!
//! ## Overview
//!
//! Mixed-precision training uses lower precision (fp16/bf16) for:
//! - Forward pass activations (memory savings)
//! - Gradient computation (compute speedup)
//!
//! While maintaining full precision (fp32) for:
//! - Master weights (numerical stability)
//! - Loss scaling (gradient underflow prevention)
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::autograd::precision::{MixedPrecisionConfig, Precision, GradScaler};
//!
//! let config = MixedPrecisionConfig::bf16();
//! let mut scaler = GradScaler::new(config.initial_scale);
//!
//! // Forward pass in reduced precision
//! let loss = model.forward(&input);
//!
//! // Scale loss before backward
//! let scaled_loss = scaler.scale(loss);
//! backward(&mut scaled_loss, None);
//!
//! // Unscale and update
//! scaler.unscale_grads(&mut params);
//! optimizer.step(&mut params);
//! scaler.update();
//! ```

use std::fmt;

/// Data type precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// 32-bit floating point (default)
    #[default]
    Fp32,
    /// 16-bit floating point (IEEE half precision)
    Fp16,
    /// 16-bit brain floating point (truncated mantissa)
    Bf16,
}

impl Precision {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Fp16 | Precision::Bf16 => 2,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
        }
    }

    /// Whether this is a reduced precision type
    pub fn is_reduced(&self) -> bool {
        matches!(self, Precision::Fp16 | Precision::Bf16)
    }

    /// Memory multiplier compared to fp32
    pub fn memory_multiplier(&self) -> f32 {
        match self {
            Precision::Fp32 => 1.0,
            Precision::Fp16 | Precision::Bf16 => 0.5,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for mixed-precision training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision for activations and gradients
    pub compute_precision: Precision,
    /// Precision for master weights (always fp32 recommended)
    pub weight_precision: Precision,
    /// Initial loss scale factor
    pub initial_scale: f32,
    /// Factor to increase scale by on successful step
    pub scale_growth_factor: f32,
    /// Factor to decrease scale by on overflow
    pub scale_backoff_factor: f32,
    /// Number of successful steps before increasing scale
    pub scale_growth_interval: usize,
    /// Whether to use dynamic loss scaling
    pub dynamic_scaling: bool,
}

impl MixedPrecisionConfig {
    /// Create fp32 config (no mixed precision)
    pub fn fp32() -> Self {
        Self {
            compute_precision: Precision::Fp32,
            weight_precision: Precision::Fp32,
            initial_scale: 1.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            dynamic_scaling: false,
        }
    }

    /// Create fp16 mixed-precision config
    pub fn fp16() -> Self {
        Self {
            compute_precision: Precision::Fp16,
            weight_precision: Precision::Fp32,
            initial_scale: 65536.0, // 2^16
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            dynamic_scaling: true,
        }
    }

    /// Create bf16 mixed-precision config
    pub fn bf16() -> Self {
        Self {
            compute_precision: Precision::Bf16,
            weight_precision: Precision::Fp32,
            initial_scale: 1.0, // bf16 has larger dynamic range, less scaling needed
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            dynamic_scaling: false, // Often not needed for bf16
        }
    }

    /// Check if mixed precision is enabled
    pub fn is_mixed(&self) -> bool {
        self.compute_precision.is_reduced()
    }

    /// Set initial loss scale
    pub fn with_initial_scale(mut self, scale: f32) -> Self {
        self.initial_scale = scale;
        self
    }

    /// Enable/disable dynamic scaling
    pub fn with_dynamic_scaling(mut self, enabled: bool) -> Self {
        self.dynamic_scaling = enabled;
        self
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self::fp32()
    }
}

/// Gradient scaler for mixed-precision training
///
/// Handles loss scaling to prevent gradient underflow in fp16 training.
#[derive(Debug)]
pub struct GradScaler {
    /// Current loss scale
    scale: f32,
    /// Growth factor
    growth_factor: f32,
    /// Backoff factor
    backoff_factor: f32,
    /// Growth interval
    growth_interval: usize,
    /// Steps since last growth
    steps_since_growth: usize,
    /// Whether dynamic scaling is enabled
    dynamic: bool,
    /// Number of overflows encountered
    overflow_count: usize,
    /// Number of successful steps
    successful_steps: usize,
}

impl GradScaler {
    /// Create a new gradient scaler
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_growth: 0,
            dynamic: true,
            overflow_count: 0,
            successful_steps: 0,
        }
    }

    /// Create from config
    pub fn from_config(config: &MixedPrecisionConfig) -> Self {
        Self {
            scale: config.initial_scale,
            growth_factor: config.scale_growth_factor,
            backoff_factor: config.scale_backoff_factor,
            growth_interval: config.scale_growth_interval,
            steps_since_growth: 0,
            dynamic: config.dynamic_scaling,
            overflow_count: 0,
            successful_steps: 0,
        }
    }

    /// Get current scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale a loss value
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale a gradient value
    pub fn unscale_grad(&self, grad: f32) -> f32 {
        grad / self.scale
    }

    /// Unscale gradients in place and check for overflow
    ///
    /// Returns true if gradients are valid (no overflow), false otherwise.
    pub fn unscale_and_check(&self, grads: &mut [f32]) -> bool {
        let inv_scale = 1.0 / self.scale;
        let mut has_overflow = false;

        for grad in grads.iter_mut() {
            *grad *= inv_scale;
            if !grad.is_finite() {
                has_overflow = true;
            }
        }

        !has_overflow
    }

    /// Update the scale after a step
    ///
    /// Call this after each optimizer step. Pass `true` if gradients were valid.
    pub fn update(&mut self, grads_valid: bool) {
        if !self.dynamic {
            return;
        }

        if grads_valid {
            self.successful_steps += 1;
            self.steps_since_growth += 1;

            // Grow scale after interval of successful steps
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        } else {
            // Overflow detected - reduce scale
            self.overflow_count += 1;
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;

            // Ensure scale doesn't go too low
            self.scale = self.scale.max(1.0);
        }
    }

    /// Get overflow count
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// Get successful step count
    pub fn successful_steps(&self) -> usize {
        self.successful_steps
    }

    /// Check if dynamic scaling is enabled
    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    /// Enable/disable dynamic scaling
    pub fn set_dynamic(&mut self, enabled: bool) {
        self.dynamic = enabled;
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(65536.0)
    }
}

/// Convert f32 to bf16 (truncated)
///
/// BF16 uses the same exponent as f32 but only 7 mantissa bits.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // Take upper 16 bits (sign + exponent + 7 mantissa bits)
    (bits >> 16) as u16
}

/// Convert bf16 to f32
pub fn bf16_to_f32(value: u16) -> f32 {
    // Place in upper 16 bits, lower 16 are zeros
    let bits = u32::from(value) << 16;
    f32::from_bits(bits)
}

/// Convert f32 to fp16 (IEEE half precision)
///
/// Note: This is a simplified conversion that may lose precision.
pub fn f32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    // Handle special cases
    if exp == 0xFF {
        // Inf or NaN
        return ((sign << 15) | 0x7C00 | (mantissa >> 13).min(1)) as u16;
    }

    let new_exp = exp - 127 + 15; // Rebias exponent

    if new_exp <= 0 {
        // Underflow to zero
        return (sign << 15) as u16;
    }

    if new_exp >= 31 {
        // Overflow to infinity
        return ((sign << 15) | 0x7C00) as u16;
    }

    // Normal number
    let new_mantissa = mantissa >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | new_mantissa) as u16
}

/// Convert fp16 to f32
pub fn fp16_to_f32(value: u16) -> f32 {
    let sign = u32::from((value >> 15) & 1);
    let exp = u32::from((value >> 10) & 0x1F);
    let mantissa = u32::from(value & 0x3FF);

    if exp == 0x1F {
        // Inf or NaN
        let new_mantissa = if mantissa != 0 { 0x40_0000 } else { 0 };
        return f32::from_bits((sign << 31) | 0x7F80_0000 | new_mantissa);
    }

    if exp == 0 {
        // Zero or denormal
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal - convert to normal
        let mut m = mantissa;
        let mut e = 1i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let new_exp = ((e + 127 - 15) as u32) & 0xFF;
        let new_mantissa = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (new_exp << 23) | new_mantissa);
    }

    // Normal number
    let new_exp = (exp + 127 - 15) & 0xFF;
    let new_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (new_exp << 23) | new_mantissa)
}

/// Estimate memory savings from mixed precision
///
/// # Arguments
///
/// * `num_params` - Number of model parameters
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `precision` - Target precision
///
/// # Returns
///
/// Tuple of (fp32_bytes, mixed_bytes, savings_ratio)
pub fn estimate_memory_savings(
    num_params: usize,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    precision: Precision,
) -> (usize, usize, f32) {
    // FP32 memory: params + activations + gradients
    let param_bytes_fp32 = num_params * 4;
    let activation_bytes_fp32 = batch_size * seq_len * hidden_size * 4;
    let grad_bytes_fp32 = num_params * 4;
    let total_fp32 = param_bytes_fp32 + activation_bytes_fp32 + grad_bytes_fp32;

    // Mixed precision: master weights (fp32) + activations (reduced) + gradients (reduced)
    let param_bytes_mixed = num_params * 4; // Master weights in fp32
    let activation_bytes_mixed = batch_size * seq_len * hidden_size * precision.size_bytes();
    let grad_bytes_mixed = num_params * precision.size_bytes();
    let total_mixed = param_bytes_mixed + activation_bytes_mixed + grad_bytes_mixed;

    let savings = 1.0 - (total_mixed as f32 / total_fp32 as f32);
    (total_fp32, total_mixed, savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(Precision::Fp32.size_bytes(), 4);
        assert_eq!(Precision::Fp16.size_bytes(), 2);
        assert_eq!(Precision::Bf16.size_bytes(), 2);
    }

    #[test]
    fn test_precision_name() {
        assert_eq!(Precision::Fp32.name(), "fp32");
        assert_eq!(Precision::Fp16.name(), "fp16");
        assert_eq!(Precision::Bf16.name(), "bf16");
    }

    #[test]
    fn test_precision_is_reduced() {
        assert!(!Precision::Fp32.is_reduced());
        assert!(Precision::Fp16.is_reduced());
        assert!(Precision::Bf16.is_reduced());
    }

    #[test]
    fn test_precision_memory_multiplier() {
        assert_eq!(Precision::Fp32.memory_multiplier(), 1.0);
        assert_eq!(Precision::Fp16.memory_multiplier(), 0.5);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::Bf16), "bf16");
    }

    #[test]
    fn test_precision_default() {
        assert_eq!(Precision::default(), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();
        assert!(!config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp16);
        assert!(config.dynamic_scaling);
        assert_eq!(config.initial_scale, 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Bf16);
        assert!(!config.dynamic_scaling); // bf16 typically doesn't need scaling
    }

    #[test]
    fn test_mixed_precision_config_builders() {
        let config = MixedPrecisionConfig::fp16()
            .with_initial_scale(1024.0)
            .with_dynamic_scaling(false);
        assert_eq!(config.initial_scale, 1024.0);
        assert!(!config.dynamic_scaling);
    }

    #[test]
    fn test_grad_scaler_new() {
        let scaler = GradScaler::new(65536.0);
        assert_eq!(scaler.scale(), 65536.0);
        assert!(scaler.is_dynamic());
    }

    #[test]
    fn test_grad_scaler_from_config() {
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::from_config(&config);
        assert_eq!(scaler.scale(), config.initial_scale);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.scale_loss(0.001), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_grad() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.unscale_grad(1000.0), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_valid() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, 200.0, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(valid);
        assert_eq!(grads[0], 1.0);
        assert_eq!(grads[1], 2.0);
        assert_eq!(grads[2], 3.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_overflow() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, f32::INFINITY, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(!valid);
    }

    #[test]
    fn test_grad_scaler_update_success() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.growth_interval = 2; // Fast growth for testing

        scaler.update(true);
        scaler.update(true);

        // After growth_interval successful steps, scale should grow
        assert!(scaler.scale() > 1000.0);
        assert_eq!(scaler.successful_steps(), 2);
    }

    #[test]
    fn test_grad_scaler_update_overflow() {
        let mut scaler = GradScaler::new(1000.0);

        scaler.update(false); // Overflow

        assert!(scaler.scale() < 1000.0);
        assert_eq!(scaler.overflow_count(), 1);
    }

    #[test]
    fn test_grad_scaler_scale_floor() {
        let mut scaler = GradScaler::new(1.0);

        scaler.update(false); // Overflow

        // Scale should not go below 1.0
        assert!(scaler.scale() >= 1.0);
    }

    #[test]
    fn test_grad_scaler_dynamic_disabled() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.set_dynamic(false);

        scaler.update(false);

        // Scale should not change when dynamic is disabled
        assert_eq!(scaler.scale(), 1000.0);
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0, -0.001];
        for &val in &values {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            // BF16 has limited precision, so we check approximate equality
            if val.abs() > 1e-6 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "BF16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_f32_to_fp16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0];
        for &val in &values {
            let fp16 = f32_to_fp16(val);
            let back = fp16_to_f32(fp16);
            // FP16 has limited precision
            if val.abs() > 1e-4 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "FP16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_bf16_special_values() {
        // Zero
        let zero = f32_to_bf16(0.0);
        assert_eq!(bf16_to_f32(zero), 0.0);

        // Negative zero
        let neg_zero = f32_to_bf16(-0.0);
        assert_eq!(bf16_to_f32(neg_zero), -0.0);
    }

    #[test]
    fn test_fp16_infinity() {
        let inf = f32_to_fp16(f32::INFINITY);
        let back = fp16_to_f32(inf);
        assert!(back.is_infinite() && back > 0.0);

        let neg_inf = f32_to_fp16(f32::NEG_INFINITY);
        let back_neg = fp16_to_f32(neg_inf);
        assert!(back_neg.is_infinite() && back_neg < 0.0);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Bf16);

        assert!(mixed < fp32);
        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }

    #[test]
    fn test_memory_savings_no_reduction_for_fp32() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Fp32);

        assert_eq!(fp32, mixed);
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::default();
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.is_mixed());
    }
}

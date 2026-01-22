//! Precision conversion functions and memory estimation utilities.

use super::Precision;

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

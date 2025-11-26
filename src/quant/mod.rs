//! Quantization: QAT and PTQ
//!
//! Provides quantization for QLoRA and Quantization-Aware Training:
//! - 4-bit block-wise quantization for QLoRA
//! - Fake quantization with STE for QAT
//! - PTQ calibration (min-max, percentile, moving average)
//! - GGUF-compatible Q4_0/Q8_0 formats
//! - Per-channel vs per-tensor quantization granularity
//! - Quantization error analysis and metrics

mod calibration;
mod error_analysis;
mod fake_quantize;
mod gguf_quant;
mod granularity;
mod quant4bit;

pub use calibration::{
    calibrate_min_max, calibrate_percentile, CalibrationMethod, CalibrationResult, Calibrator,
};
pub use error_analysis::{
    analyze_error, analyze_outlier_impact, compare_bit_widths, error_within_bounds,
    scale_sensitivity, theoretical_max_error, theoretical_sqnr, QuantErrorStats,
};
pub use fake_quantize::{fake_quantize, ste_backward, FakeQuantConfig, FakeQuantize};
pub use gguf_quant::{GGUFQuantType, Q4_0, Q8_0, GGUF_BLOCK_SIZE};
pub use granularity::{
    calibrate_per_channel, calibrate_per_group, calibrate_per_tensor, compare_granularities,
    dequantize_tensor, dequantize_with_params, quantization_mse, quantize_tensor,
    quantize_with_params, QuantGranularity, QuantMode, QuantParams, QuantizedTensor,
};
pub use quant4bit::{dequantize_4bit, quantize_4bit, Quantized4Bit, BLOCK_SIZE};

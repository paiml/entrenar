//! Quantization: QAT and PTQ
//!
//! Provides quantization for QLoRA and Quantization-Aware Training:
//! - 4-bit block-wise quantization for QLoRA
//! - Fake quantization with STE for QAT
//! - PTQ calibration (min-max, percentile, moving average)
//! - GGUF-compatible Q4_0/Q8_0 formats

mod calibration;
mod fake_quantize;
mod gguf_quant;
mod quant4bit;

pub use calibration::{
    calibrate_min_max, calibrate_percentile, CalibrationMethod, CalibrationResult, Calibrator,
};
pub use fake_quantize::{
    fake_quantize, ste_backward, FakeQuantConfig, FakeQuantize,
};
pub use gguf_quant::{GGUFQuantType, Q4_0, Q8_0, GGUF_BLOCK_SIZE};
pub use quant4bit::{dequantize_4bit, quantize_4bit, Quantized4Bit, BLOCK_SIZE};

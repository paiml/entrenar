//! Quantization: QAT and PTQ
//!
//! Provides quantization for QLoRA and Quantization-Aware Training:
//! - 4-bit block-wise quantization for QLoRA
//! - Fake quantization with STE for QAT

mod fake_quantize;
mod quant4bit;

pub use fake_quantize::{
    fake_quantize, ste_backward, FakeQuantConfig, FakeQuantize,
};
pub use quant4bit::{dequantize_4bit, quantize_4bit, Quantized4Bit, BLOCK_SIZE};

//! Quantization: QAT and PTQ
//!
//! Provides 4-bit quantization for QLoRA and future quantization schemes.

mod quant4bit;

pub use quant4bit::{dequantize_4bit, quantize_4bit, Quantized4Bit, BLOCK_SIZE};

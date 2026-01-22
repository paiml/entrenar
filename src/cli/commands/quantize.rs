//! Quantize command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{QuantMethod, QuantizeArgs};
use crate::quant::{quantize_tensor, QuantGranularity, QuantMode, QuantizedTensor};
use safetensors::SafeTensors;
use std::collections::HashMap;

pub fn run_quantize(args: QuantizeArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Quantizing {} to {}-bit", args.model.display(), args.bits),
    );

    log(
        level,
        LogLevel::Verbose,
        &format!("  Method: {:?}", args.method),
    );
    log(
        level,
        LogLevel::Verbose,
        &format!("  Per-channel: {}", args.per_channel),
    );
    log(
        level,
        LogLevel::Verbose,
        &format!("  Output: {}", args.output.display()),
    );

    // Validate bit width
    if args.bits != 4 && args.bits != 8 {
        return Err(format!("Unsupported bit width: {}. Use 4 or 8.", args.bits));
    }

    // Load safetensors model
    let data = std::fs::read(&args.model).map_err(|e| format!("Failed to read model file: {e}"))?;

    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    // Convert CLI args to quant module types
    let mode = match args.method {
        QuantMethod::Symmetric => QuantMode::Symmetric,
        QuantMethod::Asymmetric => QuantMode::Asymmetric,
    };

    let granularity = if args.per_channel {
        QuantGranularity::PerChannel
    } else {
        QuantGranularity::PerTensor
    };

    // Quantize each tensor
    let mut quantized_tensors: HashMap<String, QuantizedTensor> = HashMap::new();
    let mut total_original_bytes = 0usize;
    let mut total_quantized_bytes = 0usize;

    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

        // Only quantize float tensors
        if tensor.dtype() != safetensors::tensor::Dtype::F32 {
            log(
                level,
                LogLevel::Verbose,
                &format!("  Skipping {name} (not F32)"),
            );
            continue;
        }

        let shape: Vec<usize> = tensor.shape().to_vec();
        let num_elements: usize = shape.iter().product();
        total_original_bytes += num_elements * 4; // 4 bytes per f32

        // Convert bytes to f32 values
        let bytes = tensor.data();
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Quantize
        let quantized = quantize_tensor(&values, &shape, granularity, mode, args.bits);
        total_quantized_bytes += quantized.memory_bytes();

        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Quantized {}: {:?} -> {} bytes",
                name,
                shape,
                quantized.memory_bytes()
            ),
        );

        quantized_tensors.insert((*name).to_string(), quantized);
    }

    // Save quantized model as JSON
    // Note: Quantized tensors use custom block formats (Q4_0, Q8_0) that are not
    // directly compatible with SafeTensors. For SafeTensors output, use GGUF export
    // or dequantize first. JSON format preserves the quantization parameters.
    let output_data = serde_json::to_vec_pretty(&quantized_tensors)
        .map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &output_data)
        .map_err(|e| format!("Failed to write output: {e}"))?;

    let compression_ratio = if total_quantized_bytes > 0 {
        total_original_bytes as f64 / total_quantized_bytes as f64
    } else {
        1.0
    };

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Quantization complete: {} tensors, {:.1}x compression",
            quantized_tensors.len(),
            compression_ratio
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Output: {}", args.output.display()),
    );

    Ok(())
}

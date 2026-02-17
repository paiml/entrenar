//! Quantize command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{QuantMethod, QuantizeArgs};
use crate::quant::{quantize_tensor, QuantGranularity, QuantMode, QuantizedTensor};
use safetensors::SafeTensors;
use std::collections::HashMap;

/// Load and deserialize a SafeTensors model from disk.
fn load_safetensors(args: &QuantizeArgs) -> Result<Vec<u8>, String> {
    std::fs::read(&args.model).map_err(|e| format!("Failed to read model file: {e}"))
}

/// Serialize quantized tensors and write to output path.
fn save_quantized(
    quantized_tensors: &HashMap<String, QuantizedTensor>,
    args: &QuantizeArgs,
) -> Result<(), String> {
    // Note: Quantized tensors use custom block formats (Q4_0, Q8_0) that are not
    // directly compatible with SafeTensors. For SafeTensors output, use GGUF export
    // or dequantize first. JSON format preserves the quantization parameters.
    let output_data = serde_json::to_vec_pretty(quantized_tensors)
        .map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &output_data)
        .map_err(|e| format!("Failed to write output: {e}"))?;

    Ok(())
}

/// Validate and convert CLI arguments to quant module types.
fn resolve_quant_params(args: &QuantizeArgs) -> Result<(QuantMode, QuantGranularity), String> {
    if args.bits != 4 && args.bits != 8 {
        return Err(format!("Unsupported bit width: {}. Use 4 or 8.", args.bits));
    }

    let mode = match args.method {
        QuantMethod::Symmetric => QuantMode::Symmetric,
        QuantMethod::Asymmetric => QuantMode::Asymmetric,
    };

    let granularity = if args.per_channel {
        QuantGranularity::PerChannel
    } else {
        QuantGranularity::PerTensor
    };

    Ok((mode, granularity))
}

/// Byte-size tracking for compression ratio computation.
struct ByteAccumulator {
    original: usize,
    quantized: usize,
}

impl ByteAccumulator {
    fn new() -> Self {
        Self {
            original: 0,
            quantized: 0,
        }
    }

    fn compression_ratio(&self) -> f64 {
        if self.quantized > 0 {
            self.original as f64 / self.quantized as f64
        } else {
            1.0
        }
    }
}

/// Quantize a single F32 tensor and return the result with byte accounting.
fn quantize_single_tensor(
    tensor: &safetensors::tensor::TensorView<'_>,
    granularity: QuantGranularity,
    mode: QuantMode,
    bits: u8,
) -> (QuantizedTensor, usize) {
    let shape: Vec<usize> = tensor.shape().to_vec();
    let num_elements: usize = shape.iter().product();
    let original_bytes = num_elements * 4; // 4 bytes per f32

    let bytes = tensor.data();
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let quantized = quantize_tensor(&values, &shape, granularity, mode, bits);
    (quantized, original_bytes)
}

/// Log verbose details about quantization arguments.
fn log_quant_args(args: &QuantizeArgs, level: LogLevel) {
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
}

pub fn run_quantize(args: QuantizeArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Quantizing {} to {}-bit", args.model.display(), args.bits),
    );

    log_quant_args(&args, level);

    let (mode, granularity) = resolve_quant_params(&args)?;

    let data = load_safetensors(&args)?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    let mut quantized_tensors: HashMap<String, QuantizedTensor> = HashMap::new();
    let mut bytes = ByteAccumulator::new();

    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

        if tensor.dtype() != safetensors::tensor::Dtype::F32 {
            log(level, LogLevel::Verbose, &format!("  Skipping {name} (not F32)"));
            continue;
        }

        let (quantized, original_bytes) =
            quantize_single_tensor(&tensor, granularity, mode, args.bits);
        bytes.original += original_bytes;
        bytes.quantized += quantized.memory_bytes();

        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Quantized {}: {:?} -> {} bytes",
                name,
                tensor.shape(),
                quantized.memory_bytes()
            ),
        );

        quantized_tensors.insert((*name).to_string(), quantized);
    }

    save_quantized(&quantized_tensors, &args)?;

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Quantization complete: {} tensors, {:.1}x compression",
            quantized_tensors.len(),
            bytes.compression_ratio()
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Output: {}", args.output.display()),
    );

    Ok(())
}

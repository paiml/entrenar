//! Entrenar CLI
//!
//! Single-command training entry point for the entrenar library.
//!
//! # Usage
//!
//! ```bash
//! # Train from config
//! entrenar train config.yaml
//!
//! # Train with overrides
//! entrenar train config.yaml --epochs 10 --lr 0.001
//!
//! # Validate config
//! entrenar validate config.yaml
//!
//! # Show config info
//! entrenar info config.yaml
//!
//! # Quantize model
//! entrenar quantize model.gguf --output model_q4.gguf
//!
//! # Merge models
//! entrenar merge model1.gguf model2.gguf --output merged.gguf
//! ```

use clap::Parser;
use entrenar::config::{
    apply_overrides, load_config, train_from_yaml, validate_config, Cli, Command, OutputFormat,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Configure output based on verbose/quiet flags
    let log_level = if cli.quiet {
        LogLevel::Quiet
    } else if cli.verbose {
        LogLevel::Verbose
    } else {
        LogLevel::Normal
    };

    let result = match cli.command {
        Command::Train(args) => run_train(args, log_level),
        Command::Validate(args) => run_validate(args, log_level),
        Command::Info(args) => run_info(args, log_level),
        Command::Quantize(args) => run_quantize(args, log_level),
        Command::Merge(args) => run_merge(args, log_level),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum LogLevel {
    Quiet,
    Normal,
    Verbose,
}

fn log(level: LogLevel, required: LogLevel, msg: &str) {
    if level != LogLevel::Quiet && (level == required || required == LogLevel::Normal) {
        println!("{msg}");
    }
}

fn run_train(args: entrenar::config::TrainArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Entrenar: Training from {}", args.config.display()),
    );

    // Load and validate config
    let mut spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    // Apply command-line overrides
    apply_overrides(&mut spec, &args);

    if args.dry_run {
        log(
            level,
            LogLevel::Normal,
            "Dry run - config validated successfully",
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Model: {}", spec.model.path.display()),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Optimizer: {} (lr={})",
                spec.optimizer.name, spec.optimizer.lr
            ),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Epochs: {}", spec.training.epochs),
        );
        log(
            level,
            LogLevel::Verbose,
            &format!("  Batch size: {}", spec.data.batch_size),
        );
        return Ok(());
    }

    // Run training
    train_from_yaml(&args.config).map_err(|e| format!("Training error: {e}"))?;

    log(level, LogLevel::Normal, "Training complete!");
    Ok(())
}

fn run_validate(args: entrenar::config::ValidateArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Validating config: {}", args.config.display()),
    );

    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    validate_config(&spec).map_err(|e| format!("Validation failed: {e}"))?;

    log(level, LogLevel::Normal, "Configuration is valid");

    if args.detailed {
        println!();
        println!("Configuration Summary:");
        println!("  Model path: {}", spec.model.path.display());
        println!("  Target layers: {:?}", spec.model.layers);
        println!();
        println!("  Training data: {}", spec.data.train.display());
        if let Some(val) = &spec.data.val {
            println!("  Validation data: {}", val.display());
        }
        println!("  Batch size: {}", spec.data.batch_size);
        println!();
        println!("  Optimizer: {}", spec.optimizer.name);
        println!("  Learning rate: {}", spec.optimizer.lr);
        if let Some(wd) = spec.optimizer.params.get("weight_decay") {
            println!("  Weight decay: {wd}");
        }
        println!();
        println!("  Epochs: {}", spec.training.epochs);
        if let Some(clip) = spec.training.grad_clip {
            println!("  Gradient clipping: {clip}");
        }
        println!("  Output dir: {}", spec.training.output_dir.display());

        if let Some(lora) = &spec.lora {
            println!();
            println!("  LoRA:");
            println!("    Rank: {}", lora.rank);
            println!("    Alpha: {}", lora.alpha);
            if lora.dropout > 0.0 {
                println!("    Dropout: {}", lora.dropout);
            }
        }

        if let Some(quant) = &spec.quantize {
            println!();
            println!("  Quantization:");
            println!("    Bits: {}", quant.bits);
            println!("    Symmetric: {}", quant.symmetric);
        }

        if let Some(merge) = &spec.merge {
            println!();
            println!("  Merge:");
            println!("    Method: {}", merge.method);
            if let Some(weight) = merge.params.get("weight") {
                println!("    Weight: {weight}");
            }
        }
    }

    Ok(())
}

fn run_info(args: entrenar::config::InfoArgs, level: LogLevel) -> Result<(), String> {
    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    match args.format {
        OutputFormat::Text => {
            log(level, LogLevel::Normal, "Configuration Info:");
            println!();
            println!("Model: {}", spec.model.path.display());
            println!(
                "Optimizer: {} (lr={})",
                spec.optimizer.name, spec.optimizer.lr
            );
            println!("Epochs: {}", spec.training.epochs);
            println!("Batch size: {}", spec.data.batch_size);

            if spec.lora.is_some() {
                println!("LoRA: enabled");
            }
            if spec.quantize.is_some() {
                println!("Quantization: enabled");
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&spec)
                .map_err(|e| format!("JSON serialization error: {e}"))?;
            println!("{json}");
        }
        OutputFormat::Yaml => {
            let yaml = serde_yaml::to_string(&spec)
                .map_err(|e| format!("YAML serialization error: {e}"))?;
            println!("{yaml}");
        }
    }

    Ok(())
}

fn run_quantize(args: entrenar::config::QuantizeArgs, level: LogLevel) -> Result<(), String> {
    use entrenar::config::QuantMethod;
    use entrenar::quant::{quantize_tensor, QuantGranularity, QuantMode, QuantizedTensor};
    use safetensors::SafeTensors;
    use std::collections::HashMap;

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
        return Err(format!(
            "Unsupported bit width: {}. Use 4 or 8.",
            args.bits
        ));
    }

    // Load safetensors model
    let data = std::fs::read(&args.model)
        .map_err(|e| format!("Failed to read model file: {e}"))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to parse safetensors: {e}"))?;

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
        let tensor = tensors.tensor(name).map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

        // Only quantize float tensors
        if tensor.dtype() != safetensors::tensor::Dtype::F32 {
            log(level, LogLevel::Verbose, &format!("  Skipping {name} (not F32)"));
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

    // Save quantized model as JSON (for now - GGUF export would be future work)
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

fn run_merge(args: entrenar::config::MergeArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!(
            "Merging {} models using {:?}",
            args.models.len(),
            args.method
        ),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(
            level,
            LogLevel::Verbose,
            &format!("  Model {}: {}", i + 1, model.display()),
        );
    }
    log(
        level,
        LogLevel::Verbose,
        &format!("  Output: {}", args.output.display()),
    );

    // TODO: Implement actual merging via merge module
    log(
        level,
        LogLevel::Normal,
        "Merge complete (stub - not yet implemented)",
    );
    Ok(())
}

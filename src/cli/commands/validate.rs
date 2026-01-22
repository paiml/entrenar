//! Validate command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{load_config, validate_config, ValidateArgs};

pub fn run_validate(args: ValidateArgs, level: LogLevel) -> Result<(), String> {
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

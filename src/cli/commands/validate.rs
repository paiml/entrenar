//! Validate command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{load_config, validate_config, TrainSpec, ValidateArgs};

/// Format model information as a string
pub fn format_model_info(spec: &TrainSpec) -> String {
    format!(
        "  Model path: {}\n  Target layers: {:?}",
        spec.model.path.display(),
        spec.model.layers
    )
}

/// Format data configuration as a string
pub fn format_data_info(spec: &TrainSpec) -> String {
    let mut lines = vec![format!("  Training data: {}", spec.data.train.display())];
    if let Some(val) = &spec.data.val {
        lines.push(format!("  Validation data: {}", val.display()));
    }
    lines.push(format!("  Batch size: {}", spec.data.batch_size));
    lines.join("\n")
}

/// Format optimizer configuration as a string
pub fn format_optimizer_info(spec: &TrainSpec) -> String {
    let mut lines = vec![
        format!("  Optimizer: {}", spec.optimizer.name),
        format!("  Learning rate: {}", spec.optimizer.lr),
    ];
    if let Some(wd) = spec.optimizer.params.get("weight_decay") {
        lines.push(format!("  Weight decay: {wd}"));
    }
    lines.join("\n")
}

/// Format training configuration as a string
pub fn format_training_info(spec: &TrainSpec) -> String {
    let mut lines = vec![format!("  Epochs: {}", spec.training.epochs)];
    if let Some(clip) = spec.training.grad_clip {
        lines.push(format!("  Gradient clipping: {clip}"));
    }
    lines.push(format!(
        "  Output dir: {}",
        spec.training.output_dir.display()
    ));
    lines.join("\n")
}

/// Format LoRA configuration as a string
pub fn format_lora_info(spec: &TrainSpec) -> Option<String> {
    spec.lora.as_ref().map(|lora| {
        let mut lines = vec![
            "  LoRA:".to_string(),
            format!("    Rank: {}", lora.rank),
            format!("    Alpha: {}", lora.alpha),
        ];
        if lora.dropout > 0.0 {
            lines.push(format!("    Dropout: {}", lora.dropout));
        }
        lines.join("\n")
    })
}

/// Format quantization configuration as a string
pub fn format_quant_info(spec: &TrainSpec) -> Option<String> {
    spec.quantize.as_ref().map(|quant| {
        format!(
            "  Quantization:\n    Bits: {}\n    Symmetric: {}",
            quant.bits, quant.symmetric
        )
    })
}

/// Format merge configuration as a string
pub fn format_merge_info(spec: &TrainSpec) -> Option<String> {
    spec.merge.as_ref().map(|merge| {
        let mut lines = vec![
            "  Merge:".to_string(),
            format!("    Method: {}", merge.method),
        ];
        if let Some(weight) = merge.params.get("weight") {
            lines.push(format!("    Weight: {weight}"));
        }
        lines.join("\n")
    })
}

/// Print detailed configuration summary
pub fn print_detailed_summary(spec: &TrainSpec) {
    println!();
    println!("Configuration Summary:");
    println!("{}", format_model_info(spec));
    println!();
    println!("{}", format_data_info(spec));
    println!();
    println!("{}", format_optimizer_info(spec));
    println!();
    println!("{}", format_training_info(spec));

    if let Some(lora_info) = format_lora_info(spec) {
        println!();
        println!("{lora_info}");
    }

    if let Some(quant_info) = format_quant_info(spec) {
        println!();
        println!("{quant_info}");
    }

    if let Some(merge_info) = format_merge_info(spec) {
        println!();
        println!("{merge_info}");
    }
}

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
        print_detailed_summary(&spec);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DataConfig, LoRASpec, MergeSpec, ModelRef, OptimSpec, QuantSpec, TrainingParams,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_test_spec() -> TrainSpec {
        TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/model/path"),
                layers: vec!["layer1".to_string()],
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/train.parquet"),
                val: Some(PathBuf::from("/val.parquet")),
                batch_size: 32,
                ..Default::default()
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: {
                    let mut p = HashMap::new();
                    p.insert("weight_decay".to_string(), serde_json::json!(0.01));
                    p
                },
            },
            training: TrainingParams {
                epochs: 10,
                grad_clip: Some(1.0),
                output_dir: PathBuf::from("/output"),
                ..Default::default()
            },
            lora: Some(LoRASpec {
                rank: 16,
                alpha: 32.0,
                dropout: 0.1,
                target_modules: vec!["q_proj".to_string()],
            }),
            quantize: Some(QuantSpec {
                bits: 4,
                symmetric: true,
                per_channel: true,
            }),
            merge: Some(MergeSpec {
                method: "slerp".to_string(),
                params: {
                    let mut p = HashMap::new();
                    p.insert("weight".to_string(), serde_json::json!(0.5));
                    p
                },
            }),
        }
    }

    #[test]
    fn test_format_model_info() {
        let spec = make_test_spec();
        let info = format_model_info(&spec);
        assert!(info.contains("/model/path"));
        assert!(info.contains("layer1"));
    }

    #[test]
    fn test_format_data_info() {
        let spec = make_test_spec();
        let info = format_data_info(&spec);
        assert!(info.contains("/train.parquet"));
        assert!(info.contains("/val.parquet"));
        assert!(info.contains("32"));
    }

    #[test]
    fn test_format_data_info_no_val() {
        let mut spec = make_test_spec();
        spec.data.val = None;
        let info = format_data_info(&spec);
        assert!(info.contains("/train.parquet"));
        assert!(!info.contains("Validation"));
    }

    #[test]
    fn test_format_optimizer_info() {
        let spec = make_test_spec();
        let info = format_optimizer_info(&spec);
        assert!(info.contains("adam"));
        assert!(info.contains("0.001"));
        // weight_decay is in params, check it's present in output
        assert!(info.contains("Weight decay"));
    }

    #[test]
    fn test_format_training_info() {
        let spec = make_test_spec();
        let info = format_training_info(&spec);
        assert!(info.contains("10"));
        assert!(info.contains("1"));
        assert!(info.contains("/output"));
    }

    #[test]
    fn test_format_lora_info() {
        let spec = make_test_spec();
        let info = format_lora_info(&spec).unwrap();
        assert!(info.contains("16"));
        assert!(info.contains("32"));
        assert!(info.contains("0.1"));
    }

    #[test]
    fn test_format_lora_info_none() {
        let mut spec = make_test_spec();
        spec.lora = None;
        assert!(format_lora_info(&spec).is_none());
    }

    #[test]
    fn test_format_quant_info() {
        let spec = make_test_spec();
        let info = format_quant_info(&spec).unwrap();
        assert!(info.contains("4"));
        assert!(info.contains("true"));
    }

    #[test]
    fn test_format_quant_info_none() {
        let mut spec = make_test_spec();
        spec.quantize = None;
        assert!(format_quant_info(&spec).is_none());
    }

    #[test]
    fn test_format_merge_info() {
        let spec = make_test_spec();
        let info = format_merge_info(&spec).unwrap();
        assert!(info.contains("slerp"));
        assert!(info.contains("0.5"));
    }

    #[test]
    fn test_format_merge_info_none() {
        let mut spec = make_test_spec();
        spec.merge = None;
        assert!(format_merge_info(&spec).is_none());
    }
}

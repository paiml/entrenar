//! CLI argument parsing and validation
//!
//! This module provides the command-line interface for entrenar training.
//!
//! # Usage
//!
//! ```bash
//! entrenar train config.yaml
//! entrenar train config.yaml --output-dir ./checkpoints
//! entrenar train config.yaml --resume checkpoint.json
//! entrenar validate config.yaml
//! entrenar info config.yaml
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Entrenar: Training & Optimization Library
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(name = "entrenar")]
#[command(author = "PAIML")]
#[command(version)]
#[command(
    about = "Training & Optimization Library with autograd, LoRA, quantization, and model merging"
)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Command,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

/// Available commands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum Command {
    /// Train a model from YAML configuration
    Train(TrainArgs),

    /// Validate a configuration file without training
    Validate(ValidateArgs),

    /// Display information about a configuration
    Info(InfoArgs),

    /// Quantize a model
    Quantize(QuantizeArgs),

    /// Merge multiple models
    Merge(MergeArgs),
}

/// Arguments for the train command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct TrainArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Override output directory
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Resume training from checkpoint
    #[arg(short, long)]
    pub resume: Option<PathBuf>,

    /// Override number of epochs
    #[arg(short, long)]
    pub epochs: Option<usize>,

    /// Override batch size
    #[arg(short, long)]
    pub batch_size: Option<usize>,

    /// Override learning rate
    #[arg(short, long)]
    pub lr: Option<f32>,

    /// Dry run (validate config but don't train)
    #[arg(long)]
    pub dry_run: bool,

    /// Save checkpoint every N steps
    #[arg(long)]
    pub save_every: Option<usize>,

    /// Log metrics every N steps
    #[arg(long)]
    pub log_every: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Arguments for the validate command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ValidateArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Show detailed validation report
    #[arg(short, long)]
    pub detailed: bool,
}

/// Arguments for the info command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InfoArgs {
    /// Path to YAML configuration file
    #[arg(value_name = "CONFIG")]
    pub config: PathBuf,

    /// Output format (text, json, yaml)
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,
}

/// Arguments for the quantize command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct QuantizeArgs {
    /// Path to model file
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output path for quantized model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Quantization bits (4 or 8)
    #[arg(short, long, default_value = "4")]
    pub bits: u8,

    /// Quantization method (symmetric or asymmetric)
    #[arg(short, long, default_value = "symmetric")]
    pub method: QuantMethod,

    /// Use per-channel quantization
    #[arg(long)]
    pub per_channel: bool,

    /// Path to calibration data
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,
}

/// Arguments for the merge command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct MergeArgs {
    /// Paths to models to merge
    #[arg(value_name = "MODELS", num_args = 2..)]
    pub models: Vec<PathBuf>,

    /// Output path for merged model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Merge method (ties, dare, slerp, average)
    #[arg(short, long, default_value = "ties")]
    pub method: MergeMethod,

    /// Interpolation weight (for slerp)
    #[arg(short, long)]
    pub weight: Option<f32>,

    /// Density threshold (for ties/dare)
    #[arg(short, long)]
    pub density: Option<f32>,

    /// Model weights (comma-separated, for weighted average)
    #[arg(long)]
    pub weights: Option<String>,
}

/// Output format for info command
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Yaml,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            _ => Err(format!(
                "Unknown output format: {}. Valid formats: text, json, yaml",
                s
            )),
        }
    }
}

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QuantMethod {
    #[default]
    Symmetric,
    Asymmetric,
}

impl std::str::FromStr for QuantMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "symmetric" | "sym" => Ok(QuantMethod::Symmetric),
            "asymmetric" | "asym" => Ok(QuantMethod::Asymmetric),
            _ => Err(format!(
                "Unknown quantization method: {}. Valid methods: symmetric, asymmetric",
                s
            )),
        }
    }
}

/// Merge method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MergeMethod {
    #[default]
    Ties,
    Dare,
    Slerp,
    Average,
}

impl std::str::FromStr for MergeMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ties" => Ok(MergeMethod::Ties),
            "dare" => Ok(MergeMethod::Dare),
            "slerp" => Ok(MergeMethod::Slerp),
            "average" | "avg" => Ok(MergeMethod::Average),
            _ => Err(format!(
                "Unknown merge method: {}. Valid methods: ties, dare, slerp, average",
                s
            )),
        }
    }
}

/// Parse CLI arguments from a string slice (for testing)
pub fn parse_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)
}

/// Apply command-line overrides to a TrainSpec
pub fn apply_overrides(spec: &mut super::TrainSpec, args: &TrainArgs) {
    if let Some(output_dir) = &args.output_dir {
        spec.training.output_dir = output_dir.clone();
    }
    if let Some(epochs) = args.epochs {
        spec.training.epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        spec.data.batch_size = batch_size;
    }
    if let Some(lr) = args.lr {
        spec.optimizer.lr = lr;
    }
    if let Some(save_every) = args.save_every {
        spec.training.save_interval = save_every;
    }
    // Note: log_every and seed are CLI-only options not persisted in config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_train_command() {
        let cli = parse_args(["entrenar", "train", "config.yaml"]).unwrap();
        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert!(!args.dry_run);
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_with_overrides() {
        let cli = parse_args([
            "entrenar",
            "train",
            "config.yaml",
            "--epochs",
            "10",
            "--batch-size",
            "32",
            "--lr",
            "0.001",
            "--output-dir",
            "./output",
        ])
        .unwrap();

        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.epochs, Some(10));
                assert_eq!(args.batch_size, Some(32));
                assert!((args.lr.unwrap() - 0.001).abs() < 1e-6);
                assert_eq!(args.output_dir, Some(PathBuf::from("./output")));
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_with_resume() {
        let cli = parse_args([
            "entrenar",
            "train",
            "config.yaml",
            "--resume",
            "checkpoint.json",
        ])
        .unwrap();

        match cli.command {
            Command::Train(args) => {
                assert_eq!(args.resume, Some(PathBuf::from("checkpoint.json")));
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_train_dry_run() {
        let cli = parse_args(["entrenar", "train", "config.yaml", "--dry-run"]).unwrap();
        match cli.command {
            Command::Train(args) => {
                assert!(args.dry_run);
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn test_parse_validate_command() {
        let cli = parse_args(["entrenar", "validate", "config.yaml"]).unwrap();
        match cli.command {
            Command::Validate(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert!(!args.detailed);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_parse_validate_detailed() {
        let cli = parse_args(["entrenar", "validate", "config.yaml", "--detailed"]).unwrap();
        match cli.command {
            Command::Validate(args) => {
                assert!(args.detailed);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_parse_info_command() {
        let cli = parse_args(["entrenar", "info", "config.yaml"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.config, PathBuf::from("config.yaml"));
                assert_eq!(args.format, OutputFormat::Text);
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_parse_info_json_format() {
        let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "json"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.format, OutputFormat::Json);
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_parse_quantize_command() {
        let cli = parse_args([
            "entrenar",
            "quantize",
            "model.gguf",
            "--output",
            "model_q4.gguf",
        ])
        .unwrap();

        match cli.command {
            Command::Quantize(args) => {
                assert_eq!(args.model, PathBuf::from("model.gguf"));
                assert_eq!(args.output, PathBuf::from("model_q4.gguf"));
                assert_eq!(args.bits, 4);
                assert_eq!(args.method, QuantMethod::Symmetric);
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_parse_quantize_with_options() {
        let cli = parse_args([
            "entrenar",
            "quantize",
            "model.gguf",
            "--output",
            "model_q8.gguf",
            "--bits",
            "8",
            "--method",
            "asymmetric",
            "--per-channel",
        ])
        .unwrap();

        match cli.command {
            Command::Quantize(args) => {
                assert_eq!(args.bits, 8);
                assert_eq!(args.method, QuantMethod::Asymmetric);
                assert!(args.per_channel);
            }
            _ => panic!("Expected Quantize command"),
        }
    }

    #[test]
    fn test_parse_merge_command() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "--output",
            "merged.gguf",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.models.len(), 2);
                assert_eq!(args.output, PathBuf::from("merged.gguf"));
                assert_eq!(args.method, MergeMethod::Ties);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_parse_merge_slerp() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "--output",
            "merged.gguf",
            "--method",
            "slerp",
            "--weight",
            "0.7",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.method, MergeMethod::Slerp);
                assert!((args.weight.unwrap() - 0.7).abs() < 1e-6);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_parse_merge_multiple_models() {
        let cli = parse_args([
            "entrenar",
            "merge",
            "model1.gguf",
            "model2.gguf",
            "model3.gguf",
            "--output",
            "merged.gguf",
            "--method",
            "average",
        ])
        .unwrap();

        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.models.len(), 3);
                assert_eq!(args.method, MergeMethod::Average);
            }
            _ => panic!("Expected Merge command"),
        }
    }

    #[test]
    fn test_global_verbose_flag() {
        let cli = parse_args(["entrenar", "-v", "train", "config.yaml"]).unwrap();
        assert!(cli.verbose);
        assert!(!cli.quiet);
    }

    #[test]
    fn test_global_quiet_flag() {
        let cli = parse_args(["entrenar", "-q", "train", "config.yaml"]).unwrap();
        assert!(!cli.verbose);
        assert!(cli.quiet);
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_quant_method_from_str() {
        assert_eq!(
            "symmetric".parse::<QuantMethod>().unwrap(),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "sym".parse::<QuantMethod>().unwrap(),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "asymmetric".parse::<QuantMethod>().unwrap(),
            QuantMethod::Asymmetric
        );
        assert_eq!(
            "asym".parse::<QuantMethod>().unwrap(),
            QuantMethod::Asymmetric
        );
        assert!("invalid".parse::<QuantMethod>().is_err());
    }

    #[test]
    fn test_merge_method_from_str() {
        assert_eq!("ties".parse::<MergeMethod>().unwrap(), MergeMethod::Ties);
        assert_eq!("dare".parse::<MergeMethod>().unwrap(), MergeMethod::Dare);
        assert_eq!("slerp".parse::<MergeMethod>().unwrap(), MergeMethod::Slerp);
        assert_eq!(
            "average".parse::<MergeMethod>().unwrap(),
            MergeMethod::Average
        );
        assert_eq!("avg".parse::<MergeMethod>().unwrap(), MergeMethod::Average);
        assert!("invalid".parse::<MergeMethod>().is_err());
    }

    #[test]
    fn test_missing_config_file() {
        let result = parse_args(["entrenar", "train"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_command() {
        let result = parse_args(["entrenar", "unknown"]);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for valid config paths
    fn config_path_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,20}\\.(yaml|yml)"
    }

    // Strategy for valid output paths
    fn output_path_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_/-]{0,30}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_train_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "train", &config]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.config.to_str().unwrap(), &config);
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_validate_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "validate", &config]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Validate(args) => {
                    prop_assert_eq!(args.config.to_str().unwrap(), &config);
                }
                _ => prop_assert!(false, "Expected Validate command"),
            }
        }

        #[test]
        fn prop_info_command_parses(config in config_path_strategy()) {
            let result = parse_args(["entrenar", "info", &config]);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_epochs_override_positive(
            config in config_path_strategy(),
            epochs in 1usize..10000
        ) {
            let epochs_str = epochs.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--epochs", &epochs_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.epochs, Some(epochs));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_batch_size_override_positive(
            config in config_path_strategy(),
            batch_size in 1usize..1024
        ) {
            let batch_str = batch_size.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--batch-size", &batch_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.batch_size, Some(batch_size));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_learning_rate_override(
            config in config_path_strategy(),
            lr in 1e-10f32..1.0
        ) {
            let lr_str = format!("{:.10}", lr);
            let result = parse_args([
                "entrenar", "train", &config,
                "--lr", &lr_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    let parsed_lr = args.lr.unwrap();
                    // Allow for float parsing precision
                    prop_assert!((parsed_lr - lr).abs() < 1e-6 || (parsed_lr / lr - 1.0).abs() < 1e-4);
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_seed_override(
            config in config_path_strategy(),
            seed in 0u64..u64::MAX
        ) {
            let seed_str = seed.to_string();
            let result = parse_args([
                "entrenar", "train", &config,
                "--seed", &seed_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Train(args) => {
                    prop_assert_eq!(args.seed, Some(seed));
                }
                _ => prop_assert!(false, "Expected Train command"),
            }
        }

        #[test]
        fn prop_quantize_bits_valid(
            model in output_path_strategy(),
            bits in prop::sample::select(vec![4u8, 8])
        ) {
            let bits_str = bits.to_string();
            let result = parse_args([
                "entrenar", "quantize", &model,
                "--output", "out.gguf",
                "--bits", &bits_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Quantize(args) => {
                    prop_assert_eq!(args.bits, bits);
                }
                _ => prop_assert!(false, "Expected Quantize command"),
            }
        }

        #[test]
        fn prop_merge_weight_valid(
            weight in 0.0f32..=1.0
        ) {
            let weight_str = format!("{:.4}", weight);
            let result = parse_args([
                "entrenar", "merge",
                "model1.gguf", "model2.gguf",
                "--output", "merged.gguf",
                "--method", "slerp",
                "--weight", &weight_str,
            ]);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Merge(args) => {
                    let parsed_weight = args.weight.unwrap();
                    prop_assert!((parsed_weight - weight).abs() < 1e-3);
                }
                _ => prop_assert!(false, "Expected Merge command"),
            }
        }

        #[test]
        fn prop_output_format_case_insensitive(
            format in prop::sample::select(vec!["text", "TEXT", "Text", "json", "JSON", "Json", "yaml", "YAML", "Yaml"])
        ) {
            let result = format.parse::<OutputFormat>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_merge_method_case_insensitive(
            method in prop::sample::select(vec!["ties", "TIES", "dare", "DARE", "slerp", "SLERP", "average", "avg"])
        ) {
            let result = method.parse::<MergeMethod>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_quant_method_case_insensitive(
            method in prop::sample::select(vec!["symmetric", "SYMMETRIC", "sym", "SYM", "asymmetric", "asym"])
        ) {
            let result = method.parse::<QuantMethod>();
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_verbose_quiet_exclusive(config in config_path_strategy()) {
            // Can't have both verbose and quiet
            let cli_v = parse_args(["entrenar", "-v", "train", &config]).unwrap();
            let cli_q = parse_args(["entrenar", "-q", "train", &config]).unwrap();

            prop_assert!(cli_v.verbose && !cli_v.quiet);
            prop_assert!(!cli_q.verbose && cli_q.quiet);
        }

        #[test]
        fn prop_multiple_models_merge(
            model_count in 2usize..=5
        ) {
            let mut args: Vec<String> = vec!["entrenar".to_string(), "merge".to_string()];
            let models: Vec<String> = (0..model_count).map(|i| format!("model{}.gguf", i)).collect();
            for m in &models {
                args.push(m.clone());
            }
            args.push("--output".to_string());
            args.push("merged.gguf".to_string());

            let result = parse_args(&args);
            prop_assert!(result.is_ok());
            let cli = result.unwrap();
            match cli.command {
                Command::Merge(merge_args) => {
                    prop_assert_eq!(merge_args.models.len(), model_count);
                }
                _ => prop_assert!(false, "Expected Merge command"),
            }
        }
    }
}

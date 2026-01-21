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

mod core;
mod extended;
mod init;
mod quant_merge;
mod research;
mod types;

// Re-export all public types
pub use core::{apply_overrides, parse_args, Cli, Command, InfoArgs, TrainArgs, ValidateArgs};
pub use extended::{AuditArgs, BenchArgs, CompletionArgs, InspectArgs, MonitorArgs};
pub use init::{InitArgs, InitTemplate};
pub use quant_merge::{MergeArgs, MergeMethod, QuantMethod, QuantizeArgs};
pub use research::{
    BundleArgs, CiteArgs, DepositArgs, ExportArgs, PreregisterArgs, ResearchArgs, ResearchCommand,
    ResearchInitArgs, VerifyArgs,
};
pub use types::{
    ArchiveProviderArg, ArtifactTypeArg, AuditType, CitationFormat, ExportFormat, InspectMode,
    LicenseArg, OutputFormat, ShellType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
    fn test_missing_config_file() {
        let result = parse_args(["entrenar", "train"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_command() {
        let result = parse_args(["entrenar", "unknown"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_overrides_output_dir() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: Some(PathBuf::from("./custom_output")),
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("./custom_output"));
    }

    #[test]
    fn test_apply_overrides_epochs() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: Some(50),
            batch_size: None,
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.epochs, 50);
    }

    #[test]
    fn test_apply_overrides_batch_size() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: Some(64),
            lr: None,
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.data.batch_size, 64);
    }

    #[test]
    fn test_apply_overrides_lr() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: Some(0.0001),
            save_every: None,
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert!((spec.optimizer.lr - 0.0001).abs() < 1e-8);
    }

    #[test]
    fn test_apply_overrides_save_every() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            save_every: Some(5),
            log_every: None,
            dry_run: false,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.save_interval, 5);
    }

    #[test]
    fn test_apply_overrides_all() {
        let mut spec = create_test_spec();
        let args = TrainArgs {
            config: PathBuf::from("config.yaml"),
            output_dir: Some(PathBuf::from("./all_overrides")),
            resume: Some(PathBuf::from("checkpoint.json")),
            epochs: Some(100),
            batch_size: Some(128),
            lr: Some(0.01),
            save_every: Some(10),
            log_every: Some(50),
            dry_run: true,
            seed: Some(42),
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("./all_overrides"));
        assert_eq!(spec.training.epochs, 100);
        assert_eq!(spec.data.batch_size, 128);
        assert!((spec.optimizer.lr - 0.01).abs() < 1e-8);
        assert_eq!(spec.training.save_interval, 10);
    }

    fn create_test_spec() -> crate::config::TrainSpec {
        crate::config::TrainSpec {
            model: crate::config::ModelRef {
                path: PathBuf::from("model.gguf"),
                layers: vec![],
            },
            data: crate::config::DataConfig {
                train: PathBuf::from("train.parquet"),
                val: None,
                batch_size: 8,
                auto_infer_types: true,
                seq_len: None,
            },
            optimizer: crate::config::OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: Default::default(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: Default::default(),
        }
    }

    #[test]
    fn test_verbose_and_quiet_flags() {
        let cli = parse_args(["entrenar", "--verbose", "train", "config.yaml"]).unwrap();
        assert!(cli.verbose);
        assert!(!cli.quiet);

        let cli = parse_args(["entrenar", "--quiet", "train", "config.yaml"]).unwrap();
        assert!(!cli.verbose);
        assert!(cli.quiet);
    }

    #[test]
    fn test_info_yaml_format() {
        let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "yaml"]).unwrap();
        match cli.command {
            Command::Info(args) => {
                assert_eq!(args.format, OutputFormat::Yaml);
            }
            _ => panic!("Expected Info command"),
        }
    }

    #[test]
    fn test_parse_init_command() {
        let cli = parse_args(["entrenar", "init"]).unwrap();
        match cli.command {
            Command::Init(args) => {
                assert_eq!(args.template, InitTemplate::Minimal);
                assert_eq!(args.name, "my-experiment");
            }
            _ => panic!("Expected Init command"),
        }
    }

    #[test]
    fn test_parse_init_with_template() {
        let cli = parse_args([
            "entrenar",
            "init",
            "--template",
            "lora",
            "--name",
            "test-exp",
        ])
        .unwrap();
        match cli.command {
            Command::Init(args) => {
                assert_eq!(args.template, InitTemplate::Lora);
                assert_eq!(args.name, "test-exp");
            }
            _ => panic!("Expected Init command"),
        }
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
            let lr_str = format!("{lr:.10}");
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
            let weight_str = format!("{weight:.4}");
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
            let models: Vec<String> = (0..model_count).map(|i| format!("model{i}.gguf")).collect();
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

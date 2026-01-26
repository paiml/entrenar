//! Train command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{apply_overrides, load_config, train_from_yaml, TrainArgs};

pub fn run_train(args: TrainArgs, level: LogLevel) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_args(config_path: &str, dry_run: bool) -> TrainArgs {
        TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run,
            save_every: None,
            log_every: None,
            seed: None,
        }
    }

    #[test]
    fn test_train_dry_run_valid_config() {
        // Create a minimal valid config file
        let config_content = r#"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
"#;
        let config_path = "/tmp/test_train_config.yaml";
        std::fs::write(config_path, config_content).unwrap();

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Quiet);
        assert!(result.is_ok(), "Dry run should succeed: {:?}", result);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_train_invalid_config_path() {
        let args = make_args("/nonexistent/config.yaml", false);
        let result = run_train(args, LogLevel::Quiet);
        assert!(result.is_err(), "Should fail with invalid config path");
    }

    #[test]
    fn test_train_dry_run_logs_correctly() {
        let config_content = r#"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 16
optimizer:
  name: sgd
  lr: 0.01
training:
  epochs: 5
"#;
        let config_path = "/tmp/test_train_config_logs.yaml";
        std::fs::write(config_path, config_content).unwrap();

        let args = make_args(config_path, true);
        // Test with verbose logging to cover log branches
        let result = run_train(args, LogLevel::Verbose);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }
}

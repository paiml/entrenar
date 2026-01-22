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

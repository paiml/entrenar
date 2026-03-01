//! Finetune command implementation (plan/apply classification training)

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{FinetuneArgs, FinetuneCommand};

pub fn run_finetune(args: FinetuneArgs, level: LogLevel) -> Result<(), String> {
    match args.command {
        FinetuneCommand::Plan {
            data,
            model_path,
            model_size,
            num_classes,
            output_dir,
            strategy,
            budget,
            scout,
            max_epochs,
            lr,
            lora_rank,
            batch_size,
            lora_alpha,
            warmup,
            gradient_clip,
            lr_min_ratio,
            class_weights,
            target_modules,
        } => {
            run_plan(
                data,
                model_path,
                model_size,
                num_classes,
                output_dir,
                strategy,
                budget,
                scout,
                max_epochs,
                lr,
                lora_rank,
                batch_size,
                lora_alpha,
                warmup,
                gradient_clip,
                lr_min_ratio,
                class_weights,
                target_modules,
                level,
            )
        }
        FinetuneCommand::Apply { plan, model_path, data, output_dir } => {
            run_apply(plan, model_path, data, output_dir, level)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_plan(
    data: std::path::PathBuf,
    model_path: Option<std::path::PathBuf>,
    model_size: String,
    num_classes: usize,
    output_dir: std::path::PathBuf,
    strategy: String,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    manual_lora_alpha: Option<f32>,
    manual_warmup: Option<f32>,
    manual_gradient_clip: Option<f32>,
    manual_lr_min_ratio: Option<f32>,
    manual_class_weights: Option<String>,
    manual_target_modules: Option<String>,
    level: LogLevel,
) -> Result<(), String> {
    use crate::finetune::training_plan::{plan, PlanConfig};

    log(level, LogLevel::Normal, "Generating training plan...");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: data,
        val_path: None,
        test_path: None,
        model_size,
        model_path,
        num_classes,
        output_dir: output_dir.clone(),
        strategy,
        budget,
        scout,
        max_epochs,
        manual_lr,
        manual_lora_rank,
        manual_batch_size,
        manual_lora_alpha,
        manual_warmup,
        manual_gradient_clip,
        manual_lr_min_ratio,
        manual_class_weights,
        manual_target_modules,
    };

    let training_plan = plan(&config).map_err(|e| format!("Plan generation failed: {e}"))?;

    // Print plan summary
    print_plan_summary(&training_plan, level);

    // Save plan to output dir
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;
    let plan_path = output_dir.join("plan.yaml");
    std::fs::write(&plan_path, training_plan.to_yaml())
        .map_err(|e| format!("Failed to write plan: {e}"))?;

    log(level, LogLevel::Normal, &format!("Plan saved to: {}", plan_path.display()));
    log(
        level,
        LogLevel::Normal,
        &format!(
            "\nTo execute: apr finetune apply --plan {} --model-path <MODEL_DIR> --data <DATA.jsonl> -o {}",
            plan_path.display(),
            output_dir.display()
        ),
    );

    Ok(())
}

fn run_apply(
    plan_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
    data_path: std::path::PathBuf,
    output_dir: std::path::PathBuf,
    level: LogLevel,
) -> Result<(), String> {
    use crate::finetune::training_plan::{execute_plan, ApplyConfig, TrainingPlan};

    log(level, LogLevel::Normal, &format!("Loading plan from: {}", plan_path.display()));

    let plan_str =
        std::fs::read_to_string(&plan_path).map_err(|e| format!("Failed to read plan: {e}"))?;
    let plan =
        TrainingPlan::from_str(&plan_str).map_err(|e| format!("Failed to parse plan: {e}"))?;

    print_plan_summary(&plan, level);

    log(level, LogLevel::Normal, &format!("Model: {}", model_path.display()));
    log(level, LogLevel::Normal, &format!("Data:  {}", data_path.display()));
    log(level, LogLevel::Normal, &format!("Output: {}", output_dir.display()));
    log(level, LogLevel::Normal, "");
    log(level, LogLevel::Normal, "Starting training...");

    let apply = ApplyConfig {
        model_path,
        data_path,
        output_dir,
        on_trial_complete: Some(|trial_id, total, summary| {
            eprintln!(
                "  [{}/{}] val_loss={:.4} val_acc={:.1}% [{}]",
                trial_id + 1,
                total,
                summary.val_loss,
                summary.val_accuracy * 100.0,
                summary.status,
            );
        }),
    };

    let result = execute_plan(&plan, &apply).map_err(|e| format!("Training failed: {e}"))?;

    // Print results
    log(level, LogLevel::Normal, "");
    log(level, LogLevel::Normal, "Training complete!");
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Strategy: {} | Trials: {} | Time: {:.1}s",
            result.strategy,
            result.trials.len(),
            result.total_time_ms as f64 / 1000.0,
        ),
    );

    if let Some(best) = result.trials.get(result.best_trial_id) {
        log(level, LogLevel::Normal, &format!("  Best trial #{}", result.best_trial_id + 1));
        log(
            level,
            LogLevel::Normal,
            &format!(
                "    val_loss={:.4} val_acc={:.1}% epochs={}",
                best.val_loss,
                best.val_accuracy * 100.0,
                best.epochs_run,
            ),
        );
    }

    Ok(())
}

fn print_plan_summary(
    plan: &crate::finetune::training_plan::TrainingPlan,
    level: LogLevel,
) {
    let (pass, warn, fail) = plan.check_counts();

    log(level, LogLevel::Normal, &format!("Plan: {} v{}", plan.task, plan.version));
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Data: {} samples, {} classes",
            plan.data.train_samples,
            plan.data.class_counts.len(),
        ),
    );
    if plan.data.imbalance_ratio > 2.0 {
        log(
            level,
            LogLevel::Normal,
            &format!(
                "  Imbalance: {:.1}x (auto class weights: {})",
                plan.data.imbalance_ratio, plan.data.auto_class_weights,
            ),
        );
    }
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Model: {} ({}, {} layers, hidden={})",
            plan.model.architecture, plan.model.size, plan.model.num_layers, plan.model.hidden_size,
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  HPO: {} (budget={}, scout={}, max_epochs={})",
            plan.hyperparameters.strategy,
            plan.hyperparameters.budget,
            plan.hyperparameters.scout,
            plan.hyperparameters.max_epochs,
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Resources: {:.1} GB VRAM, {:.0} min/epoch, {:.0} min total",
            plan.resources.estimated_vram_gb,
            plan.resources.estimated_minutes_per_epoch,
            plan.resources.estimated_total_minutes,
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Pre-flight: {pass} pass, {warn} warn, {fail} fail"),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Verdict: {:?}", plan.verdict),
    );
}

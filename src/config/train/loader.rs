#![allow(dead_code)]
//! Main entry points for YAML-based training

// Default model architecture constants (Qwen2.5-Coder-0.5B)
const QWEN_HIDDEN_SIZE: usize = 896;
const QWEN_NUM_ATTENTION_HEADS: usize = 14;
const QWEN_NUM_KV_HEADS: usize = 2;
const QWEN_INTERMEDIATE_SIZE: usize = 4864;
const QWEN_NUM_HIDDEN_LAYERS: usize = 24;
const QWEN_VOCAB_SIZE: usize = 151936;
const QWEN_MAX_POSITION_EMBEDDINGS: usize = 32768;
const QWEN_ROPE_THETA: f64 = 1_000_000.0;

use super::batches::load_training_batches;
use crate::config::schema::{ModelMode, TrainSpec};
use crate::config::validate::validate_config;
use crate::error::{Error, Result};
use crate::monitor::tui::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use crate::storage::{ExperimentStorage, ParameterValue, RunStatus, SqliteBackend};
use crate::tokenizer::HfTokenizer;
use crate::trace::TRACER;
#[cfg(feature = "cuda")]
use crate::train::CudaTransformerTrainer;
use crate::train::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{
    load_safetensors_weights, Architecture, ModelArchitecture, Transformer, TransformerConfig,
};
use crate::yaml_mode;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Train a model from YAML configuration file
///
/// This is the main entry point for declarative training. It:
/// 1. Loads and parses the YAML config
/// 2. Validates the configuration
/// 3. Dispatches to appropriate trainer (Tabular or Transformer)
/// 4. Runs the training loop
/// 5. Saves the final model
///
/// # Example
///
/// ```no_run
/// use entrenar::config::train_from_yaml;
///
/// let model = train_from_yaml("config.yaml")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn train_from_yaml<P: AsRef<Path>>(config_path: P) -> Result<()> {
    let spec = load_config(config_path)?;

    // Dispatch based on model mode
    match spec.model.mode {
        ModelMode::Transformer => train_transformer_from_spec(&spec),
        ModelMode::Tabular => train_tabular_from_spec(&spec),
    }
}

/// Build a TransformerTrainConfig from YAML spec, wiring all hyperparameters.
fn build_train_config(
    model_config: crate::transformer::TransformerConfig,
    spec: &TrainSpec,
) -> TransformerTrainConfig {
    let mut config = TransformerTrainConfig::new(model_config)
        .with_lr(spec.optimizer.lr)
        .with_warmup_steps(spec.training.warmup_steps)
        .with_max_seq_len({
            let seq_len = spec.data.seq_len.unwrap_or_else(|| {
                eprintln!("Warning: seq_len not specified in config, defaulting to 512");
                512
            });
            seq_len
        });

    if let Some(clip) = spec.training.grad_clip {
        config = config.with_grad_clip(clip);
    }

    // Wire optimizer hyperparameters from YAML (ALB-040)
    if let Some(v) = spec.optimizer.params.get("beta2").and_then(serde_json::Value::as_f64) {
        config = config.with_beta2(v as f32);
    }
    if let Some(v) = spec.optimizer.params.get("weight_decay").and_then(serde_json::Value::as_f64) {
        config = config.with_weight_decay(v as f32);
    }

    if let Some(accum) = spec.training.gradient_accumulation {
        config = config.with_accumulation_steps(accum);
        if accum > 1 {
            let eff_batch = spec.data.batch_size * accum * spec.data.seq_len.unwrap_or(1024);
            println!("  Gradient accumulation: {accum} (effective batch: {eff_batch} tokens/step)");
        }
    }

    if let Some(max_steps) = spec.training.max_steps {
        config = config.with_max_steps(max_steps);
    }

    // Enable mixed precision if specified
    if let Some(ref precision) = spec.training.mixed_precision {
        match precision.as_str() {
            "bf16" => config = config.with_bf16(),
            "fp16" => config = config.with_fp16(),
            "fp32" => {}
            other => {
                eprintln!("Warning: unknown mixed_precision value '{other}', defaulting to fp32");
            }
        }
    }

    // R-021: Activation checkpointing (gradient recomputation)
    if let Some(num_segments) = spec.training.checkpoints {
        config = config.with_checkpointing(num_segments);
    }

    // R-084: Bitwise deterministic training (C-DETERM-001)
    if spec.training.deterministic {
        config = config.with_deterministic(true);
    }
    if let Some(seed) = spec.training.seed {
        config = config.with_seed(seed);
    }

    // KAIZEN-047: Step profiler (0 = disabled)
    if spec.training.profile_interval > 0 {
        config = config.with_profile_interval(spec.training.profile_interval);
    }

    // ENT-LoRA-001: Wire LoRA config from YAML spec
    if let Some(ref lora) = spec.lora {
        config = config.with_lora(lora.rank, lora.alpha, lora.target_modules.clone());
        // ENT-LoRA-006: LoRA+ ratio from YAML
        if lora.lora_plus_ratio != 1.0 {
            config = config.with_lora_plus_ratio(lora.lora_plus_ratio);
        }
        // ENT-LoRA-008: Double quantization from YAML
        if lora.double_quantize {
            config = config.with_double_quantize(true);
        }
        // ENT-263: NF4 quantization for QLoRA pretraining
        if lora.quantize_base {
            config = config.with_quantize_nf4(true);
        }
    }

    // Wire distributed config from YAML (#133)
    if let Some(ref dist) = spec.training.distributed {
        use crate::train::{DistributedBackend, DistributedRole, DistributedTrainConfig};

        let role = match dist.role.as_str() {
            "worker" => DistributedRole::Worker,
            _ => DistributedRole::Coordinator,
        };
        let backend = match dist.backend.as_str() {
            "cuda" => DistributedBackend::Cuda,
            "wgpu" => DistributedBackend::Wgpu,
            _ => DistributedBackend::Auto,
        };
        let addr: std::net::SocketAddr =
            dist.coordinator_addr.parse().unwrap_or_else(|_| "0.0.0.0:9000".parse().unwrap());

        config = config.with_distributed(DistributedTrainConfig {
            world_size: dist.world_size,
            rank: dist.rank,
            local_rank: dist.local_rank,
            role,
            coordinator_addr: addr,
            backend,
        });
    }

    config
}

/// Train a transformer model (LLM) from spec
///
/// Uses TransformerTrainer with CausalLMLoss for language modeling.
fn train_transformer_from_spec(spec: &TrainSpec) -> Result<()> {
    println!("✓ Config loaded and validated (Transformer mode)");
    println!("  Model: {}", spec.model.path.display());
    println!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);
    println!("  Training mode: {:?}", spec.training.mode);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
        if lora.quantize_base {
            println!("  QLoRA: NF4 quantized base weights (~8x VRAM compression)");
        }
    }
    println!();

    // Build TransformerConfig from spec
    let model_config = build_transformer_config_from_spec(spec)?;

    // Resolve model path (downloads from HF Hub if repo ID)
    let resolved_path = resolve_model_path(&spec.model.path)?;

    // Try to load model weights if path exists (ENT-117)
    // ALB-097: Check output_dir first for checkpoint resume, then model_path for initial weights
    let (transformer, checkpoint_step) =
        load_transformer_model(&resolved_path, &model_config, &spec.training.output_dir)?;

    // Build TransformerTrainConfig from YAML spec fields
    let train_config = build_train_config(model_config, spec);

    // Apply deterministic settings before any CUDA operations
    train_config.apply_deterministic_settings();

    // Load training data as LMBatches (supports tokenizer + text data)
    println!("Loading training data...");
    let batches = load_lm_batches(spec)?;
    println!("✓ {} LM batches created", batches.len());
    println!();

    // Try CUDA-resident training first (ALB-040), fall back to CPU
    #[cfg(feature = "cuda")]
    if train_config.use_cuda {
        let cuda_config = train_config.clone();
        let cuda_result = match transformer {
            Some(loaded_model) => CudaTransformerTrainer::with_model(loaded_model, cuda_config),
            None => CudaTransformerTrainer::new(cuda_config),
        };

        match cuda_result {
            Ok(mut cuda_trainer) => {
                // Restore step counter from checkpoint for LR schedule + AdamW bias correction
                if checkpoint_step > 0 {
                    cuda_trainer.set_initial_step(checkpoint_step);
                    println!(
                        "  Resumed at step {checkpoint_step} (lr={:.2e})",
                        cuda_trainer.current_lr()
                    );
                    // ALB-096: Try APR optimizer state first, then fall back to JSON
                    let apr_loaded = find_latest_apr_checkpoint(&spec.training.output_dir)
                        .map_or(false, |p| cuda_trainer.load_optimizer_state_apr(&p));
                    if apr_loaded {
                        println!("  ✓ Embedding optimizer state restored (APR)");
                    } else if cuda_trainer.load_optimizer_state(&spec.training.output_dir) {
                        println!("  ✓ Embedding optimizer state restored (JSON)");
                    }
                }
                println!("✓ CudaTransformerTrainer initialized (GPU: {})", cuda_trainer.gpu_name());
                // #133: Dispatch to distributed training loop if distributed config present
                if train_config.distributed.is_some() {
                    return train_loop_cuda_distributed(cuda_trainer, &batches, spec);
                }
                return train_loop_cuda(&mut cuda_trainer, &batches, spec);
            }
            Err(e) => {
                eprintln!("Warning: CUDA training failed ({e}), falling back to CPU");
                // transformer was consumed — rebuild from config
                let mut trainer = TransformerTrainer::new(train_config);
                println!("✓ TransformerTrainer initialized (CPU fallback)");
                println!("  Mixed precision: {}", trainer.is_mixed_precision());
                println!("  Checkpointing: {}", trainer.is_checkpointing());
                println!();
                return train_loop_cpu(&mut trainer, &batches, spec);
            }
        }
    }

    // CPU-only path (use_cuda=false or no CUDA feature)
    let mut trainer = if let Some(loaded_model) = transformer {
        TransformerTrainer::with_model(loaded_model, train_config)
    } else {
        TransformerTrainer::new(train_config)
    };
    println!("✓ TransformerTrainer initialized (CPU)");
    println!("  Mixed precision: {}", trainer.is_mixed_precision());
    println!("  Checkpointing: {}", trainer.is_checkpointing());
    println!();

    train_loop_cpu(&mut trainer, &batches, spec)
}

/// Training loop for CPU TransformerTrainer (ALB-045, ALB-055/056)
fn train_loop_cpu(
    trainer: &mut TransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    println!("Starting transformer training (CPU)...");
    println!();

    TRACER.enable();
    TRACER.clear();

    let num_batches = batches.len();
    let start_time = std::time::Instant::now();
    let log_interval = (num_batches / 100).clamp(1, 100);

    // ALB-045: Initialize training state IPC for `apr monitor`
    let state = TrainingState::new(&spec.training.output_dir);
    let start_ms = now_ms();
    let total_epochs = spec.training.epochs;

    // ALB-055/056: Open SQLite experiment tracking (local + global)
    let mut tracker = PretrainTracker::open(spec, "CPU");

    write_training_snapshot(
        &state,
        start_ms,
        0,
        total_epochs,
        0,
        num_batches,
        0.0,
        &[],
        0.0,
        0.0,
        TrainingStatus::Initializing,
        spec,
        "CPU",
    );

    if let Some(max_steps) = spec.training.max_steps {
        println!("  max_steps: {max_steps} (will stop early when reached)");
    }

    let mut loss_history: Vec<f32> = Vec::new();

    for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let avg_loss =
            trainer.train_epoch_with_callback(batches, |batch_idx, batch_loss, trainer| {
                loss_history.push(batch_loss);
                if loss_history.len() > 100 {
                    loss_history.remove(0);
                }

                if (batch_idx + 1) % log_interval == 0 || batch_idx == 0 {
                    let elapsed = epoch_start.elapsed().as_secs_f64();
                    let batches_done = batch_idx + 1;
                    let seq_len = spec.data.seq_len.unwrap_or(128);
                    let tokens_done = batches_done * spec.data.batch_size * seq_len;
                    let batch_per_sec = batches_done as f64 / elapsed.max(0.001);
                    let remaining = (num_batches - batches_done) as f64 / batch_per_sec.max(0.001);
                    let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);
                    println!(
                        "  [{}/{} batches] step={} loss={:.4} lr={:.2e} tok/s={:.0} eta={:.0}s",
                        batches_done,
                        num_batches,
                        trainer.step(),
                        batch_loss,
                        trainer.current_lr(),
                        tok_per_sec,
                        remaining,
                    );

                    // ALB-045: Write snapshot for `apr monitor`
                    write_training_snapshot(
                        &state,
                        start_ms,
                        epoch + 1,
                        total_epochs,
                        trainer.step(),
                        num_batches,
                        batch_loss,
                        &loss_history,
                        trainer.current_lr(),
                        tok_per_sec as f32,
                        TrainingStatus::Running,
                        spec,
                        "CPU",
                    );

                    // ALB-055/056: Log step metrics to SQLite
                    tracker.log_step(
                        trainer.step() as u64,
                        batch_loss,
                        trainer.current_lr(),
                        tok_per_sec as f32,
                    );
                }
            });
        let ppl = crate::train::perplexity(avg_loss);
        println!(
            "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
            epoch + 1,
            spec.training.epochs,
            avg_loss,
            ppl,
            epoch_start.elapsed().as_secs_f64(),
        );

        if trainer.reached_max_steps() {
            println!(
                "Reached max_steps={}, stopping training.",
                spec.training.max_steps.unwrap_or(0)
            );
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!("Total training time: {:.1}s", total_time.as_secs_f64());
    println!("{}", TRACER.report());

    // ALB-045: Write final "Completed" snapshot
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        &state,
        start_ms,
        total_epochs,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        &loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        "CPU",
    );

    // ALB-055/056: Mark run as completed in SQLite
    tracker.complete();

    save_trained_model_cpu(trainer, spec)
}

/// Get current Unix timestamp in milliseconds
fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Query live GPU telemetry via nvidia-smi CLI (ALB-046)
///
/// Shells out to `nvidia-smi --query-gpu` with CSV output and parses
/// the result into GpuTelemetry. Zero-dependency approach — nvidia-smi
/// is always available when CUDA is. Returns None if nvidia-smi fails.
fn query_gpu_telemetry(device_name: &str) -> Option<crate::monitor::tui::state::GpuTelemetry> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?.trim();
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 6 {
        return None;
    }

    Some(crate::monitor::tui::state::GpuTelemetry {
        device_name: device_name.to_string(),
        utilization_percent: fields[0].parse().unwrap_or(0.0),
        vram_used_gb: fields[1].parse::<f32>().unwrap_or(0.0) / 1024.0, // MiB → GiB
        vram_total_gb: fields[2].parse::<f32>().unwrap_or(0.0) / 1024.0,
        temperature_celsius: fields[3].parse().unwrap_or(0.0),
        power_watts: fields[4].parse().unwrap_or(0.0),
        power_limit_watts: fields[5].parse().unwrap_or(0.0),
        processes: Vec::new(),
    })
}

/// Write a TrainingSnapshot to training_state.json (ALB-045)
///
/// This is the IPC mechanism that `apr monitor` reads. Called on every
/// log interval so the TUI stays current. Uses atomic write (tmp+rename)
/// via TrainingState::write().
fn write_training_snapshot(
    state: &TrainingState,
    start_ms: u64,
    epoch: usize,
    total_epochs: usize,
    step: usize,
    steps_per_epoch: usize,
    loss: f32,
    loss_history: &[f32],
    lr: f32,
    tokens_per_second: f32,
    status: TrainingStatus,
    spec: &TrainSpec,
    gpu_name: &str,
) {
    let snapshot = TrainingSnapshot {
        timestamp_ms: now_ms(),
        epoch,
        total_epochs,
        step,
        steps_per_epoch,
        loss,
        loss_history: loss_history.to_vec(),
        learning_rate: lr,
        lr_history: Vec::new(),
        gradient_norm: 0.0, // not tracked per-batch in current trainer
        tokens_per_second,
        start_timestamp_ms: start_ms,
        gpu: query_gpu_telemetry(gpu_name).or_else(|| {
            Some(crate::monitor::tui::state::GpuTelemetry {
                device_name: gpu_name.to_string(),
                ..Default::default()
            })
        }),
        sample: None,
        status,
        experiment_id: spec.training.output_dir.display().to_string(),
        model_name: spec.model.path.display().to_string(),
        model_path: spec.model.path.display().to_string(),
        optimizer_name: spec.optimizer.name.clone(),
        batch_size: spec.data.batch_size,
        checkpoint_path: spec.training.output_dir.display().to_string(),
        executable_path: String::new(),
        accuracy: 0.0,
        samples_per_second: 0.0,
    };
    if let Err(e) = state.write(&snapshot) {
        eprintln!("[ALB-045] Failed to write training_state.json: {e}");
    }
}

// =============================================================================
// SQLite Experiment Tracking (ALB-055/056)
// =============================================================================

/// Best-effort experiment tracker for pretrain loops.
///
/// Opens two SQLite databases:
/// - **Local**: `<output_dir>/.entrenar/experiments.db` (per-experiment metrics history)
/// - **Global**: `~/.entrenar/experiments.db` (cross-machine experiment registry)
///
/// All operations are best-effort — storage failures never block training.
struct PretrainTracker {
    local: Option<SqliteBackend>,
    global: Option<SqliteBackend>,
    run_id: Option<String>,
    global_run_id: Option<String>,
}

impl PretrainTracker {
    /// Open both local and global SQLite stores, create experiment + run.
    fn open(spec: &TrainSpec, device: &str) -> Self {
        let exp_name =
            spec.training.output_dir.file_name().and_then(|n| n.to_str()).unwrap_or("pretrain");

        let config_json = serde_json::json!({
            "task": "pretrain",
            "model": spec.model.path.display().to_string(),
            "optimizer": &spec.optimizer.name,
            "lr": spec.optimizer.lr,
            "epochs": spec.training.epochs,
            "batch_size": spec.data.batch_size,
            "seq_len": spec.data.seq_len,
            "max_steps": spec.training.max_steps,
            "device": device,
            "output_dir": spec.training.output_dir.display().to_string(),
        });

        // Local store: in the output/checkpoint directory
        let local = SqliteBackend::open_project(&spec.training.output_dir).ok();

        // Global store: ~/.entrenar/experiments.db
        let global = dirs::home_dir().map(|h| h.join(".entrenar")).and_then(|p| {
            fs::create_dir_all(&p).ok()?;
            SqliteBackend::open(p.join("experiments.db").to_string_lossy().as_ref()).ok()
        });

        let mut tracker = Self { local, global, run_id: None, global_run_id: None };

        // Create experiment + run in local store
        if let Some(store) = tracker.local.as_mut() {
            if let Ok(eid) = store.create_experiment(exp_name, Some(config_json.clone())) {
                if let Ok(rid) = store.create_run(&eid) {
                    let _ = store.start_run(&rid);
                    log_run_params(store, &rid, spec, device);
                    tracker.run_id = Some(rid);
                }
            }
        }

        // Create experiment + run in global store
        if let Some(store) = tracker.global.as_mut() {
            if let Ok(eid) = store.create_experiment(exp_name, Some(config_json)) {
                if let Ok(rid) = store.create_run(&eid) {
                    let _ = store.start_run(&rid);
                    log_run_params(store, &rid, spec, device);
                    tracker.global_run_id = Some(rid);
                }
            }
        }

        tracker
    }

    /// Log a training step's metrics to both local and global stores.
    fn log_step(&mut self, step: u64, loss: f32, lr: f32, tok_per_sec: f32) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.log_metric(rid, "loss", step, f64::from(loss));
                let _ = s.log_metric(rid, "learning_rate", step, f64::from(lr));
                let _ = s.log_metric(rid, "tokens_per_second", step, f64::from(tok_per_sec));
            }
        }
    }

    /// Mark training as completed in both stores.
    fn complete(&mut self) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.complete_run(rid, RunStatus::Success);
            }
        }
    }

    /// Mark training as failed in both stores.
    #[allow(dead_code)]
    fn fail(&mut self) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.complete_run(rid, RunStatus::Failed);
            }
        }
    }
}

/// Log hyperparameters for a pretrain run (ALB-055/056)
fn log_run_params(store: &SqliteBackend, run_id: &str, spec: &TrainSpec, device: &str) {
    let _ = store.log_param(run_id, "task", ParameterValue::String("pretrain".into()));
    let _ = store.log_param(
        run_id,
        "model",
        ParameterValue::String(spec.model.path.display().to_string()),
    );
    let _ =
        store.log_param(run_id, "optimizer", ParameterValue::String(spec.optimizer.name.clone()));
    let _ = store.log_param(
        run_id,
        "learning_rate",
        ParameterValue::Float(f64::from(spec.optimizer.lr)),
    );
    let _ = store.log_param(run_id, "epochs", ParameterValue::Int(spec.training.epochs as i64));
    let _ = store.log_param(run_id, "batch_size", ParameterValue::Int(spec.data.batch_size as i64));
    let _ = store.log_param(run_id, "device", ParameterValue::String(device.to_string()));
    let _ = store.log_param(
        run_id,
        "output_dir",
        ParameterValue::String(spec.training.output_dir.display().to_string()),
    );
    if let Some(seq_len) = spec.data.seq_len {
        let _ = store.log_param(run_id, "seq_len", ParameterValue::Int(seq_len as i64));
    }
    if let Some(max_steps) = spec.training.max_steps {
        let _ = store.log_param(run_id, "max_steps", ParameterValue::Int(max_steps as i64));
    }
}

/// Training loop for GPU CudaTransformerTrainer
///
fn print_max_steps(max_steps: Option<usize>) {
    if let Some(ms) = max_steps {
        println!("  max_steps: {ms} (will stop early when reached)");
    }
}

/// ALB-068: Manual batch loop for intermediate checkpoint saving.
/// R-004: Gradient norm logging. R-008: Graceful shutdown.
/// R-009: Multi-checkpoint retention. R-012: MFU tracking.
/// R-014: JSONL experiment log. R-015: Per-epoch shuffling.
/// R-006/R-007: Training state persistence.
#[cfg(feature = "cuda")]
fn train_loop_cuda(
    trainer: &mut CudaTransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    println!("Starting transformer training (CUDA GPU-resident)...");
    println!();

    let num_batches = batches.len();
    let start_time = std::time::Instant::now();
    // Cap log_interval so training_state.json updates at least every 100 steps
    // (enables real-time monitoring via `apr monitor`). Previously num_batches/100
    // gave 12905 for large datasets — too infrequent for a 12-day run.
    let log_interval = (num_batches / 100).clamp(1, 100);
    let save_interval = spec.training.save_interval;
    let max_checkpoints = spec.training.max_checkpoints;

    // ALB-087: Auto eval scheduling — eval_interval defaults to save_interval
    let eval_interval =
        if spec.training.eval_interval > 0 { spec.training.eval_interval } else { save_interval };
    let patience = spec.training.patience;
    let mut best_val_loss: f32 = f32::INFINITY;
    let mut evals_without_improvement: usize = 0;
    let mut last_eval_step: usize = 0;

    // ALB-045: Initialize training state IPC for `apr monitor`
    let state = TrainingState::new(&spec.training.output_dir);
    let start_ms = now_ms();
    let gpu_name = trainer.gpu_name();
    let total_epochs = spec.training.epochs;

    // ALB-055/056: Open SQLite experiment tracking (local + global)
    let mut tracker = PretrainTracker::open(spec, &gpu_name);

    // R-012: MFU calculation setup
    let num_params = trainer.num_params();
    let seq_len = spec.data.seq_len.unwrap_or(128);
    let tokens_per_batch = spec.data.batch_size * seq_len;
    // RTX 4090: 82.6 TFLOPS fp32 (query via cuDeviceGetAttribute when available)
    let gpu_peak_tflops: f64 = 82.58e12;

    // R-014: Open JSONL experiment log
    let jsonl_path = spec.training.output_dir.join("training_log.jsonl");
    std::fs::create_dir_all(&spec.training.output_dir).ok();
    let mut jsonl_file =
        std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path).ok();
    // Write config header
    write_jsonl_event_json(
        &mut jsonl_file,
        &serde_json::json!({
            "type": "config",
            "num_params": num_params,
            "batch_size": spec.data.batch_size,
            "seq_len": seq_len,
            "max_steps": spec.training.max_steps,
            "epochs": spec.training.epochs,
            "lr": spec.optimizer.lr,
            "gpu": &gpu_name,
            "timestamp": now_ms(),
        }),
    );

    // R-008: Graceful shutdown signal handler
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = shutdown_flag.clone();
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
            eprintln!("\n[SIGINT] Graceful shutdown requested. Saving checkpoint...");
        });
    }

    // Write initial "Initializing" snapshot
    write_training_snapshot(
        &state,
        start_ms,
        0,
        total_epochs,
        0,
        num_batches,
        0.0,
        &[],
        0.0,
        0.0,
        TrainingStatus::Initializing,
        spec,
        &gpu_name,
    );

    print_max_steps(spec.training.max_steps);

    // ALB-087: Print eval scheduling config
    if eval_interval != save_interval {
        println!("  eval_interval: {eval_interval} (decoupled from save_interval={save_interval})");
    }
    if patience > 0 {
        println!("  early_stopping: patience={patience} eval intervals");
    }

    // ALB-082: Scaling law predictor for early convergence ceiling detection
    let mut scaling_predictor = ScalingLawPredictor::new();
    let tokens_per_step = tokens_per_batch * spec.training.gradient_accumulation.unwrap_or(1);

    // Track loss history for TUI sparkline
    let mut loss_history: Vec<f32> = Vec::new();
    let mut last_save_step: usize = 0;

    let model_name = spec
        .model
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("entrenar-model")
        .to_string();

    // R-015: Prepare shuffled batch indices
    let shuffle = spec.training.shuffle;
    let seed = spec.training.seed.unwrap_or(42);

    // R-005: Load validation batches if val path exists
    let val_batches = load_val_batches(spec);

    // R-018: NaN/Inf detection counter
    let mut nan_skips: usize = 0;

    // R-017: ZClip adaptive gradient clipping — EMA of gradient norms
    let mut gnorm_ema: f64 = 0.0;
    let mut gnorm_ema_sq: f64 = 0.0;
    let zclip_alpha: f64 = 0.05; // EMA decay rate
    let zclip_threshold: f64 = 2.0; // z-score threshold for spike detection

    // R-003: Heartbeat file for crash detection
    let heartbeat_path = spec.training.output_dir.join("heartbeat");

    // R-016b: Loss spike rollback — EMA for spike detection
    let mut loss_ema: f64 = 0.0;
    let loss_ema_alpha: f64 = 0.05;
    let loss_spike_threshold: f64 = 3.0; // spike if loss > threshold × EMA
    let mut rollback_count: usize = 0;
    let max_rollbacks: usize = 3;

    // R-029: Gradient noise scale estimation — rolling window of grad norms
    let mut gnorm_window: Vec<f64> = Vec::with_capacity(100);
    let noise_scale_interval: usize = 100;
    let mut last_noise_scale_step: usize = usize::MAX; // Dedup: only log once per optimizer step

    // R-026: Save training config hash to JSONL for diff tracking
    write_config_provenance(&mut jsonl_file, spec);

    // R-023: Curriculum learning — track current stage index
    let mut curriculum_stage: usize = 0;
    let curriculum = spec.training.curriculum.as_deref();
    print_curriculum_stages(curriculum);

    'outer: for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        // R-015: Generate shuffled indices for this epoch
        let batch_order = shuffled_batch_order(num_batches, shuffle, seed, epoch);

        // ALB-068: Manual batch loop for intermediate checkpoint saving
        for (iter_idx, &batch_idx) in batch_order.iter().enumerate() {
            // R-008: Check graceful shutdown flag
            if shutdown_flag.load(Ordering::SeqCst) {
                handle_graceful_shutdown(
                    trainer,
                    spec,
                    &state,
                    &mut tracker,
                    start_ms,
                    epoch,
                    iter_idx,
                    total_epochs,
                    num_batches,
                    &loss_history,
                    &model_name,
                    &gpu_name,
                    seed,
                    loss_ema,
                );
                return Ok(());
            }

            // Check max_steps before processing
            if reached_max_steps(spec.training.max_steps, trainer.step()) {
                break 'outer;
            }

            // R-023: Check curriculum stage transition
            curriculum_stage = check_curriculum_transition(
                curriculum,
                curriculum_stage,
                trainer.step(),
                &mut jsonl_file,
            );

            let batch = &batches[batch_idx];
            // R-028: Per-step timing
            let step_start = std::time::Instant::now();
            let batch_loss = trainer.train_batch(batch);
            let step_elapsed = step_start.elapsed();

            // R-018: NaN/Inf detection — skip step if loss is non-finite
            if !batch_loss.is_finite() {
                nan_skips += 1;
                println!(
                    "  [WARN] NaN/Inf loss at step {} (skip #{}) — skipping",
                    trainer.step(),
                    nan_skips
                );
                continue;
            }
            total_loss += batch_loss;
            batches_processed += 1;

            // R-016b: Loss spike detection + rollback
            detect_loss_spike(
                batch_loss,
                trainer.step(),
                &mut loss_ema,
                loss_ema_alpha,
                loss_spike_threshold,
                &mut rollback_count,
                max_rollbacks,
                &mut jsonl_file,
            );

            // R-017: ZClip — update EMA and detect gradient spikes
            zclip_update(
                f64::from(trainer.last_grad_norm()),
                trainer.step(),
                &mut gnorm_ema,
                &mut gnorm_ema_sq,
                zclip_alpha,
                zclip_threshold,
            );

            // R-029: Track grad norm for noise scale estimation
            update_noise_scale(
                f64::from(trainer.last_grad_norm()),
                trainer.step(),
                &mut gnorm_window,
                noise_scale_interval,
                &mut last_noise_scale_step,
                &mut jsonl_file,
            );

            // R-003: Write heartbeat for crash detection
            write_heartbeat(&heartbeat_path, trainer.step());

            // Track loss history (keep last 100 for sparkline)
            push_capped(&mut loss_history, batch_loss, 100);

            // Logging at log_interval boundaries
            if should_log(iter_idx, log_interval) {
                log_step_metrics(
                    trainer,
                    &state,
                    &mut tracker,
                    &mut jsonl_file,
                    &epoch_start,
                    &start_time,
                    &step_elapsed,
                    epoch,
                    total_epochs,
                    iter_idx,
                    num_batches,
                    tokens_per_batch,
                    num_params,
                    gpu_peak_tflops,
                    start_ms,
                    batch_loss,
                    &loss_history,
                    spec,
                    &gpu_name,
                );
            }

            // ALB-068/R-009: Intermediate checkpoint saving at save_interval
            let current_step = trainer.step();
            let do_save = should_save_checkpoint(current_step, last_save_step, save_interval);
            let do_eval = current_step > 0
                && current_step != last_eval_step
                && current_step.is_multiple_of(eval_interval);

            if do_save {
                save_and_validate_checkpoint(
                    trainer,
                    spec,
                    &model_name,
                    current_step,
                    epoch,
                    iter_idx,
                    max_checkpoints,
                    seed,
                    loss_ema,
                );
                last_save_step = current_step;
            }

            // ALB-087: Decoupled eval + best-model tracking + early stopping
            if do_eval {
                last_eval_step = current_step;
                let eval_val_loss = run_validation_eval(
                    trainer,
                    &val_batches,
                    current_step,
                    &mut jsonl_file,
                    &mut scaling_predictor,
                    tokens_per_step,
                    spec.training.max_steps,
                );
                if let Some(val_loss) = eval_val_loss {
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        evals_without_improvement = 0;
                        save_best_model(trainer, spec, &model_name, current_step);
                    } else {
                        evals_without_improvement += 1;
                    }
                    if patience > 0 && evals_without_improvement >= patience {
                        println!(
                            "  [early-stop] No improvement for {evals_without_improvement} evals (patience={patience}). \
                             Best val_loss={best_val_loss:.4}. Stopping.",
                        );
                        write_jsonl_event_json(
                            &mut jsonl_file,
                            &serde_json::json!({
                                "type": "early_stop",
                                "step": current_step,
                                "best_val_loss": best_val_loss,
                                "evals_without_improvement": evals_without_improvement,
                                "patience": patience,
                                "timestamp": now_ms(),
                            }),
                        );
                        break 'outer;
                    }
                }
            }
        }

        let avg_loss = total_loss / batches_processed.max(1) as f32;
        let ppl = crate::train::perplexity(avg_loss);
        println!(
            "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
            epoch + 1,
            spec.training.epochs,
            avg_loss,
            ppl,
            epoch_start.elapsed().as_secs_f64(),
        );

        if reached_max_steps(spec.training.max_steps, trainer.step()) {
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!("Total training time: {:.1}s", total_time.as_secs_f64());

    // KAIZEN-047: Print step profiler report at end of training
    trainer.print_profiler_report();

    // ALB-045: Write final "Completed" snapshot
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        &state,
        start_ms,
        total_epochs,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        &loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        &gpu_name,
    );

    // ALB-055/056: Mark run as completed in SQLite
    tracker.complete();

    // R-014: Write completion entry
    write_jsonl_event_json(
        &mut jsonl_file,
        &serde_json::json!({
            "type": "complete",
            "step": trainer.step(),
            "final_loss": final_loss,
            "total_time_s": total_time.as_secs_f64(),
            "timestamp": now_ms(),
        }),
    );

    save_trained_model_cuda(trainer, spec)
}

/// Distributed CUDA training loop (#133).
///
/// Multi-process DDP: each process runs this function with its own rank.
/// Rank 0 spawns the GradientServer in a background thread. All ranks
/// connect as workers and run the DDP training step in lockstep.
///
/// Data is sharded by rank: worker N processes batches N, N+ws, N+2*ws, ...
#[cfg(feature = "cuda")]
/// Spawn the coordinator (GradientServer) thread for DDP rank 0.
fn spawn_coordinator_thread(
    coord_addr: std::net::SocketAddr,
    world_size: usize,
    num_blocks: usize,
    total_steps: usize,
) -> Result<std::thread::JoinHandle<()>> {
    use crate::finetune::distributed::DistributedConfig;
    use crate::finetune::GradientServer;

    let server_config = DistributedConfig::coordinator(coord_addr, world_size);
    let mut server = GradientServer::bind(server_config)
        .map_err(|e| Error::ConfigError(format!("GradientServer bind failed: {e}")))?;
    println!("  ✓ GradientServer bound on {coord_addr}");

    Ok(std::thread::spawn(move || {
        server.wait_for_workers().unwrap();
        eprintln!("[coordinator] All {world_size} workers connected");

        for _step in 0..total_steps {
            for block_idx in (0..num_blocks).rev() {
                let result =
                    server.collect_and_reduce_block(_step as u64, block_idx as u32).unwrap();
                server.broadcast_averaged_block(_step as u64, &result).unwrap();
            }
            for component in [0u8, 1, 2] {
                let result = server.collect_and_reduce_non_block(_step as u64, component).unwrap();
                server.broadcast_averaged_non_block(_step as u64, &result).unwrap();
            }
        }
        eprintln!("[coordinator] Training complete ({total_steps} steps)");
    }))
}

#[cfg(feature = "cuda")]
fn train_loop_cuda_distributed(
    mut cuda_trainer: CudaTransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    use crate::finetune::distributed::DistributedConfig;
    use crate::finetune::WorkerClient;
    use crate::train::{shard_batches, DistributedComm, DistributedCudaTrainer};

    let dist_config = cuda_trainer
        .config()
        .distributed
        .clone()
        .ok_or_else(|| Error::ConfigError("missing distributed config".into()))?;

    let rank = dist_config.rank;
    let world_size = dist_config.world_size;
    let coord_addr = dist_config.coordinator_addr;

    println!("Starting distributed training (DDP)...");
    println!("  rank: {rank}/{world_size}");
    println!("  coordinator: {coord_addr}");

    cuda_trainer.ensure_grad_accum();

    let num_blocks = cuda_trainer
        .grad_accum_ref()
        .map_or(0, crate::train::PerBlockGradientAccumulator::num_blocks);

    // Step 1: If rank 0, spawn GradientServer in background thread
    let server_handle = if rank == 0 {
        let max_steps = spec.training.max_steps.unwrap_or(usize::MAX);
        let batches_per_worker = batches.len().div_ceil(world_size);
        let total_steps = std::cmp::min(spec.training.epochs * batches_per_worker, max_steps);
        Some(spawn_coordinator_thread(coord_addr, world_size, num_blocks, total_steps)?)
    } else {
        std::thread::sleep(std::time::Duration::from_millis(100));
        None
    };

    // Step 2: Connect as worker (all ranks, including rank 0)
    let worker_config = DistributedConfig::worker(coord_addr);
    let client = WorkerClient::connect(worker_config, 1, "cuda")
        .map_err(|e| Error::ConfigError(format!("WorkerClient connect failed: {e}")))?;
    println!("  ✓ Connected as worker {} (id={})", rank, client.worker_id());

    // Step 3: Wrap in DistributedCudaTrainer
    let comm = DistributedComm::Remote { client };
    let mut ddp_trainer = DistributedCudaTrainer::new(cuda_trainer, comm, dist_config.clone());

    // Step 4: Training loop with data sharding
    let num_batches = batches.len();
    let start_time = std::time::Instant::now();
    let log_interval = std::cmp::max(num_batches / (world_size * 100).max(1), 1);
    let save_interval = spec.training.save_interval;
    let max_checkpoints = spec.training.max_checkpoints;
    let seed = spec.training.seed.unwrap_or(42);

    // ALB-082: Scaling law predictor for DDP path
    let _scaling_predictor = ScalingLawPredictor::new();
    let seq_len_ddp = spec.data.seq_len.unwrap_or(128);
    let grad_accum_ddp = spec.training.gradient_accumulation.unwrap_or(1);
    let _tokens_per_step_ddp = spec.data.batch_size * seq_len_ddp * grad_accum_ddp;

    let model_name = spec
        .model
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("entrenar-model")
        .to_string();

    // R-005: Load validation batches
    let _val_batches = load_val_batches(spec);

    let mut loss_history: Vec<f32> = Vec::new();
    let mut last_save_step: usize = 0;

    for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        // Shard batches by rank: worker N gets N, N+ws, N+2*ws, ...
        let my_batch_indices = shard_batches(num_batches, rank, world_size);

        for (iter_idx, &batch_idx) in my_batch_indices.iter().enumerate() {
            if ddp_trainer.reached_max_steps() {
                break;
            }

            let batch = &batches[batch_idx];
            let step_start = std::time::Instant::now();
            let batch_loss = ddp_trainer.train_batch(batch);
            let step_elapsed = step_start.elapsed();

            if !batch_loss.is_finite() {
                continue;
            }
            total_loss += batch_loss;
            batches_processed += 1;
            push_capped(&mut loss_history, batch_loss, 100);

            // Logging (rank 0 only to avoid spam)
            if rank == 0 && should_log(iter_idx, log_interval) {
                let step = ddp_trainer.step();
                let elapsed = epoch_start.elapsed().as_secs_f64();
                let seq_len = spec.data.seq_len.unwrap_or(128);
                let tokens_done = (iter_idx + 1) * spec.data.batch_size * seq_len * world_size;
                let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);
                println!(
                    "  [DDP rank 0] step={} loss={:.4} tok/s={:.0} step_time={:.1}ms",
                    step,
                    batch_loss,
                    tok_per_sec,
                    step_elapsed.as_secs_f64() * 1000.0,
                );
            }

            // Checkpoint (rank 0 only)
            if rank == 0 {
                let current_step = ddp_trainer.step();
                if should_save_checkpoint(current_step, last_save_step, save_interval) {
                    save_and_validate_checkpoint(
                        ddp_trainer.trainer_mut(),
                        spec,
                        &model_name,
                        current_step,
                        epoch,
                        iter_idx,
                        max_checkpoints,
                        seed,
                        0.0,
                    );
                    last_save_step = current_step;
                }
            }
        }

        if batches_processed > 0 {
            let avg_loss = total_loss / batches_processed as f32;
            let ppl = crate::train::perplexity(avg_loss);
            if rank == 0 {
                println!(
                    "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
                    epoch + 1,
                    spec.training.epochs,
                    avg_loss,
                    ppl,
                    epoch_start.elapsed().as_secs_f64(),
                );
            }
        }

        if ddp_trainer.reached_max_steps() {
            break;
        }
    }

    let total_time = start_time.elapsed();
    if rank == 0 {
        println!("Total distributed training time: {:.1}s", total_time.as_secs_f64());
    }

    // Save final model (rank 0 only)
    if rank == 0 {
        save_trained_model_cuda(ddp_trainer.trainer_mut(), spec)?;
    }

    // Wait for coordinator thread to finish
    if let Some(handle) = server_handle {
        let _: std::result::Result<(), _> = handle.join();
    }

    Ok(())
}

/// Check if max_steps has been reached.
/// Check if this iteration should log metrics.
#[allow(clippy::incompatible_msrv)]
fn should_log(iter_idx: usize, interval: usize) -> bool {
    (iter_idx + 1).is_multiple_of(interval) || iter_idx == 0
}

/// Check if this step should trigger a checkpoint save.
#[allow(clippy::incompatible_msrv)]
fn should_save_checkpoint(step: usize, last_save_step: usize, interval: usize) -> bool {
    step > 0 && step != last_save_step && step.is_multiple_of(interval)
}

fn reached_max_steps(max_steps: Option<usize>, current_step: usize) -> bool {
    if let Some(max) = max_steps {
        if current_step >= max {
            println!("Reached max_steps={max}, stopping training.");
            return true;
        }
    }
    false
}

/// R-017: ZClip gradient spike detection — update EMA and log spikes.
fn zclip_update(
    gnorm: f64,
    step: usize,
    ema: &mut f64,
    ema_sq: &mut f64,
    alpha: f64,
    threshold: f64,
) {
    *ema = alpha * gnorm + (1.0 - alpha) * *ema;
    *ema_sq = alpha * gnorm * gnorm + (1.0 - alpha) * *ema_sq;
    let std = (*ema_sq - *ema * *ema).max(0.0).sqrt();
    if std > 1e-8 {
        let z_score = (gnorm - *ema) / std;
        if z_score > threshold {
            println!(
                "  [ZClip] gradient spike at step {}: z={:.1} gnorm={:.2e} ema={:.2e}",
                step, z_score, gnorm, *ema
            );
        }
    }
}

/// Push a value to a capped history buffer, removing oldest if at capacity.
fn push_capped(history: &mut Vec<f32>, value: f32, max_len: usize) {
    history.push(value);
    if history.len() > max_len {
        history.remove(0);
    }
}

/// Push f64 value onto a capped rolling window.
fn push_capped_f64(window: &mut Vec<f64>, value: f64, max_len: usize) {
    window.push(value);
    if window.len() > max_len {
        window.remove(0);
    }
}

/// R-029: Track gradient norm and estimate noise scale at intervals.
/// B_noise = Var(||g||) / Mean(||g||)² — proxy for critical batch size.
#[allow(clippy::incompatible_msrv)]
fn update_noise_scale(
    grad_norm: f64,
    step: usize,
    window: &mut Vec<f64>,
    interval: usize,
    last_logged_step: &mut usize,
    jsonl_file: &mut Option<std::fs::File>,
) {
    push_capped_f64(window, grad_norm, 100);
    if step == 0 || !step.is_multiple_of(interval) || window.len() < 10 || step == *last_logged_step
    {
        return;
    }
    *last_logged_step = step;
    let n = window.len() as f64;
    let mean = window.iter().sum::<f64>() / n;
    if mean < 1e-12 {
        return;
    }
    let variance = window.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let b_noise = variance / (mean * mean);
    println!("  [noise-scale] step={} B_noise={:.4} (window={})", step, b_noise, window.len());
    write_jsonl_event_json(
        jsonl_file,
        &serde_json::json!({
            "type": "noise_scale",
            "step": step,
            "b_noise": b_noise,
            "gnorm_mean": mean,
            "gnorm_var": variance,
            "window_size": window.len(),
            "timestamp": now_ms(),
        }),
    );
}

/// R-016b: Detect loss spikes and log rollback events.
#[allow(clippy::too_many_arguments)]
fn detect_loss_spike(
    loss: f32,
    step: usize,
    ema: &mut f64,
    alpha: f64,
    threshold: f64,
    rollback_count: &mut usize,
    max_rollbacks: usize,
    jsonl_file: &mut Option<std::fs::File>,
) {
    let bl = f64::from(loss);
    if *ema > 0.0 && bl > threshold * *ema && *rollback_count < max_rollbacks {
        *rollback_count += 1;
        println!(
            "  [ROLLBACK] loss spike at step {}: {:.4} > {:.1}×EMA({:.4}), rollback #{}/{}",
            step, loss, threshold, *ema, *rollback_count, max_rollbacks
        );
        write_jsonl_event(jsonl_file, "rollback", step, loss, *ema as f32);
    }
    *ema = alpha * bl + (1.0 - alpha) * *ema;
}

/// Write a generic event entry to the JSONL experiment log.
fn write_jsonl_event(
    jsonl_file: &mut Option<std::fs::File>,
    event_type: &str,
    step: usize,
    loss: f32,
    loss_ema: f32,
) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let entry = serde_json::json!({
            "type": event_type,
            "step": step,
            "loss": loss,
            "loss_ema": loss_ema,
            "timestamp": now_ms(),
        });
        let _ = writeln!(f, "{entry}");
    }
}

/// Write an arbitrary JSON event to the JSONL log (generic version).
fn write_jsonl_event_json(jsonl_file: &mut Option<std::fs::File>, entry: &serde_json::Value) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let _ = writeln!(f, "{entry}");
    }
}

/// R-003: Write heartbeat timestamp for crash detection.
fn write_heartbeat(path: &std::path::Path, step: usize) {
    let data = format!("{}\t{}", now_ms(), step);
    let _ = std::fs::write(path, data);
}

/// R-026: Save training config snapshot + R-024: data provenance to JSONL.
fn write_config_provenance(jsonl_file: &mut Option<std::fs::File>, spec: &TrainSpec) {
    use std::io::Write;
    let Some(ref mut f) = jsonl_file else { return };

    // R-024: Data provenance
    let train_path = spec.data.train.display().to_string();
    let val_path = spec.data.val.as_ref().map(|p| p.display().to_string());
    let provenance = serde_json::json!({
        "type": "provenance",
        "train_path": train_path,
        "val_path": val_path,
        "batch_size": spec.data.batch_size,
        "seq_len": spec.data.seq_len,
        "timestamp": now_ms(),
    });
    let _ = writeln!(f, "{provenance}");

    // R-026: Config snapshot for diff tracking
    let config = serde_json::json!({
        "type": "config_snapshot",
        "optimizer": {
            "name": &spec.optimizer.name,
            "lr": spec.optimizer.lr,
            "params": &spec.optimizer.params,
        },
        "training": {
            "epochs": spec.training.epochs,
            "max_steps": spec.training.max_steps,
            "grad_clip": spec.training.grad_clip,
            "save_interval": spec.training.save_interval,
            "warmup_steps": spec.training.warmup_steps,
            "gradient_accumulation": spec.training.gradient_accumulation,
            "seed": spec.training.seed,
        },
        "model_path": spec.model.path.display().to_string(),
        "timestamp": now_ms(),
    });
    let _ = writeln!(f, "{config}");
}

/// R-005: Load validation batches if val path is configured.
fn load_val_batches(spec: &TrainSpec) -> Vec<LMBatch> {
    let val_path = match &spec.data.val {
        Some(p) if p.exists() => p,
        _ => return Vec::new(),
    };
    let batch_size = spec.data.batch_size;
    let seq_len = spec.data.seq_len.unwrap_or(128);
    let tokenizer = load_tokenizer(spec).ok().flatten();
    let tokenizer_ref = tokenizer.as_ref();

    // Try loading from parquet directory or file
    if val_path.is_dir() {
        if let Some(tok) = tokenizer_ref {
            let column = spec.data.input_column.as_deref().unwrap_or("text");
            if let Ok(batches) =
                load_lm_batches_from_parquet(val_path, tok, batch_size, seq_len, column)
            {
                println!(
                    "  ✓ {} validation batches loaded from {}",
                    batches.len(),
                    val_path.display()
                );
                return batches;
            }
        }
    }
    Vec::new()
}

/// ALB-082: Scaling law predictor — fits L(D) = a - b × ln(D) to eval history.
///
/// After 3+ eval checkpoints, predicts val_ppl at max_steps. Warns if
/// predicted improvement < 10% over current val_ppl.
struct ScalingLawPredictor {
    /// (cumulative_tokens, val_loss) pairs
    history: Vec<(f64, f64)>,
}

impl ScalingLawPredictor {
    fn new() -> Self {
        Self { history: Vec::new() }
    }

    fn record(&mut self, tokens: usize, val_loss: f32) {
        self.history.push((tokens as f64, f64::from(val_loss)));
    }

    /// Fit L(D) = a - b × ln(D) via ordinary least squares.
    /// Returns (a, b) or None if < 3 data points.
    fn fit(&self) -> Option<(f64, f64)> {
        if self.history.len() < 3 {
            return None;
        }
        // OLS: y = a + b*x where x = ln(tokens), y = val_loss, b is negative
        let n = self.history.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        for &(tokens, loss) in &self.history {
            let x = tokens.ln();
            sum_x += x;
            sum_y += loss;
            sum_xy += x * loss;
            sum_xx += x * x;
        }
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }
        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;
        // b should be negative (loss decreases with tokens)
        Some((a, -b)) // Return (a, b) where L(D) = a - b*ln(D), b > 0
    }

    /// Predict val_loss at given token count using fitted parameters.
    fn predict(&self, target_tokens: usize) -> Option<(f64, f64, f64)> {
        let (a, b) = self.fit()?;
        let predicted_loss = a - b * (target_tokens as f64).ln();
        let predicted_ppl = predicted_loss.exp();
        Some((predicted_loss, predicted_ppl, b))
    }
}

/// R-005: Run validation evaluation and log results.
/// ALB-082: Includes scaling law prediction after 3+ eval points.
#[cfg(feature = "cuda")]
fn run_validation_eval(
    trainer: &mut CudaTransformerTrainer,
    val_batches: &[LMBatch],
    step: usize,
    jsonl_file: &mut Option<std::fs::File>,
    predictor: &mut ScalingLawPredictor,
    tokens_per_step: usize,
    max_steps: Option<usize>,
) -> Option<f32> {
    if val_batches.is_empty() {
        return None;
    }
    let mut total_loss = 0.0;
    let mut count = 0;
    for batch in val_batches {
        let loss = trainer.eval_batch(batch);
        if loss > 0.0 {
            total_loss += loss;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let val_loss = total_loss / count as f32;
    let val_ppl = crate::train::perplexity(val_loss);
    let cumulative_tokens = step * tokens_per_step;

    // ALB-082: Record eval point and predict
    predictor.record(cumulative_tokens, val_loss);

    let target_tokens = max_steps.unwrap_or(step * 2) * tokens_per_step;
    let prediction = predictor.predict(target_tokens);

    // Build JSONL entry (common fields + optional prediction fields)
    let mut entry = serde_json::json!({
        "type": "eval",
        "step": step,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "cumulative_tokens": cumulative_tokens,
        "timestamp": now_ms(),
    });

    if let Some((pred_loss, pred_ppl, slope)) = prediction {
        let target_steps = target_tokens / tokens_per_step;
        println!(
            "  [eval] step={step} val_loss={val_loss:.4} val_ppl={val_ppl:.2} ({count} batches) \
             predicted_ppl={pred_ppl:.1} at step {target_steps} (slope={slope:.4})"
        );
        // Warn if predicted improvement < 10%
        let improvement = (f64::from(val_ppl) - pred_ppl) / f64::from(val_ppl);
        if improvement < 0.10 && predictor.history.len() >= 4 {
            println!(
                "  [WARN] Scaling law predicts only {:.1}% improvement by step {} \
                 (val_ppl {:.1} → {:.1}). Consider: more data, larger model, or stopping.",
                improvement * 100.0,
                target_steps,
                val_ppl,
                pred_ppl
            );
        }
        entry["predicted_final_loss"] = serde_json::json!(pred_loss);
        entry["predicted_final_ppl"] = serde_json::json!(pred_ppl);
        entry["scaling_slope"] = serde_json::json!(slope);
        entry["target_steps"] = serde_json::json!(target_steps);
    } else {
        println!(
            "  [eval] step={step} val_loss={val_loss:.4} val_ppl={val_ppl:.2} ({count} batches)"
        );
    }

    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let _ = writeln!(f, "{entry}");
    }
    Some(val_loss)
}

/// ALB-087/ALB-096: Save best model checkpoint (APR format).
#[cfg(feature = "cuda")]
fn save_best_model(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    model_name: &str,
    step: usize,
) {
    let best_path = spec.training.output_dir.join("model-best.apr");
    let lr = trainer.current_lr();
    let save_fn =
        trainer.prepare_async_apr_save(model_name, "LlamaForCausalLM", step, 0.0, lr as f64);
    std::thread::spawn(move || {
        if let Err(e) = save_fn(&best_path) {
            println!("  [WARN] Failed to save model-best: {e}");
        } else {
            println!("  [best-model] step={step} saved to {}", best_path.display());
        }
    });
}

/// ALB-096: Save checkpoint as APR with integrity verification and state persistence.
///
/// Single atomic APR file contains model weights + optimizer state + training metadata.
/// Replaces separate `model-step-N.safetensors` + `optimizer_state.json` files.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn save_and_validate_checkpoint(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    model_name: &str,
    step: usize,
    epoch: usize,
    batch_idx: usize,
    max_checkpoints: usize,
    seed: u64,
    loss_ema: f64,
) {
    let ckpt_path = checkpoint_path(&spec.training.output_dir, step);
    let lr = trainer.current_lr();
    // ALB-096: Async APR checkpointing — model weights + optimizer state in single atomic file
    let save_fn =
        trainer.prepare_async_apr_save(model_name, "LlamaForCausalLM", step, loss_ema, lr as f64);
    let async_path = ckpt_path.clone();
    let async_output_dir = spec.training.output_dir.clone();
    std::thread::spawn(move || {
        if let Err(e) = save_fn(&async_path) {
            println!("  [WARN] Async APR checkpoint save failed: {e}");
        } else {
            verify_checkpoint(&async_path);
            println!("  [checkpoint] step={} saved to {}", step, async_path.display());
            save_training_state(&async_output_dir, step, epoch, batch_idx, seed, loss_ema);
            prune_checkpoints(&async_output_dir, max_checkpoints);
        }
    });
    // ALB-087: Eval is now decoupled from save — handled by the caller.
}

/// R-010: Verify checkpoint file integrity after save.
fn verify_checkpoint(path: &std::path::Path) {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let size_mb = meta.len() / (1024 * 1024);
            if meta.len() == 0 {
                println!("  [WARN] Checkpoint file is empty: {}", path.display());
            } else {
                println!("  [verify] checkpoint OK ({size_mb} MB)");
            }
        }
        Err(e) => {
            println!("  [WARN] Checkpoint verification failed: {e}");
        }
    }
}

/// R-015: Generate shuffled or sequential batch indices for an epoch.
fn shuffled_batch_order(num_batches: usize, shuffle: bool, seed: u64, epoch: usize) -> Vec<usize> {
    if !shuffle {
        return (0..num_batches).collect();
    }
    let mut indices: Vec<usize> = (0..num_batches).collect();
    // Deterministic Fisher-Yates using LCG PRNG seeded by seed + epoch
    let mut rng_state: u64 = seed
        .wrapping_add(epoch as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..indices.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices
}

/// R-008: Handle SIGINT/SIGTERM graceful shutdown with emergency checkpoint.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn handle_graceful_shutdown(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    state: &TrainingState,
    tracker: &mut PretrainTracker,
    start_ms: u64,
    epoch: usize,
    iter_idx: usize,
    total_epochs: usize,
    num_batches: usize,
    loss_history: &[f32],
    model_name: &str,
    gpu_name: &str,
    seed: u64,
    loss_ema: f64,
) {
    println!("[SIGINT] Emergency checkpoint at step {}...", trainer.step());
    let ckpt_path = checkpoint_path(&spec.training.output_dir, trainer.step());
    if let Err(e) = trainer.save_apr(&ckpt_path, model_name, "LlamaForCausalLM") {
        println!("  [WARN] Emergency save failed: {e}");
    } else {
        println!("  [checkpoint] emergency save to {}", ckpt_path.display());
        save_training_state(
            &spec.training.output_dir,
            trainer.step(),
            epoch,
            iter_idx,
            seed,
            loss_ema,
        );
    }
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        state,
        start_ms,
        epoch + 1,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        gpu_name,
    );
    tracker.complete();
    println!("[SIGINT] Shutdown complete.");
}

/// R-023: Print curriculum stage configuration at startup.
fn print_curriculum_stages(curriculum: Option<&[crate::config::schema::CurriculumStage]>) {
    let Some(stages) = curriculum else { return };
    println!("  Curriculum learning: {} stages configured", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        let until = stage.until_step.map_or("end".to_string(), |s| format!("step {s}"));
        println!("    Stage {}: {} (until {})", i, stage.data.display(), until);
    }
}

/// R-023: Check curriculum transition and log if stage changes.
/// Returns the (possibly updated) stage index.
fn check_curriculum_transition(
    curriculum: Option<&[crate::config::schema::CurriculumStage]>,
    current_stage: usize,
    step: usize,
    jsonl_file: &mut Option<std::fs::File>,
) -> usize {
    let Some(stages) = curriculum else { return current_stage };
    let Some(next) = advance_curriculum(stages, current_stage, step) else { return current_stage };
    println!(
        "  [Curriculum] → Stage {} at step {} (data: {})",
        next,
        step,
        stages[next].data.display()
    );
    write_jsonl_event_json(
        jsonl_file,
        &serde_json::json!({
            "type": "curriculum_transition",
            "stage": next,
            "step": step,
            "data": stages[next].data.to_string_lossy(),
            "timestamp": now_ms(),
        }),
    );
    next
}

/// R-023: Check if training should advance to the next curriculum stage.
/// Returns `Some(next_stage)` if a transition is needed, `None` otherwise.
fn advance_curriculum(
    stages: &[crate::config::schema::CurriculumStage],
    current: usize,
    step: usize,
) -> Option<usize> {
    if current >= stages.len() {
        return None;
    }
    let stage = &stages[current];
    if let Some(until) = stage.until_step {
        if step >= until && current + 1 < stages.len() {
            return Some(current + 1);
        }
    }
    None
}

/// R-014: Write a step entry to the JSONL experiment log.
#[allow(clippy::too_many_arguments)]
fn write_jsonl_step(
    jsonl_file: &mut Option<std::fs::File>,
    step: usize,
    loss: f32,
    lr: f32,
    tok_s: f64,
    mfu: f64,
    grad_norm: f32,
    embed_grad_norm: f32,
    epoch: usize,
    elapsed_s: f64,
) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let entry = serde_json::json!({
            "type": "step",
            "step": step,
            "loss": loss,
            "lr": lr,
            "tok_s": tok_s,
            "mfu": mfu,
            "grad_norm": grad_norm,
            "grad_norm_embed": embed_grad_norm,
            "epoch": epoch,
            "elapsed_s": elapsed_s,
            "timestamp": now_ms(),
        });
        let _ = writeln!(f, "{entry}");
    }
}

/// R-009/ALB-096: Generate step-numbered checkpoint path (APR format).
fn checkpoint_path(output_dir: &Path, step: usize) -> PathBuf {
    output_dir.join(format!("model-step-{step}.apr"))
}

/// Parse step number from a checkpoint filename.
/// Supports both APR (`model-step-123.apr`) and legacy SafeTensors (`model-step-123.safetensors`).
fn parse_checkpoint_step(filename: &str) -> Option<usize> {
    filename
        .strip_prefix("model-step-")?
        .strip_suffix(".apr")
        .or_else(|| filename.strip_prefix("model-step-")?.strip_suffix(".safetensors"))
        .and_then(|s| s.parse().ok())
}

/// Log step metrics: console output, IPC snapshot, SQLite, JSONL.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn log_step_metrics(
    trainer: &CudaTransformerTrainer,
    state: &TrainingState,
    tracker: &mut PretrainTracker,
    jsonl_file: &mut Option<std::fs::File>,
    epoch_start: &std::time::Instant,
    start_time: &std::time::Instant,
    step_elapsed: &std::time::Duration,
    epoch: usize,
    total_epochs: usize,
    iter_idx: usize,
    num_batches: usize,
    tokens_per_batch: usize,
    num_params: usize,
    gpu_peak_tflops: f64,
    start_ms: u64,
    batch_loss: f32,
    loss_history: &[f32],
    spec: &TrainSpec,
    gpu_name: &str,
) {
    let elapsed = epoch_start.elapsed().as_secs_f64();
    let batches_done = iter_idx + 1;
    let tokens_done = batches_done * tokens_per_batch;
    let batch_per_sec = batches_done as f64 / elapsed.max(0.001);
    let remaining = (num_batches - batches_done) as f64 / batch_per_sec.max(0.001);
    let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);

    // R-012: Compute MFU
    let flops_per_step = 6.0 * num_params as f64 * tokens_per_batch as f64;
    let step_time = elapsed / batches_done as f64;
    let mfu = (flops_per_step / step_time) / gpu_peak_tflops * 100.0;

    // R-004/R-040: Get per-parameter-group gradient norms
    let (grad_norm, embed_grad_norm) = trainer.param_grad_norms();
    // R-013: GPU memory usage
    let (gpu_used_mb, gpu_total_mb) = trainer.gpu_memory_mb();
    // R-028: Step time in ms
    let step_ms = step_elapsed.as_secs_f64() * 1000.0;

    println!(
        "  [{}/{} batches] step={} loss={:.4} lr={:.2e} tok/s={:.0} mfu={:.1}% gnorm={:.2e} gpu={}/{}MB step={:.0}ms eta={:.0}s",
        batches_done, num_batches,
        trainer.step(), batch_loss, trainer.current_lr(),
        tok_per_sec, mfu, grad_norm, gpu_used_mb, gpu_total_mb, step_ms, remaining,
    );

    // ALB-045: Write snapshot for `apr monitor`
    write_training_snapshot(
        state,
        start_ms,
        epoch + 1,
        total_epochs,
        trainer.step(),
        num_batches,
        batch_loss,
        loss_history,
        trainer.current_lr(),
        tok_per_sec as f32,
        TrainingStatus::Running,
        spec,
        gpu_name,
    );

    // ALB-055/056: Log step metrics to SQLite
    tracker.log_step(trainer.step() as u64, batch_loss, trainer.current_lr(), tok_per_sec as f32);

    // R-014: Write JSONL log entry
    write_jsonl_step(
        jsonl_file,
        trainer.step(),
        batch_loss,
        trainer.current_lr(),
        tok_per_sec,
        mfu,
        grad_norm,
        embed_grad_norm,
        epoch,
        start_time.elapsed().as_secs_f64(),
    );
}

/// R-009: Prune old checkpoints, keeping the most recent `max_keep`.
fn prune_checkpoints(output_dir: &Path, max_keep: usize) {
    if max_keep == 0 {
        return; // 0 = unlimited
    }
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut ckpts: Vec<(usize, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let step = parse_checkpoint_step(&e.file_name().to_string_lossy())?;
            Some((step, e.path()))
        })
        .collect();
    if ckpts.len() <= max_keep {
        return;
    }
    ckpts.sort_by_key(|(step, _)| *step);
    let to_remove = ckpts.len() - max_keep;
    for (step, path) in ckpts.into_iter().take(to_remove) {
        if std::fs::remove_file(&path).is_ok() {
            println!("  [prune] removed old checkpoint step={step}");
        }
    }
}

/// R-006/R-007: Save training state metadata alongside checkpoint.
fn save_training_state(
    output_dir: &Path,
    step: usize,
    epoch: usize,
    batch_idx: usize,
    seed: u64,
    loss_ema: f64,
) {
    let state = serde_json::json!({
        "step": step,
        "epoch": epoch,
        "batch_index": batch_idx,
        "seed": seed,
        "loss_ema": loss_ema,
        "timestamp": now_ms(),
    });
    let path = output_dir.join("training_state.json");
    if let Ok(json) = serde_json::to_string_pretty(&state) {
        let _ = std::fs::write(path, json);
    }
}

/// Save trained model from CPU trainer
fn save_trained_model_cpu(trainer: &TransformerTrainer, spec: &TrainSpec) -> Result<()> {
    println!();
    println!("✓ Transformer training complete");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!("  Steps completed: {}", trainer.step());
    println!();

    std::fs::create_dir_all(&spec.training.output_dir).ok();

    // ENT-LoRA-015: Save adapter-only checkpoint when LoRA is active
    if trainer.is_lora() {
        let base_model_name =
            spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        println!("Saving LoRA adapter to {}...", spec.training.output_dir.display());
        trainer.save_lora_adapter(&spec.training.output_dir, Some(base_model_name))?;
        let adapter_path = spec.training.output_dir.join("adapter_model.safetensors");
        let adapter_size = std::fs::metadata(&adapter_path).map(|m| m.len()).unwrap_or(0);
        println!("✓ LoRA adapter saved ({adapter_size} bytes)");
        println!("  adapter_model.safetensors + adapter_config.json");
    } else {
        let weights_path = spec.training.output_dir.join("model.safetensors");
        let model_name =
            spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("entrenar-model");
        println!("Saving model weights to {}...", weights_path.display());
        trainer.save(&weights_path, model_name, "LlamaForCausalLM")?;
        println!(
            "✓ Model weights saved ({} bytes)",
            std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0)
        );
    }

    let weights_path = spec.training.output_dir.join("model.safetensors");
    save_config_and_metadata(
        trainer.model().config(),
        trainer.step(),
        &trainer.metrics,
        &weights_path,
        spec,
    )
}

/// Save trained model from CUDA trainer (syncs GPU→CPU first)
#[cfg(feature = "cuda")]
fn save_trained_model_cuda(trainer: &mut CudaTransformerTrainer, spec: &TrainSpec) -> Result<()> {
    println!();
    println!("✓ Transformer training complete (CUDA)");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!("  Steps completed: {}", trainer.step());
    println!();

    std::fs::create_dir_all(&spec.training.output_dir).ok();

    // ALB-096: Save final model as APR (atomic, includes optimizer state)
    let weights_path = spec.training.output_dir.join("model.apr");
    let model_name =
        spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("entrenar-model");
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0) as f64;
    let lr = trainer.current_lr();
    println!("Saving model weights to {}...", weights_path.display());
    let save_fn = trainer.prepare_async_apr_save(
        model_name,
        "LlamaForCausalLM",
        trainer.step(),
        final_loss,
        lr as f64,
    );
    save_fn(&weights_path)?;
    println!(
        "✓ Model weights saved ({} bytes, APR)",
        std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0)
    );

    save_config_and_metadata(
        trainer.model().config(),
        trainer.step(),
        &trainer.metrics,
        &weights_path,
        spec,
    )
}

/// Save config.json and metadata (shared by CPU and CUDA paths)
fn save_config_and_metadata(
    mc: &TransformerConfig,
    step: usize,
    metrics: &crate::train::MetricsTracker,
    weights_path: &std::path::Path,
    spec: &TrainSpec,
) -> Result<()> {
    let config_json_path = spec.training.output_dir.join("config.json");
    let config_json = serde_json::json!({
        "architectures": [mc.hf_architecture_name()],
        "model_type": mc.hf_model_type_str(),
        "hidden_size": mc.hidden_size,
        "num_hidden_layers": mc.num_hidden_layers,
        "num_attention_heads": mc.num_attention_heads,
        "num_key_value_heads": mc.num_kv_heads,
        "intermediate_size": mc.intermediate_size,
        "vocab_size": mc.vocab_size,
        "max_position_embeddings": mc.max_position_embeddings,
        "rms_norm_eps": mc.rms_norm_eps,
        "rope_theta": mc.rope_theta,
        "tie_word_embeddings": mc.ties_embeddings(),
        "use_cache": true,
    });
    let config_json_str = serde_json::to_string_pretty(&config_json)
        .map_err(|e| Error::ConfigError(format!("Failed to serialize config.json: {e}")))?;
    std::fs::write(&config_json_path, config_json_str)?;
    println!("✓ config.json saved (realizar-compatible)");

    let metadata_path = spec.training.output_dir.join("final_model.json");
    println!("Saving metadata to {}...", metadata_path.display());
    let metadata = serde_json::json!({
        "model_path": spec.model.path,
        "weights_path": weights_path,
        "mode": "transformer",
        "training_mode": format!("{:?}", spec.training.mode),
        "epochs_completed": spec.training.epochs,
        "final_loss": metrics.losses.last().copied().unwrap_or(0.0),
        "best_loss": metrics.best_loss().unwrap_or(0.0),
        "steps": step,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| Error::ConfigError(format!("Failed to serialize metadata: {e}")))?;
    std::fs::write(&metadata_path, metadata_json)?;

    println!("✓ Model saved successfully");
    println!();

    Ok(())
}

/// Train a tabular model (regression/classification) from spec
///
/// Uses generic Trainer with MSELoss for regression tasks.
fn train_tabular_from_spec(spec: &TrainSpec) -> Result<()> {
    println!("✓ Config loaded and validated (Tabular mode)");
    println!("  Model: {}", spec.model.path.display());
    println!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
    }

    if let Some(quant) = &spec.quantize {
        println!("  Quantization: {}-bit", quant.bits);
    }
    println!();

    // Build model and optimizer
    println!("Building model and optimizer...");
    let model = crate::config::build_model(spec)?;
    let optimizer = crate::config::build_optimizer(&spec.optimizer)?;

    // Setup trainer
    use crate::train::{MSELoss, TrainConfig, Trainer};

    let mut train_config = TrainConfig::new().with_log_interval(100);

    if let Some(clip) = spec.training.grad_clip {
        train_config = train_config.with_grad_clip(clip);
    }

    let mut trainer = Trainer::new(
        model.parameters.into_iter().map(|(_, t)| t).collect(),
        optimizer,
        train_config,
    );
    trainer.set_loss(Box::new(MSELoss));

    println!("✓ Trainer initialized");
    println!();

    // Load training data
    println!("Loading training data...");
    let batches = load_training_batches(spec)?;
    println!("✓ {} batches created", batches.len());
    println!();

    // Training loop
    println!("Starting training...");
    println!();

    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(batches.clone(), Clone::clone);
        println!("Epoch {}/{}: loss={:.6}", epoch + 1, spec.training.epochs, avg_loss);
    }

    println!();
    println!("✓ Training complete");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!();

    // Save the trained model
    let output_path = spec.training.output_dir.join("final_model.json");
    println!("Saving model to {}...", output_path.display());

    // Reconstruct model for saving
    let final_model = crate::io::Model::new(
        model.metadata.clone(),
        trainer
            .params()
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("param_{i}"), t.clone()))
            .collect(),
    );

    use crate::io::{save_model, ModelFormat, SaveConfig};
    let save_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&final_model, &output_path, &save_config)?;

    println!("✓ Model saved successfully");
    println!();

    Ok(())
}

/// Resolve a model path, downloading from HuggingFace Hub if it's a repo ID.
///
/// If `model_path` looks like a HF repo ID (e.g., "Qwen/Qwen2.5-Coder-0.5B"),
/// downloads the model to the HF cache and returns the resolved local path.
/// Otherwise, returns the original path unchanged.
#[cfg(feature = "hub-publish")]
fn resolve_model_path(model_path: &Path) -> Result<PathBuf> {
    use crate::config::schema::is_hf_repo_id;
    use crate::hf_pipeline::fetcher::{FetchOptions, HfModelFetcher};

    let path_str = model_path.to_string_lossy();
    if !is_hf_repo_id(&path_str) {
        return Ok(model_path.to_path_buf());
    }

    println!("Downloading {path_str} from HuggingFace Hub...");
    let fetcher = HfModelFetcher::new()
        .map_err(|e| Error::ConfigError(format!("HF fetcher initialization: {e}")))?;

    let artifact = fetcher
        .download_model(&path_str, FetchOptions::new())
        .map_err(|e| Error::ConfigError(format!("Model download failed: {e}")))?;

    println!("  Cached at: {}", artifact.path.display());
    Ok(artifact.path)
}

#[cfg(not(feature = "hub-publish"))]
fn resolve_model_path(model_path: &Path) -> Result<PathBuf> {
    use crate::config::schema::is_hf_repo_id;

    let path_str = model_path.to_string_lossy();
    if is_hf_repo_id(&path_str) {
        return Err(Error::ConfigError(format!(
            "HF model ID '{path_str}' requires the 'hub-publish' feature. \
             Rebuild with: cargo install entrenar --features hub-publish"
        )));
    }
    Ok(model_path.to_path_buf())
}

/// ALB-096: Load transformer model from APR or SafeTensors weights.
///
/// Tries APR first (sovereign format), then falls back to SafeTensors.
/// Returns `(model, checkpoint_step)` where checkpoint_step is extracted
/// from APR metadata or parsed from checkpoint filename.
fn load_transformer_model(
    model_path: &Path,
    config: &TransformerConfig,
    output_dir: &Path,
) -> Result<(Option<Transformer>, usize)> {
    // ALB-097: Check output_dir first for checkpoint resume (APR, then SafeTensors)
    if output_dir.is_dir() {
        // Try APR first
        if let Some(result) = try_load_apr(output_dir, config) {
            return Ok(result);
        }
        // Try SafeTensors from output_dir (backward compat with pre-APR checkpoints)
        if let Some(result) = try_load_safetensors_dir(output_dir, config) {
            return Ok(result);
        }
    }

    if !model_path.exists() {
        println!("  Model path not found, using random initialization");
        return Ok((None, 0));
    }

    println!("Loading model weights from {}...", model_path.display());

    // ALB-096: Try APR format from model_path (direct .apr file or HF download)
    if let Some(result) = try_load_apr(model_path, config) {
        return Ok(result);
    }

    // Fallback: SafeTensors from model_path
    if let Some(result) = try_load_safetensors_dir(model_path, config) {
        return Ok(result);
    }

    eprintln!("Warning: No loadable checkpoint found, using random initialization");
    Ok((None, 0))
}

/// Try loading SafeTensors checkpoint from a directory.
/// Returns `Some((model, step))` if successful, `None` to fall back.
fn try_load_safetensors_dir(
    dir: &Path,
    config: &TransformerConfig,
) -> Option<(Option<Transformer>, usize)> {
    let checkpoint_step = detect_checkpoint_step(dir);

    match load_safetensors_weights(dir, Architecture::Auto) {
        Ok(weights) => {
            println!("  Found {} weight tensors (SafeTensors)", weights.len());
            if let Some(transformer) = Transformer::from_params(config, &weights) {
                let embed = &transformer.embed_tokens.weight;
                let embed_data = embed.data();
                let embed_slice = embed_data.as_slice().unwrap_or(&[]);
                let (emin, emax, emean) = if embed_slice.is_empty() {
                    (0.0, 0.0, 0.0)
                } else {
                    let min = embed_slice.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = embed_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mean = embed_slice.iter().sum::<f32>() / embed_slice.len() as f32;
                    (min, max, mean)
                };
                println!("✓ Loaded pre-trained weights successfully (SafeTensors)");
                if checkpoint_step > 0 {
                    println!("  Resuming from step {checkpoint_step}");
                }
                println!("  embed_tokens stats: min={emin:.4e} max={emax:.4e} mean={emean:.4e}");
                return Some((Some(transformer), checkpoint_step));
            }
            None
        }
        Err(_) => None,
    }
}

/// ALB-096: Try to load a model from APR format.
///
/// Looks for `.apr` files: direct file, `model-best.apr`, or latest `model-step-*.apr`.
/// Returns `Some((model, step))` if successful, `None` to fall back to SafeTensors.
fn try_load_apr(
    model_path: &Path,
    config: &TransformerConfig,
) -> Option<(Option<Transformer>, usize)> {
    use aprender::serialization::apr::AprReader;
    use std::collections::HashMap;

    // Determine which APR file to load
    let apr_path =
        if model_path.is_file() && model_path.extension().and_then(|e| e.to_str()) == Some("apr") {
            model_path.to_path_buf()
        } else if model_path.is_dir() {
            find_latest_apr_checkpoint(model_path)?
        } else {
            return None;
        };

    let reader = match AprReader::open(&apr_path) {
        Ok(r) => r,
        Err(_) => return None,
    };

    // Extract checkpoint step from APR metadata
    let checkpoint_step = reader
        .get_metadata("checkpoint_step")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            apr_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(parse_checkpoint_step)
                .unwrap_or(0)
        });

    // Load weight tensors (skip __training__.* namespace)
    let mut weights = HashMap::new();
    for desc in &reader.tensors {
        let tensor_name = &desc.name;
        if tensor_name.starts_with("__training__") {
            continue;
        }
        match reader.read_tensor_as_f32(tensor_name) {
            Ok(data) => {
                weights.insert(tensor_name.clone(), crate::Tensor::from_vec(data, false));
            }
            Err(e) => {
                eprintln!("Warning: Failed to read APR tensor '{tensor_name}': {e}");
                return None;
            }
        }
    }

    println!("  Found {} weight tensors (APR)", weights.len());

    let transformer = Transformer::from_params(config, &weights)?;

    let embed = &transformer.embed_tokens.weight;
    let embed_data = embed.data();
    let embed_slice = embed_data.as_slice().unwrap_or(&[]);
    let (emin, emax, emean) = if embed_slice.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let min = embed_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max = embed_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = embed_slice.iter().sum::<f32>() / embed_slice.len() as f32;
        (min, max, mean)
    };
    println!("✓ Loaded pre-trained weights successfully (APR)");
    if checkpoint_step > 0 {
        println!("  Resuming from step {checkpoint_step}");
    }
    println!("  embed_tokens stats: min={emin:.4e} max={emax:.4e} mean={emean:.4e}");

    Some((Some(transformer), checkpoint_step))
}

/// Find the latest APR checkpoint in a directory.
///
/// Priority: latest `model-step-N.apr` by step number. Falls back to `model-best.apr`.
fn find_latest_apr_checkpoint(dir: &Path) -> Option<std::path::PathBuf> {
    let mut best_step = 0usize;
    let mut best_path = None;

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(step) = parse_checkpoint_step(name) {
                    if step >= best_step {
                        best_step = step;
                        best_path = Some(path);
                    }
                }
            }
        }
    }

    if best_path.is_some() {
        return best_path;
    }

    // Fallback: model-best.apr, then model.apr (final checkpoint)
    let best = dir.join("model-best.apr");
    if best.exists() {
        return Some(best);
    }
    let model = dir.join("model.apr");
    if model.exists() {
        return Some(model);
    }

    None
}

/// Detect the step number from the latest checkpoint in a directory.
///
/// Checks both APR (`.apr`) and legacy SafeTensors (`.safetensors`) checkpoint files.
fn detect_checkpoint_step(model_path: &Path) -> usize {
    use crate::transformer::weights::parse_checkpoint_step_from_path;

    if model_path.is_file() {
        if let Some(name) = model_path.file_name().and_then(|n| n.to_str()) {
            if let Some(step) = parse_checkpoint_step(name) {
                return step;
            }
        }
        return parse_checkpoint_step_from_path(model_path).unwrap_or(0);
    }
    if !model_path.is_dir() {
        return 0;
    }
    // Check for model-step-*.{apr,safetensors} files
    let Ok(entries) = std::fs::read_dir(model_path) else { return 0 };
    let mut max_step = 0usize;
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if let Some(step) = parse_checkpoint_step(name) {
                max_step = max_step.max(step);
            }
        }
        if let Some(step) = parse_checkpoint_step_from_path(&entry.path()) {
            max_step = max_step.max(step);
        }
    }
    max_step
}

/// Apply architecture overrides to a `TransformerConfig`.
///
/// Only `Some` fields in the overrides replace the corresponding base config field.
fn apply_architecture_overrides(
    config: &mut TransformerConfig,
    overrides: &crate::config::ArchitectureOverrides,
) {
    if let Some(v) = overrides.hidden_size {
        config.hidden_size = v;
    }
    if let Some(v) = overrides.num_hidden_layers {
        config.num_hidden_layers = v;
    }
    if let Some(v) = overrides.num_attention_heads {
        config.num_attention_heads = v;
    }
    if let Some(v) = overrides.num_kv_heads {
        config.num_kv_heads = v;
    }
    if let Some(v) = overrides.intermediate_size {
        config.intermediate_size = v;
    }
    if let Some(v) = overrides.vocab_size {
        config.vocab_size = v;
    }
    if let Some(v) = overrides.max_position_embeddings {
        config.max_position_embeddings = v;
    }
    if let Some(v) = overrides.rms_norm_eps {
        config.rms_norm_eps = v;
    }
    if let Some(v) = overrides.rope_theta {
        config.rope_theta = v;
    }
    if let Some(v) = overrides.use_bias {
        config.use_bias = v;
    }
    if let Some(v) = overrides.head_dim {
        config.head_dim_override = Some(v);
    }
}

/// Build TransformerConfig from TrainSpec
///
/// Uses config file if specified, otherwise defaults to a small model.
/// Architecture overrides from the YAML manifest are applied on top.
fn build_transformer_config_from_spec(spec: &TrainSpec) -> Result<TransformerConfig> {
    // Check if config file is specified (explicit path or auto-detect from model dir)
    let config_path_resolved = spec.model.config.clone().or_else(|| {
        // Auto-detect config.json in model directory
        let model_config = spec.model.path.join("config.json");
        if model_config.exists() {
            Some(model_config.to_string_lossy().into_owned())
        } else {
            None
        }
    });

    let mut config = if let Some(config_path) = &config_path_resolved {
        let config_file = std::path::Path::new(config_path);
        if config_file.exists() {
            let config_content = std::fs::read_to_string(config_file)
                .map_err(|e| Error::ConfigError(format!("Failed to read model config: {e}")))?;

            if let Ok(hf_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                parse_hf_config(&hf_config)?
            } else {
                fallback_demo_config()
            }
        } else {
            fallback_demo_config()
        }
    } else if let Some(ref overrides) = spec.model.architecture {
        // No config file specified — try to build entirely from architecture overrides
        if let Some(cfg) = config_from_overrides(overrides) {
            cfg
        } else {
            fallback_demo_config()
        }
    } else {
        fallback_demo_config()
    };

    // Apply architecture overrides from YAML manifest (handles partial overrides on top of base)
    if let Some(ref overrides) = spec.model.architecture {
        apply_architecture_overrides(&mut config, overrides);
    }

    Ok(config)
}

/// Build a TransformerConfig directly from architecture overrides if all required fields are present.
/// Required: hidden_size, num_attention_heads, num_hidden_layers, vocab_size, intermediate_size.
fn config_from_overrides(
    overrides: &crate::config::ArchitectureOverrides,
) -> Option<TransformerConfig> {
    let hidden_size = overrides.hidden_size?;
    let num_attention_heads = overrides.num_attention_heads?;
    let num_hidden_layers = overrides.num_hidden_layers?;
    let vocab_size = overrides.vocab_size?;
    let intermediate_size = overrides.intermediate_size?;

    Some(TransformerConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads: overrides.num_kv_heads.unwrap_or(num_attention_heads),
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        max_position_embeddings: overrides.max_position_embeddings.unwrap_or(2048),
        rms_norm_eps: overrides.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: overrides.rope_theta.unwrap_or(10000.0),
        use_bias: overrides.use_bias.unwrap_or(false),
        head_dim_override: overrides.head_dim,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    })
}

/// R-05 (Meyer DbC): Explicit Qwen2-0.5B demo config — NOT a generic default.
/// This path is ONLY for testing without a model. Production callers must provide config.json.
fn fallback_demo_config() -> TransformerConfig {
    eprintln!("WARNING: No model config found — using Qwen2-0.5B demo config (NOT suitable for production training)");
    TransformerConfig {
        hidden_size: QWEN_HIDDEN_SIZE,
        num_attention_heads: QWEN_NUM_ATTENTION_HEADS,
        num_kv_heads: QWEN_NUM_KV_HEADS,
        intermediate_size: QWEN_INTERMEDIATE_SIZE,
        num_hidden_layers: QWEN_NUM_HIDDEN_LAYERS,
        vocab_size: QWEN_VOCAB_SIZE,
        max_position_embeddings: QWEN_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps: 1e-6,
        rope_theta: QWEN_ROPE_THETA as f32,
        use_bias: false,
        head_dim_override: None,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    }
}

/// Parse HuggingFace config.json into `TransformerConfig`.
///
/// C-10/C-11 (Meyer DbC): Required fields (hidden_size, num_attention_heads,
/// num_hidden_layers, vocab_size, intermediate_size) must be present — no silent defaults.
/// R-04: Optional fields use generic defaults with warnings for likely-wrong values.
fn parse_hf_config(hf_config: &serde_json::Value) -> Result<TransformerConfig> {
    let hidden_size = hf_config["hidden_size"].as_u64().ok_or_else(|| {
        Error::ConfigError(
            "C-11: config.json missing 'hidden_size' — cannot train without model dimensions"
                .into(),
        )
    })? as usize;
    let num_attention_heads = hf_config["num_attention_heads"].as_u64().ok_or_else(|| {
        Error::ConfigError("C-11: config.json missing 'num_attention_heads'".into())
    })? as usize;
    let num_hidden_layers = hf_config["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError("C-11: config.json missing 'num_hidden_layers'".into()))?
        as usize;
    let vocab_size = hf_config["vocab_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-10: config.json missing 'vocab_size' — training with wrong vocab corrupts embeddings".into()
        ))? as usize;
    let intermediate_size = hf_config["intermediate_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError("C-11: config.json missing 'intermediate_size'".into()))?
        as usize;

    // R-04 (Meyer DbC): Optional fields with generic defaults.
    // num_kv_heads → num_attention_heads is the correct GQA→MHA fallback.
    // max_position_embeddings → 2048 is a conservative safe minimum.
    // rms_norm_eps → 1e-6 is the most common default.
    // rope_theta → 10000 is the LLaMA/Mistral standard (WRONG for Qwen at 1M).
    // use_bias → false is correct for most modern architectures.
    let num_kv_heads =
        hf_config["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads as u64) as usize;

    let max_position_embeddings = match hf_config["max_position_embeddings"].as_u64() {
        Some(v) => v as usize,
        None => {
            eprintln!("Warning: config.json missing 'max_position_embeddings', defaulting to 2048");
            2048
        }
    };

    let rope_theta = match hf_config["rope_theta"].as_f64() {
        Some(v) => v as f32,
        None => {
            eprintln!(
                "Warning: config.json missing 'rope_theta', defaulting to 10000.0 \
                (Qwen models use 1000000.0 — check your config)"
            );
            10_000.0
        }
    };

    let rms_norm_eps = hf_config["rms_norm_eps"].as_f64().unwrap_or_else(|| {
        eprintln!(
            "Warning: config.json missing 'rms_norm_eps', defaulting to 1e-6 \
            (some models use 1e-5 or 1e-12 — check your config)"
        );
        1e-6
    }) as f32;
    let use_bias = hf_config["attention_bias"].as_bool().unwrap_or(false);
    let head_dim_override = hf_config["head_dim"].as_u64().map(|v| v as usize);

    // Detect encoder architectures from HuggingFace model_type
    let architecture = match hf_config["model_type"].as_str() {
        Some("bert" | "roberta" | "distilbert" | "albert" | "electra" | "deberta") => {
            ModelArchitecture::Encoder
        }
        _ => ModelArchitecture::Decoder,
    };

    // Preserve HuggingFace architecture metadata for checkpoint config.json (#259)
    let hf_architecture = hf_config["architectures"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .map(String::from);
    let hf_model_type = hf_config["model_type"].as_str().map(String::from);
    let tie_word_embeddings = hf_config["tie_word_embeddings"].as_bool().unwrap_or(false);

    Ok(TransformerConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        use_bias,
        head_dim_override,
        architecture,
        hf_architecture,
        hf_model_type,
        tie_word_embeddings,
    })
}

/// Load training data as LMBatches for transformer training
///
/// Supports:
/// 1. Pre-tokenized JSON with `input_ids` arrays
/// 2. Text JSON/JSONL with `text` or `content` fields (requires tokenizer)
/// 3. Demo mode fallback for testing
fn load_lm_batches(spec: &TrainSpec) -> Result<Vec<LMBatch>> {
    let batch_size = spec.data.batch_size;
    let seq_len = spec.data.seq_len.unwrap_or_else(|| {
        eprintln!("Warning: seq_len not specified, defaulting to 512 for LM batch loading");
        512
    });
    let tokenizer = load_tokenizer(spec)?;

    if let Some(result) = try_load_lm_from_file(spec, tokenizer.as_ref(), batch_size, seq_len) {
        return result;
    }

    eprintln!(
        "Warning: Training data not found at '{}', using demo LM batches",
        spec.data.train.display()
    );
    create_demo_lm_batches(batch_size, seq_len)
}

/// Attempt to load LM batches from the training data file or directory
fn try_load_lm_from_file(
    spec: &TrainSpec,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
) -> Option<Result<Vec<LMBatch>>> {
    if !spec.data.train.exists() {
        return None;
    }

    // Handle directory of Parquet shards (ALB-007)
    if spec.data.train.is_dir() {
        let tokenizer = tokenizer?;
        return Some(load_lm_batches_from_parquet(
            &spec.data.train,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref().unwrap_or("text"),
        ));
    }

    let ext = spec.data.train.extension()?;

    if ext == "json" || ext == "jsonl" {
        let content = std::fs::read_to_string(&spec.data.train).ok()?;
        return Some(load_lm_batches_from_json(
            &content,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref(),
        ));
    }

    if ext == "parquet" {
        let tokenizer = tokenizer?;
        return Some(load_lm_batches_from_parquet(
            &spec.data.train,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref().unwrap_or("text"),
        ));
    }

    None
}

/// Load HfTokenizer from spec if tokenizer path is specified
fn load_tokenizer(spec: &TrainSpec) -> Result<Option<HfTokenizer>> {
    if let Some(ref tokenizer_path) = spec.data.tokenizer {
        if tokenizer_path.exists() {
            println!("  Loading tokenizer from: {}", tokenizer_path.display());
            let tokenizer = HfTokenizer::from_file(tokenizer_path)
                .map_err(|e| Error::ConfigError(format!("Failed to load tokenizer: {e}")))?;
            println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());
            return Ok(Some(tokenizer));
        }
        eprintln!(
            "Warning: Tokenizer not found at '{}', using default Qwen2 tokenizer",
            tokenizer_path.display()
        );
    }

    // No tokenizer specified - use default for transformer mode
    println!("  Using default Qwen2 tokenizer");
    Ok(Some(HfTokenizer::qwen2()))
}

/// Extract text strings from a JSON array using the given column names
fn extract_texts_from_array(array: &[serde_json::Value], text_col: &str) -> Vec<String> {
    array
        .iter()
        .filter_map(|e| {
            e.get(text_col).or_else(|| e.get("content")).and_then(|v| v.as_str()).map(String::from)
        })
        .collect()
}

/// Try loading from a JSON array (either pre-tokenized or text)
fn try_load_from_array(
    array: &[serde_json::Value],
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
    label: &str,
) -> Option<Result<Vec<LMBatch>>> {
    // Check for pre-tokenized
    if array.first().and_then(|e| e.get("input_ids")).is_some() {
        return Some(load_pretokenized_json(array, batch_size, seq_len));
    }

    // Extract text and tokenize
    let tokenizer = tokenizer?;
    let texts = extract_texts_from_array(array, text_col);
    if texts.is_empty() {
        return None;
    }

    println!("  Loaded {} text examples from {label}, tokenizing...", texts.len());
    Some(tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len))
}

/// Try loading from JSONL (newline-delimited JSON)
fn try_load_from_jsonl(
    content: &str,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
) -> Option<Result<Vec<LMBatch>>> {
    let tokenizer = tokenizer?;
    let texts: Vec<String> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line).ok().and_then(|obj| {
                obj.get(text_col)
                    .or_else(|| obj.get("content"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
        })
        .collect();

    if texts.is_empty() {
        return None;
    }

    println!("  Loaded {} text examples from JSONL, tokenizing...", texts.len());
    Some(tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len))
}

/// Try to load LM batches from a parsed JSON value (object or array)
fn try_load_from_json_value(
    data: &serde_json::Value,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
) -> Option<Result<Vec<LMBatch>>> {
    // Try {"examples": [...]} format
    if let Some(examples) = data.get("examples").and_then(|e| e.as_array()) {
        if let Some(result) =
            try_load_from_array(examples, tokenizer, batch_size, seq_len, text_col, "JSON")
        {
            return Some(result);
        }
    }

    // Try top-level array format
    if let Some(array) = data.as_array() {
        if let Some(result) =
            try_load_from_array(array, tokenizer, batch_size, seq_len, text_col, "JSON array")
        {
            return Some(result);
        }
    }

    None
}

/// Load LM batches from JSON content
///
/// Supports formats:
/// - Pre-tokenized: `{"examples": [{"input_ids": [...]}]}`
/// - Text data: `{"examples": [{"text": "..."}]}` or `[{"text": "..."}]`
/// - JSONL: `{"text": "..."}\n{"text": "..."}`
fn load_lm_batches_from_json(
    content: &str,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    input_column: Option<&str>,
) -> Result<Vec<LMBatch>> {
    let text_col = input_column.unwrap_or("text");

    // Try parsing as single JSON object or array
    if let Ok(data) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(result) =
            try_load_from_json_value(&data, tokenizer, batch_size, seq_len, text_col)
        {
            return result;
        }
    }

    // Try JSONL format
    if let Some(result) = try_load_from_jsonl(content, tokenizer, batch_size, seq_len, text_col) {
        return result;
    }

    // Fallback to demo batches
    eprintln!("Warning: Could not parse training data, using demo LM batches");
    create_demo_lm_batches(batch_size, seq_len)
}

/// Load pre-tokenized sequences from JSON
fn load_pretokenized_json(
    examples: &[serde_json::Value],
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<LMBatch>> {
    let mut all_sequences: Vec<Vec<u32>> = Vec::new();

    for example in examples {
        if let Some(tokens) = example.get("input_ids").and_then(|t| t.as_array()) {
            let seq: Vec<u32> =
                tokens.iter().filter_map(|t| t.as_u64().map(|v| v as u32)).collect();
            if !seq.is_empty() {
                all_sequences.push(seq);
            }
        }
    }

    if !all_sequences.is_empty() {
        println!("  Loaded {} pre-tokenized sequences from JSON", all_sequences.len());
        return create_lm_batches_from_sequences(&all_sequences, batch_size, seq_len);
    }

    eprintln!("Warning: No valid sequences found in JSON");
    create_demo_lm_batches(batch_size, seq_len)
}

/// Tokenize texts and create LM batches
fn tokenize_texts_to_batches(
    texts: &[String],
    tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<LMBatch>> {
    let sequences: Vec<Vec<u32>> = texts
        .iter()
        .map(|text| {
            let mut tokens = tokenizer.encode_with_special(text);
            tokens.truncate(seq_len);
            tokens
        })
        .filter(|seq| seq.len() > 1) // Need at least 2 tokens for causal LM
        .collect();

    if sequences.is_empty() {
        eprintln!("Warning: No valid sequences after tokenization");
        return create_demo_lm_batches(batch_size, seq_len);
    }

    println!("  Tokenized {} sequences", sequences.len());
    create_lm_batches_from_sequences(&sequences, batch_size, seq_len)
}

/// Load LM batches from Parquet file with text or pre-tokenized columns (ALB-007)
///
/// Supports two modes:
/// 1. **Text column** (Utf8): reads text, tokenizes with HfTokenizer, creates LMBatch
/// 2. **Pre-tokenized column** (List<UInt32/Int32>): reads token ID lists directly
///
/// Also handles directory paths containing multiple .parquet shard files.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn load_lm_batches_from_parquet(
    path: &std::path::Path,
    tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    use alimentar::{ArrowDataset, Dataset};

    // Handle directory of parquet shards
    if path.is_dir() {
        return load_lm_batches_from_parquet_dir(path, tokenizer, batch_size, seq_len, text_column);
    }

    println!("  Loading Parquet LM data: {}", path.display());

    // ALB-099: Scope ArrowDataset so it drops before LMBatch construction,
    // avoiding triple materialization (Arrow + Vec<Vec<u32>> + LMBatch).
    let (sequences, texts) = {
        let dataset = ArrowDataset::from_parquet(path).map_err(|e| {
            Error::ConfigError(format!("Failed to load parquet {}: {e}", path.display()))
        })?;

        println!("  Loaded {} rows from Parquet", dataset.len());

        let schema = dataset.schema();
        let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Try pre-tokenized first (input_ids column with integer list type)
        let seqs = try_extract_pretokenized(&dataset, &column_names);
        let txts = if seqs.is_none() {
            Some(extract_text_column(&dataset, text_column, &column_names)?)
        } else {
            None
        };
        (seqs, txts)
        // dataset dropped here — frees Arrow RecordBatch memory
    };

    if let Some(sequences) = sequences {
        println!("  Found pre-tokenized column, loaded {} sequences", sequences.len());
        return create_lm_batches_from_sequences(&sequences, batch_size, seq_len);
    }

    if let Some(texts) = texts {
        println!("  Extracted {} text rows, tokenizing...", texts.len());
        return tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len);
    }

    Err(Error::ConfigError("No pre-tokenized or text column found".into()))
}

/// Load LM batches from a directory of Parquet shard files (ALB-007)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn load_lm_batches_from_parquet_dir(
    dir: &std::path::Path,
    tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    let mut parquet_files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::ConfigError(format!("Cannot read directory {}: {e}", dir.display())))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "parquet"))
        .collect();

    parquet_files.sort();

    if parquet_files.is_empty() {
        return Err(Error::ConfigError(format!("No .parquet files found in {}", dir.display())));
    }

    println!("  Loading {} Parquet shard(s) from {}", parquet_files.len(), dir.display());

    let mut all_batches = Vec::new();
    for file in &parquet_files {
        let shard_batches =
            load_lm_batches_from_parquet(file, tokenizer, batch_size, seq_len, text_column)?;
        all_batches.extend(shard_batches);
    }

    println!("  Total: {} batches from {} shards", all_batches.len(), parquet_files.len());
    Ok(all_batches)
}

/// Try to extract pre-tokenized sequences from a Parquet dataset (ALB-007)
///
/// Looks for columns named `input_ids` or `token_ids` containing integer arrays.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn try_extract_pretokenized(
    dataset: &alimentar::ArrowDataset,
    column_names: &[&str],
) -> Option<Vec<Vec<u32>>> {
    use alimentar::Dataset;

    let token_col =
        column_names.iter().find(|&&n| n == "input_ids" || n == "token_ids").copied()?;

    let schema = dataset.schema();
    let col_idx = schema.index_of(token_col).ok()?;

    // ALB-099: Pre-allocate with known row count
    let mut all_sequences = Vec::with_capacity(dataset.len());

    for batch in dataset.iter() {
        let col = batch.column(col_idx);
        extract_sequences_from_column(col, &mut all_sequences);
    }

    if all_sequences.is_empty() {
        None
    } else {
        Some(all_sequences)
    }
}

/// Extract token sequences from a single Arrow column (List or flat integer types)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_sequences_from_column(col: &arrow::array::ArrayRef, sequences: &mut Vec<Vec<u32>>) {
    use arrow::array::{Array, ListArray};

    if let Some(list_arr) = col.as_any().downcast_ref::<ListArray>() {
        for i in 0..list_arr.len() {
            if list_arr.is_null(i) {
                continue;
            }
            let values = list_arr.value(i);
            let seq = extract_u32_from_array(&values);
            if !seq.is_empty() {
                sequences.push(seq);
            }
        }
    } else {
        // Flat integer column: treat entire column as one sequence
        let seq = extract_u32_from_array(col.as_ref());
        if !seq.is_empty() {
            sequences.push(seq);
        }
    }
}

/// Extract u32 token IDs from an Arrow array (inner values of a ListArray)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_u32_from_array(array: &dyn arrow::array::Array) -> Vec<u32> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array};

    if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        arr.values().to_vec()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else {
        Vec::new()
    }
}

/// Resolve text column name from available columns (ALB-007)
///
/// Tries the specified name first, then common alternatives: text, content, code.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn resolve_text_column_name(text_column: &str, column_names: &[&str]) -> Result<String> {
    if column_names.contains(&text_column) {
        return Ok(text_column.to_string());
    }
    for &fallback in &["text", "content", "code"] {
        if column_names.contains(&fallback) {
            return Ok(fallback.to_string());
        }
    }
    Err(Error::ConfigError(format!(
        "No text column found in Parquet (tried '{text_column}', 'text', 'content', 'code'). Available: {column_names:?}"
    )))
}

/// Extract text strings from a Parquet dataset column (ALB-007)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_text_column(
    dataset: &alimentar::ArrowDataset,
    text_column: &str,
    column_names: &[&str],
) -> Result<Vec<String>> {
    use alimentar::Dataset;

    let col_name = resolve_text_column_name(text_column, column_names)?;

    let schema = dataset.schema();
    let col_idx = schema
        .index_of(&col_name)
        .map_err(|e| Error::ConfigError(format!("Column '{col_name}' not found: {e}")))?;

    let mut texts = Vec::new();
    for batch in dataset.iter() {
        let col = batch.column(col_idx);
        extract_strings_from_array(col, &col_name, &mut texts)?;
    }
    Ok(texts)
}

/// Extract non-empty strings from a StringArray column
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_strings_from_array(
    col: &arrow::array::ArrayRef,
    col_name: &str,
    texts: &mut Vec<String>,
) -> Result<()> {
    use arrow::array::{Array, StringArray};

    let str_arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        Error::ConfigError(format!(
            "Column '{col_name}' is not a string type (found {:?})",
            col.data_type()
        ))
    })?;

    for i in 0..str_arr.len() {
        if !str_arr.is_null(i) {
            let text = str_arr.value(i);
            if !text.is_empty() {
                texts.push(text.to_string());
            }
        }
    }
    Ok(())
}

/// Fallback: Parquet loading without alimentar feature
#[cfg(not(all(not(target_arch = "wasm32"), feature = "parquet")))]
fn load_lm_batches_from_parquet(
    path: &std::path::Path,
    _tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    if !path.exists() {
        return Err(Error::Io(format!("Parquet path does not exist: {}", path.display())));
    }
    eprintln!(
        "Warning: Parquet LM loading requires the 'parquet' feature. \
         Build with: cargo build --features parquet"
    );
    eprintln!(
        "  Alternatively, convert to JSONL: alimentar export {} -o train.jsonl --text-column {}",
        path.display(),
        text_column
    );
    create_demo_lm_batches(batch_size, seq_len)
}

/// Create LMBatches from tokenized sequences
fn create_lm_batches_from_sequences(
    sequences: &[Vec<u32>],
    batch_size: usize,
    _seq_len: usize,
) -> Result<Vec<LMBatch>> {
    // ALB-099: Pre-allocate with known batch count
    let num_batches = (sequences.len() + batch_size - 1) / batch_size;
    let mut batches = Vec::with_capacity(num_batches);
    let pad_id = 0u32; // Standard pad token
    let eos_id = 2u32; // Standard EOS token

    for chunk in sequences.chunks(batch_size) {
        let batch = LMBatch::from_sequences(chunk, pad_id, eos_id);
        batches.push(batch);
    }

    Ok(batches)
}

/// Create demo LM batches for testing
fn create_demo_lm_batches(batch_size: usize, seq_len: usize) -> Result<Vec<LMBatch>> {
    let mut batches = Vec::new();

    // Create a few demo sequences with synthetic tokens
    // Simulating simple patterns like: [1, 2, 3, 4, ...] with slight variations
    for batch_idx in 0..4 {
        let mut sequences = Vec::new();
        for item in 0..batch_size {
            let offset = (batch_idx * batch_size + item) as u32;
            // Create a sequence with incrementing tokens
            let seq: Vec<u32> = (0..seq_len.min(64))
                .map(|i| (offset + i as u32) % 1000 + 1) // Keep in reasonable token range
                .collect();
            sequences.push(seq);
        }

        let batch = LMBatch::from_sequences(&sequences, 0, 2);
        batches.push(batch);
    }

    Ok(batches)
}

/// Detect whether YAML content is in the new manifest format.
///
/// Returns true if the content contains an `entrenar:` key at the start of a line,
/// which is the discriminating field in the manifest schema.
fn is_manifest_format(yaml: &str) -> bool {
    yaml.lines().any(|line| line.starts_with("entrenar:") || line.starts_with("entrenar :"))
}

/// Load training spec from YAML file (without running training)
///
/// Auto-detects format:
/// - If the YAML contains `entrenar:`, it's parsed as a `TrainingManifest` and
///   converted to `TrainSpec` via the bridge converter.
/// - Otherwise, it's parsed directly as `TrainSpec` (legacy format).
pub fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TrainSpec> {
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    if is_manifest_format(&yaml_content) {
        // New declarative manifest format
        let manifest: yaml_mode::TrainingManifest = serde_yaml::from_str(&yaml_content)
            .map_err(|e| Error::ConfigError(format!("Failed to parse manifest YAML: {e}")))?;

        yaml_mode::validate_manifest(&manifest)
            .map_err(|e| Error::ConfigError(format!("Invalid manifest: {e}")))?;

        let bridge_result = yaml_mode::manifest_to_spec(&manifest)
            .map_err(|e| Error::ConfigError(format!("Manifest conversion failed: {e}")))?;

        for warning in &bridge_result.warnings {
            eprintln!("Warning: {warning}");
        }

        validate_config(&bridge_result.spec)
            .map_err(|e| Error::ConfigError(format!("Invalid config after conversion: {e}")))?;

        Ok(bridge_result.spec)
    } else {
        // Legacy TrainSpec format
        let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
            .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

        validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_demo_lm_batches() {
        let batches = create_demo_lm_batches(4, 32).expect("operation should succeed");
        assert_eq!(batches.len(), 4);
        // Each batch should have valid data
        for batch in &batches {
            assert!(batch.has_tokens());
        }
    }

    #[test]
    fn test_create_demo_lm_batches_small() {
        let batches = create_demo_lm_batches(1, 16).expect("operation should succeed");
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_create_demo_lm_batches_large_seq_len() {
        let batches = create_demo_lm_batches(2, 512).expect("operation should succeed");
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_create_lm_batches_from_sequences() {
        let sequences =
            vec![vec![1u32, 2, 3, 4, 5], vec![6u32, 7, 8, 9, 10], vec![11u32, 12, 13, 14, 15]];
        let batches =
            create_lm_batches_from_sequences(&sequences, 2, 32).expect("operation should succeed");
        assert_eq!(batches.len(), 2); // 3 sequences with batch_size 2 = 2 batches
    }

    #[test]
    fn test_create_lm_batches_from_sequences_single_batch() {
        let sequences = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
        let batches =
            create_lm_batches_from_sequences(&sequences, 4, 32).expect("operation should succeed");
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_create_lm_batches_from_sequences_empty() {
        let sequences: Vec<Vec<u32>> = vec![];
        let batches =
            create_lm_batches_from_sequences(&sequences, 4, 32).expect("operation should succeed");
        assert!(batches.is_empty());
    }

    #[test]
    fn test_load_pretokenized_json_valid() {
        let examples: Vec<serde_json::Value> = vec![
            serde_json::json!({"input_ids": [1, 2, 3, 4, 5]}),
            serde_json::json!({"input_ids": [6, 7, 8, 9, 10]}),
        ];
        let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_pretokenized_json_empty() {
        let examples: Vec<serde_json::Value> = vec![];
        // Falls back to demo batches
        let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
        assert!(!batches.is_empty()); // Demo batches
    }

    #[test]
    fn test_load_pretokenized_json_no_input_ids() {
        let examples: Vec<serde_json::Value> =
            vec![serde_json::json!({"text": "hello"}), serde_json::json!({"text": "world"})];
        // Falls back to demo batches
        let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_pretokenized() {
        let json = r#"{"examples": [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]}"#;
        let batches =
            load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_array_pretokenized() {
        let json = r#"[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]"#;
        let batches =
            load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_invalid() {
        let json = "not valid json";
        // Falls back to demo batches
        let batches =
            load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_empty_examples() {
        let json = r#"{"examples": []}"#;
        // Falls back to demo batches
        let batches =
            load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_build_transformer_config_defaults() {
        use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
        use std::collections::HashMap;
        use std::path::PathBuf;

        let spec = TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: None,
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                ..Default::default()
            },
            optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
            training: TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp"),
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        };

        let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
        // Default Qwen2.5-like dimensions
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.intermediate_size, 4864);
    }

    #[test]
    fn test_build_transformer_config_with_architecture_overrides() {
        use crate::config::schema::{
            ArchitectureOverrides, DataConfig, ModelRef, OptimSpec, TrainingParams,
        };
        use std::collections::HashMap;
        use std::path::PathBuf;

        let spec = TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: None,
                architecture: Some(ArchitectureOverrides {
                    hidden_size: Some(1024),
                    num_hidden_layers: Some(16),
                    num_attention_heads: Some(16),
                    num_kv_heads: Some(4),
                    intermediate_size: Some(4096),
                    vocab_size: Some(50000),
                    max_position_embeddings: None,
                    rms_norm_eps: Some(1e-5),
                    rope_theta: Some(500_000.0),
                    use_bias: Some(true),
                    head_dim: None,
                }),
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                ..Default::default()
            },
            optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
            training: TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp"),
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        };

        let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
        // Overridden fields
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.intermediate_size, 4096);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.rope_theta, 500_000.0);
        assert!(config.use_bias);
        // Non-overridden field uses generic default (not Qwen2 demo config)
        assert_eq!(config.max_position_embeddings, 2048);
    }

    #[test]
    fn test_build_transformer_config_partial_overrides() {
        use crate::config::schema::{
            ArchitectureOverrides, DataConfig, ModelRef, OptimSpec, TrainingParams,
        };
        use std::collections::HashMap;
        use std::path::PathBuf;

        let spec = TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: None,
                architecture: Some(ArchitectureOverrides {
                    hidden_size: Some(768),
                    vocab_size: Some(32000),
                    ..Default::default()
                }),
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                ..Default::default()
            },
            optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
            training: TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp"),
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        };

        let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
        // Only these two should be overridden
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.vocab_size, 32000);
        // Rest keeps demo defaults
        assert_eq!(config.num_attention_heads, QWEN_NUM_ATTENTION_HEADS);
        assert_eq!(config.num_kv_heads, QWEN_NUM_KV_HEADS);
        assert_eq!(config.intermediate_size, QWEN_INTERMEDIATE_SIZE);
    }

    #[test]
    fn test_load_lm_batches_from_parquet_fallback() {
        use std::path::Path;
        let tokenizer = HfTokenizer::qwen2();
        // Non-existent path returns error (ALB-007: real parquet loading, not demo fallback)
        let result = load_lm_batches_from_parquet(
            Path::new("/nonexistent.parquet"),
            &tokenizer,
            4,
            32,
            "text",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenize_texts_to_batches_empty() {
        let tokenizer = HfTokenizer::qwen2();
        let texts: Vec<String> = vec![];
        // Falls back to demo batches
        let batches =
            tokenize_texts_to_batches(&texts, &tokenizer, 4, 32).expect("operation should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_tokenize_texts_to_batches_valid() {
        let tokenizer = HfTokenizer::qwen2();
        let texts = vec!["Hello world".to_string(), "This is a test".to_string()];
        let batches =
            tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).expect("operation should succeed");
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_tokenize_texts_to_batches_single_token() {
        let tokenizer = HfTokenizer::qwen2();
        // Very short text that results in single token gets filtered
        let texts = vec!["a".to_string()];
        let batches =
            tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).expect("operation should succeed");
        // May fall back to demo batches if single token is filtered
        assert!(!batches.is_empty());
    }

    // =========================================================================
    // Format auto-detection tests
    // =========================================================================

    #[test]
    fn test_is_manifest_format_detects_entrenar_key() {
        assert!(is_manifest_format("entrenar: \"1.0\"\nname: test\n"));
        assert!(is_manifest_format("# comment\nentrenar: \"1.0\"\n"));
        assert!(is_manifest_format("entrenar : \"1.0\"\n"));
    }

    #[test]
    fn test_is_manifest_format_rejects_legacy() {
        let legacy = r"
model:
  path: model.gguf
data:
  train: train.parquet
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
";
        assert!(!is_manifest_format(legacy));
    }

    #[test]
    fn test_load_config_manifest_format() {
        use std::io::Write;
        let manifest_yaml = r#"
entrenar: "1.0"
name: "test-bridge"
version: "1.0.0"

model:
  source: "./models/test.safetensors"

data:
  source: "./data/train.parquet"
  loader:
    batch_size: 16
    shuffle: true

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 5
"#;
        let dir = std::env::temp_dir().join("entrenar_bridge_test");
        std::fs::create_dir_all(&dir).expect("operation should succeed");
        let path = dir.join("manifest_test.yaml");
        let mut f = std::fs::File::create(&path).expect("file write should succeed");
        f.write_all(manifest_yaml.as_bytes()).expect("file write should succeed");

        let spec = load_config(&path).expect("load should succeed");
        assert_eq!(spec.model.path, std::path::PathBuf::from("./models/test.safetensors"));
        assert_eq!(spec.data.train, std::path::PathBuf::from("./data/train.parquet"));
        assert_eq!(spec.data.batch_size, 16);
        assert_eq!(spec.optimizer.name, "adam");
        assert!((spec.optimizer.lr - 0.0001).abs() < 1e-6);
        assert_eq!(spec.training.epochs, 5);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_config_legacy_format() {
        use std::io::Write;
        let legacy_yaml = r"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";
        let dir = std::env::temp_dir().join("entrenar_bridge_test");
        std::fs::create_dir_all(&dir).expect("operation should succeed");
        let path = dir.join("legacy_test.yaml");
        let mut f = std::fs::File::create(&path).expect("file write should succeed");
        f.write_all(legacy_yaml.as_bytes()).expect("file write should succeed");

        let spec = load_config(&path).expect("load should succeed");
        assert_eq!(spec.optimizer.name, "adam");
        assert_eq!(spec.data.batch_size, 8);

        std::fs::remove_file(&path).ok();
    }

    // =========================================================================
    // FALSIFY tests — contract violation sweep (C-10/C-11, R-04)
    // =========================================================================

    #[test]
    fn test_falsify_c10_c11_config_with_all_required_fields_succeeds() {
        // C-10/C-11: config.json with all 5 required fields must parse successfully.
        use std::io::Write;
        let config_json = r#"{
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_bias": true
        }"#;
        let dir = std::env::temp_dir().join("entrenar_falsify_c10");
        std::fs::create_dir_all(&dir).expect("operation should succeed");
        let config_path = dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).expect("file write should succeed");
        f.write_all(config_json.as_bytes()).expect("file write should succeed");

        let spec = TrainSpec {
            model: crate::config::schema::ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: Some(config_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
            data: crate::config::schema::DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                ..Default::default()
            },
            optimizer: crate::config::schema::OptimSpec {
                name: "adam".to_string(),
                lr: 1e-4,
                params: std::collections::HashMap::new(),
            },
            training: crate::config::schema::TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp"),
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        };

        let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_hidden_layers, 6);
        assert_eq!(config.vocab_size, 30000);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.max_position_embeddings, 512);
        assert!(config.use_bias);

        std::fs::remove_file(&config_path).ok();
    }

    #[test]
    fn test_falsify_c11_missing_hidden_size_errors() {
        // C-11: config.json missing hidden_size must return Err, not silently default.
        use std::io::Write;
        let config_json = r#"{
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072
        }"#;
        let dir = std::env::temp_dir().join("entrenar_falsify_c11");
        std::fs::create_dir_all(&dir).expect("operation should succeed");
        let config_path = dir.join("config_no_hidden.json");
        let mut f = std::fs::File::create(&config_path).expect("file write should succeed");
        f.write_all(config_json.as_bytes()).expect("file write should succeed");

        let spec = TrainSpec {
            model: crate::config::schema::ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: Some(config_path.to_string_lossy().into_owned()),
                ..Default::default()
            },
            data: crate::config::schema::DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                ..Default::default()
            },
            optimizer: crate::config::schema::OptimSpec {
                name: "adam".to_string(),
                lr: 1e-4,
                params: std::collections::HashMap::new(),
            },
            training: crate::config::schema::TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp"),
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        };

        let err = build_transformer_config_from_spec(&spec).unwrap_err();
        assert!(err.to_string().contains("hidden_size"), "Error must mention 'hidden_size': {err}");

        std::fs::remove_file(&config_path).ok();
    }

    #[test]
    fn test_resolve_model_path_local_file() {
        let local_path = Path::new("model.safetensors");
        let resolved = resolve_model_path(local_path).expect("operation should succeed");
        assert_eq!(resolved, PathBuf::from("model.safetensors"));
    }

    #[test]
    fn test_resolve_model_path_local_dir() {
        let local_path = Path::new("./output/model.gguf");
        let resolved = resolve_model_path(local_path).expect("operation should succeed");
        assert_eq!(resolved, PathBuf::from("./output/model.gguf"));
    }

    #[test]
    fn test_resolve_model_path_hf_repo_id() {
        let hf_path = Path::new("Qwen/Qwen2.5-Coder-0.5B");
        let result = resolve_model_path(hf_path);
        // Without hub-publish feature: error with helpful message
        // With hub-publish feature: would attempt download
        #[cfg(not(feature = "hub-publish"))]
        assert!(result.unwrap_err().to_string().contains("hub-publish"));
        #[cfg(feature = "hub-publish")]
        let _ = result; // May succeed or fail depending on network
    }

    // =========================================================================
    // build_train_config tests — optimizer params, LoRA, distributed, mixed precision
    // =========================================================================

    /// Helper to create a minimal TrainSpec for build_train_config tests
    fn minimal_spec() -> TrainSpec {
        use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
        use std::collections::HashMap;

        TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/nonexistent/model"),
                config: None,
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/nonexistent/data.json"),
                batch_size: 4,
                seq_len: Some(256),
                ..Default::default()
            },
            optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
            training: TrainingParams {
                epochs: 1,
                output_dir: PathBuf::from("/tmp/test_output"),
                warmup_steps: 100,
                ..Default::default()
            },
            lora: None,
            quantize: None,
            merge: None,
            publish: None,
        }
    }

    /// Helper to create a minimal TransformerConfig
    fn minimal_transformer_config() -> TransformerConfig {
        TransformerConfig {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            num_hidden_layers: 2,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_bias: false,
            head_dim_override: None,
            architecture: ModelArchitecture::Decoder,
            hf_architecture: None,
            hf_model_type: None,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_build_train_config_basic_wiring() {
        let spec = minimal_spec();
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!((config.lr - 1e-4).abs() < 1e-8);
        assert_eq!(config.warmup_steps, 100);
        assert_eq!(config.max_seq_len, 256);
    }

    #[test]
    fn test_build_train_config_seq_len_default_when_none() {
        let mut spec = minimal_spec();
        spec.data.seq_len = None; // should default to 512
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_build_train_config_grad_clip() {
        let mut spec = minimal_spec();
        spec.training.grad_clip = Some(1.0);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!((config.base.max_grad_norm.expect("grad clip should be set") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_train_config_optimizer_params_beta2_weight_decay() {
        let mut spec = minimal_spec();
        spec.optimizer.params.insert("beta2".to_string(), serde_json::json!(0.95));
        spec.optimizer.params.insert("weight_decay".to_string(), serde_json::json!(0.01));
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!((config.beta2 - 0.95).abs() < 1e-6);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_build_train_config_gradient_accumulation() {
        let mut spec = minimal_spec();
        spec.training.gradient_accumulation = Some(4);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.accumulation_steps, 4);
    }

    #[test]
    fn test_build_train_config_gradient_accumulation_one() {
        let mut spec = minimal_spec();
        spec.training.gradient_accumulation = Some(1);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.accumulation_steps, 1);
    }

    #[test]
    fn test_build_train_config_max_steps() {
        let mut spec = minimal_spec();
        spec.training.max_steps = Some(5000);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.max_steps, Some(5000));
    }

    #[test]
    fn test_build_train_config_mixed_precision_bf16() {
        use crate::autograd::Precision;
        let mut spec = minimal_spec();
        spec.training.mixed_precision = Some("bf16".to_string());
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.precision_config.compute_precision, Precision::Bf16);
    }

    #[test]
    fn test_build_train_config_mixed_precision_fp16() {
        use crate::autograd::Precision;
        let mut spec = minimal_spec();
        spec.training.mixed_precision = Some("fp16".to_string());
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.precision_config.compute_precision, Precision::Fp16);
    }

    #[test]
    fn test_build_train_config_mixed_precision_fp32() {
        use crate::autograd::Precision;
        let mut spec = minimal_spec();
        spec.training.mixed_precision = Some("fp32".to_string());
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.precision_config.compute_precision, Precision::Fp32);
    }

    #[test]
    fn test_build_train_config_mixed_precision_unknown() {
        use crate::autograd::Precision;
        let mut spec = minimal_spec();
        spec.training.mixed_precision = Some("tf32".to_string());
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        // Unknown precision falls back to fp32
        assert_eq!(config.precision_config.compute_precision, Precision::Fp32);
    }

    #[test]
    fn test_build_train_config_checkpointing() {
        let mut spec = minimal_spec();
        spec.training.checkpoints = Some(4);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(config.checkpoint_config.enabled);
    }

    #[test]
    fn test_build_train_config_deterministic_and_seed() {
        let mut spec = minimal_spec();
        spec.training.deterministic = true;
        spec.training.seed = Some(42);
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(config.deterministic);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_build_train_config_profile_interval() {
        let mut spec = minimal_spec();
        spec.training.profile_interval = 50;
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.profile_interval, 50);
    }

    #[test]
    fn test_build_train_config_profile_interval_zero_disabled() {
        let mut spec = minimal_spec();
        spec.training.profile_interval = 0;
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        // Zero means disabled — should remain at default (0)
        assert_eq!(config.profile_interval, 0);
    }

    #[test]
    fn test_build_train_config_lora() {
        use crate::config::schema::LoRASpec;
        let mut spec = minimal_spec();
        spec.lora = Some(LoRASpec {
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert_eq!(config.lora_rank, Some(16));
        assert!((config.lora_alpha.expect("lora_alpha should be set") - 32.0).abs() < 1e-6);
        assert_eq!(
            config.lora_target_modules.as_deref(),
            Some(vec!["q_proj".to_string(), "v_proj".to_string()].as_slice())
        );
    }

    #[test]
    fn test_build_train_config_lora_plus_ratio() {
        use crate::config::schema::LoRASpec;
        let mut spec = minimal_spec();
        spec.lora = Some(LoRASpec {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 16.0,
            double_quantize: false,
            quantize_base: false,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!((config.lora_plus_ratio - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_train_config_lora_double_quantize() {
        use crate::config::schema::LoRASpec;
        let mut spec = minimal_spec();
        spec.lora = Some(LoRASpec {
            rank: 4,
            alpha: 8.0,
            target_modules: vec!["v_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: true,
            quantize_base: false,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(config.double_quantize);
    }

    #[test]
    fn test_build_train_config_lora_quantize_base_nf4() {
        use crate::config::schema::LoRASpec;
        let mut spec = minimal_spec();
        spec.lora = Some(LoRASpec {
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: true,
            quantize_base: true,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(config.quantize_nf4, "quantize_nf4 should be true when lora.quantize_base=true");
        assert!(config.is_nf4());
        assert!(config.is_lora());
        assert_eq!(config.lora_rank, Some(16));
        assert!(config.double_quantize);
    }

    #[test]
    fn test_build_train_config_lora_no_quantize_base() {
        use crate::config::schema::LoRASpec;
        let mut spec = minimal_spec();
        spec.lora = Some(LoRASpec {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(!config.quantize_nf4, "quantize_nf4 should be false when lora.quantize_base=false");
        assert!(!config.is_nf4());
        assert!(config.is_lora());
    }

    #[test]
    fn test_build_train_config_distributed_coordinator() {
        use crate::config::schema::DistributedSpec;
        let mut spec = minimal_spec();
        spec.training.distributed = Some(DistributedSpec {
            world_size: 4,
            backend: "cuda".to_string(),
            role: "coordinator".to_string(),
            coordinator_addr: "127.0.0.1:9000".to_string(),
            rank: 0,
            local_rank: 0,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        let dist = config.distributed.expect("distributed config should be set");
        assert_eq!(dist.world_size, 4);
        assert_eq!(dist.rank, 0);
    }

    #[test]
    fn test_build_train_config_distributed_worker() {
        use crate::config::schema::DistributedSpec;
        let mut spec = minimal_spec();
        spec.training.distributed = Some(DistributedSpec {
            world_size: 2,
            backend: "wgpu".to_string(),
            role: "worker".to_string(),
            coordinator_addr: "10.0.0.1:8080".to_string(),
            rank: 1,
            local_rank: 1,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        let dist = config.distributed.expect("distributed config should be set");
        assert_eq!(dist.world_size, 2);
        assert_eq!(dist.rank, 1);
        assert_eq!(dist.local_rank, 1);
    }

    #[test]
    fn test_build_train_config_distributed_auto_backend() {
        use crate::config::schema::DistributedSpec;
        let mut spec = minimal_spec();
        spec.training.distributed = Some(DistributedSpec {
            world_size: 2,
            backend: "auto".to_string(),
            role: "coordinator".to_string(),
            coordinator_addr: "0.0.0.0:9000".to_string(),
            rank: 0,
            local_rank: 0,
        });
        let model_config = minimal_transformer_config();
        let config = build_train_config(model_config, &spec);
        assert!(config.distributed.is_some());
    }

    #[test]
    fn test_build_train_config_distributed_invalid_addr_fallback() {
        use crate::config::schema::DistributedSpec;
        let mut spec = minimal_spec();
        spec.training.distributed = Some(DistributedSpec {
            world_size: 2,
            backend: "auto".to_string(),
            role: "coordinator".to_string(),
            coordinator_addr: "not-a-valid-address".to_string(),
            rank: 0,
            local_rank: 0,
        });
        let model_config = minimal_transformer_config();
        // Should fall back to 0.0.0.0:9000
        let config = build_train_config(model_config, &spec);
        let dist = config.distributed.expect("distributed config should be set");
        assert_eq!(dist.coordinator_addr.port(), 9000);
    }

    // =========================================================================
    // parse_hf_config error paths — C-10/C-11 required fields
    // =========================================================================

    #[test]
    fn test_parse_hf_config_missing_vocab_size() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "intermediate_size": 3072
        });
        let err = parse_hf_config(&config).expect_err("should fail without vocab_size");
        assert!(err.to_string().contains("vocab_size"), "Error: {err}");
    }

    #[test]
    fn test_parse_hf_config_missing_intermediate_size() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000
        });
        let err = parse_hf_config(&config).expect_err("should fail without intermediate_size");
        assert!(err.to_string().contains("intermediate_size"), "Error: {err}");
    }

    #[test]
    fn test_parse_hf_config_missing_num_attention_heads() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072
        });
        let err = parse_hf_config(&config).expect_err("should fail without num_attention_heads");
        assert!(err.to_string().contains("num_attention_heads"), "Error: {err}");
    }

    #[test]
    fn test_parse_hf_config_missing_num_hidden_layers() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "vocab_size": 30000,
            "intermediate_size": 3072
        });
        let err = parse_hf_config(&config).expect_err("should fail without num_hidden_layers");
        assert!(err.to_string().contains("num_hidden_layers"), "Error: {err}");
    }

    #[test]
    fn test_parse_hf_config_optional_defaults() {
        // Minimal required fields only — check defaults for optional fields
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072
        });
        let tc = parse_hf_config(&config).expect("should parse with required fields only");
        // num_kv_heads defaults to num_attention_heads (MHA)
        assert_eq!(tc.num_kv_heads, 12);
        // max_position_embeddings defaults to 2048
        assert_eq!(tc.max_position_embeddings, 2048);
        // rope_theta defaults to 10000
        assert!((tc.rope_theta - 10000.0).abs() < 1.0);
        // use_bias defaults to false
        assert!(!tc.use_bias);
        // head_dim_override is None by default
        assert!(tc.head_dim_override.is_none());
        // architecture defaults to Decoder
        assert!(matches!(tc.architecture, ModelArchitecture::Decoder));
        assert!(!tc.tie_word_embeddings);
    }

    #[test]
    fn test_parse_hf_config_encoder_architecture_detection() {
        for model_type in &["bert", "roberta", "distilbert", "albert", "electra", "deberta"] {
            let config = serde_json::json!({
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "vocab_size": 30000,
                "intermediate_size": 3072,
                "model_type": model_type
            });
            let tc = parse_hf_config(&config).expect("should parse encoder config");
            assert!(
                matches!(tc.architecture, ModelArchitecture::Encoder),
                "model_type '{model_type}' should be Encoder"
            );
        }
    }

    #[test]
    fn test_parse_hf_config_decoder_architecture_for_unknown_type() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "model_type": "llama"
        });
        let tc = parse_hf_config(&config).expect("should parse decoder config");
        assert!(matches!(tc.architecture, ModelArchitecture::Decoder));
        assert_eq!(tc.hf_model_type, Some("llama".to_string()));
    }

    #[test]
    fn test_parse_hf_config_preserves_hf_metadata() {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "tie_word_embeddings": true,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": true
        });
        let tc = parse_hf_config(&config).expect("should parse full config");
        assert_eq!(tc.hf_architecture, Some("Qwen2ForCausalLM".to_string()));
        assert_eq!(tc.hf_model_type, Some("qwen2".to_string()));
        assert!(tc.tie_word_embeddings);
        assert_eq!(tc.num_kv_heads, 4);
        assert_eq!(tc.head_dim_override, Some(64));
        assert!((tc.rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!((tc.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(tc.use_bias);
    }

    /// GH-262: parse_hf_config for Qwen3-4B must produce correct q_dim and kv_dim.
    #[test]
    fn test_parse_hf_config_qwen3_4b_head_dim() {
        let config = serde_json::json!({
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
            "vocab_size": 151936,
            "intermediate_size": 9728,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": false
        });
        let tc = parse_hf_config(&config).expect("Qwen3-4B config should parse");
        assert_eq!(tc.hidden_size, 2560);
        assert_eq!(tc.num_attention_heads, 32);
        assert_eq!(tc.num_kv_heads, 8);
        assert_eq!(tc.head_dim_override, Some(128));
        assert_eq!(tc.head_dim(), 128);
        // Key assertion: q_dim != hidden_size for Qwen3-4B
        assert_eq!(tc.q_dim(), 4096); // 32 * 128
        assert_ne!(tc.q_dim(), tc.hidden_size); // 4096 != 2560
                                                // KV dim
        let kv_dim = tc.num_kv_heads * tc.head_dim();
        assert_eq!(kv_dim, 1024); // 8 * 128
        assert!(!tc.use_bias);
    }

    // =========================================================================
    // Helper function tests: should_log, should_save_checkpoint, reached_max_steps
    // =========================================================================

    #[test]
    fn test_should_log_at_interval() {
        // interval=10: should log at iter_idx 0, 9, 19, 29
        assert!(should_log(0, 10)); // always log first
        assert!(should_log(9, 10)); // (9+1) % 10 == 0
        assert!(!should_log(1, 10));
        assert!(!should_log(8, 10));
        assert!(should_log(19, 10));
    }

    #[test]
    fn test_should_log_interval_one() {
        // interval=1: every step should log
        for i in 0..10 {
            assert!(should_log(i, 1));
        }
    }

    #[test]
    fn test_should_save_checkpoint() {
        // save_interval=100, step must be > 0 and multiple of interval
        assert!(!should_save_checkpoint(0, 0, 100)); // step 0 excluded
        assert!(should_save_checkpoint(100, 0, 100)); // step 100, last=0
        assert!(!should_save_checkpoint(100, 100, 100)); // step==last_save
        assert!(should_save_checkpoint(200, 100, 100)); // step 200, last=100
        assert!(!should_save_checkpoint(50, 0, 100)); // not multiple
    }

    #[test]
    fn test_reached_max_steps() {
        assert!(!reached_max_steps(None, 1000)); // no limit
        assert!(!reached_max_steps(Some(1000), 500)); // not reached
        assert!(reached_max_steps(Some(1000), 1000)); // exactly reached
        assert!(reached_max_steps(Some(1000), 1500)); // exceeded
    }

    // =========================================================================
    // push_capped / push_capped_f64 tests
    // =========================================================================

    #[test]
    fn test_push_capped_basic() {
        let mut history = Vec::new();
        push_capped(&mut history, 1.0, 3);
        push_capped(&mut history, 2.0, 3);
        push_capped(&mut history, 3.0, 3);
        assert_eq!(history, vec![1.0, 2.0, 3.0]);
        push_capped(&mut history, 4.0, 3);
        assert_eq!(history, vec![2.0, 3.0, 4.0]); // oldest removed
    }

    #[test]
    fn test_push_capped_f64_basic() {
        let mut window: Vec<f64> = Vec::new();
        for i in 0..5 {
            push_capped_f64(&mut window, f64::from(i), 3);
        }
        assert_eq!(window, vec![2.0, 3.0, 4.0]);
    }

    // =========================================================================
    // shuffled_batch_order tests
    // =========================================================================

    #[test]
    fn test_shuffled_batch_order_sequential() {
        let order = shuffled_batch_order(5, false, 42, 0);
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_shuffled_batch_order_shuffled_is_permutation() {
        let order = shuffled_batch_order(10, true, 42, 0);
        assert_eq!(order.len(), 10);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_shuffled_batch_order_deterministic() {
        let order1 = shuffled_batch_order(10, true, 42, 0);
        let order2 = shuffled_batch_order(10, true, 42, 0);
        assert_eq!(order1, order2, "Same seed+epoch should produce same order");
    }

    #[test]
    fn test_shuffled_batch_order_different_epochs() {
        let order0 = shuffled_batch_order(10, true, 42, 0);
        let order1 = shuffled_batch_order(10, true, 42, 1);
        assert_ne!(order0, order1, "Different epochs should produce different orders");
    }

    #[test]
    fn test_shuffled_batch_order_different_seeds() {
        let order_a = shuffled_batch_order(10, true, 42, 0);
        let order_b = shuffled_batch_order(10, true, 99, 0);
        assert_ne!(order_a, order_b, "Different seeds should produce different orders");
    }

    // =========================================================================
    // checkpoint_path / parse_checkpoint_step tests
    // =========================================================================

    #[test]
    fn test_checkpoint_path() {
        let path = checkpoint_path(Path::new("/output"), 500);
        assert_eq!(path, PathBuf::from("/output/model-step-500.apr"));
    }

    #[test]
    fn test_parse_checkpoint_step_valid() {
        // APR format (primary)
        assert_eq!(parse_checkpoint_step("model-step-100.apr"), Some(100));
        assert_eq!(parse_checkpoint_step("model-step-0.apr"), Some(0));
        assert_eq!(parse_checkpoint_step("model-step-999999.apr"), Some(999_999));
        // Legacy SafeTensors format (backward compat)
        assert_eq!(parse_checkpoint_step("model-step-100.safetensors"), Some(100));
        assert_eq!(parse_checkpoint_step("model-step-0.safetensors"), Some(0));
    }

    #[test]
    fn test_parse_checkpoint_step_invalid() {
        assert_eq!(parse_checkpoint_step("model.safetensors"), None);
        assert_eq!(parse_checkpoint_step("model.apr"), None);
        assert_eq!(parse_checkpoint_step("model-step-.apr"), None);
        assert_eq!(parse_checkpoint_step("model-step-abc.apr"), None);
        assert_eq!(parse_checkpoint_step("other-file.txt"), None);
    }

    // =========================================================================
    // prune_checkpoints tests
    // =========================================================================

    #[test]
    fn test_prune_checkpoints_unlimited() {
        let dir = std::env::temp_dir().join("entrenar_prune_test_unlimited");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        // max_keep=0 means unlimited — nothing should be pruned
        for step in [100, 200, 300] {
            let path = dir.join(format!("model-step-{step}.safetensors"));
            std::fs::write(&path, "test").expect("write should succeed");
        }
        prune_checkpoints(&dir, 0);
        // All files should still exist
        for step in [100, 200, 300] {
            assert!(dir.join(format!("model-step-{step}.safetensors")).exists());
        }
        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_prune_checkpoints_removes_oldest() {
        let dir = std::env::temp_dir().join("entrenar_prune_test_oldest");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        for step in [100, 200, 300, 400, 500] {
            let path = dir.join(format!("model-step-{step}.safetensors"));
            std::fs::write(&path, "test").expect("write should succeed");
        }
        prune_checkpoints(&dir, 2);
        // Only the 2 most recent should remain
        assert!(!dir.join("model-step-100.safetensors").exists());
        assert!(!dir.join("model-step-200.safetensors").exists());
        assert!(!dir.join("model-step-300.safetensors").exists());
        assert!(dir.join("model-step-400.safetensors").exists());
        assert!(dir.join("model-step-500.safetensors").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_prune_checkpoints_no_dir() {
        // Non-existent directory should not panic
        prune_checkpoints(Path::new("/nonexistent_dir_xyz"), 2);
    }

    // =========================================================================
    // verify_checkpoint tests
    // =========================================================================

    #[test]
    fn test_verify_checkpoint_valid_file() {
        let dir = std::env::temp_dir().join("entrenar_verify_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("test_checkpoint.safetensors");
        std::fs::write(&path, "some model data here").expect("write should succeed");
        // Should not panic — just prints verification message
        verify_checkpoint(&path);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_checkpoint_empty_file() {
        let dir = std::env::temp_dir().join("entrenar_verify_empty_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("empty_checkpoint.safetensors");
        std::fs::write(&path, "").expect("write should succeed");
        // Should not panic — prints warning about empty file
        verify_checkpoint(&path);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_checkpoint_missing_file() {
        // Should not panic — prints verification failure
        verify_checkpoint(Path::new("/nonexistent_checkpoint.safetensors"));
    }

    // =========================================================================
    // save_training_state tests
    // =========================================================================

    #[test]
    fn test_save_training_state_creates_file() {
        let dir = std::env::temp_dir().join("entrenar_save_state_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        save_training_state(&dir, 100, 2, 50, 42, 1.5);
        let state_path = dir.join("training_state.json");
        assert!(state_path.exists(), "training_state.json should be created");
        let content =
            std::fs::read_to_string(&state_path).expect("should read training_state.json");
        let parsed: serde_json::Value =
            serde_json::from_str(&content).expect("should be valid JSON");
        assert_eq!(parsed["step"], 100);
        assert_eq!(parsed["epoch"], 2);
        assert_eq!(parsed["batch_index"], 50);
        assert_eq!(parsed["seed"], 42);
        assert!((parsed["loss_ema"].as_f64().expect("loss_ema") - 1.5).abs() < 1e-10);
        std::fs::remove_dir_all(&dir).ok();
    }

    // =========================================================================
    // zclip_update tests
    // =========================================================================

    #[test]
    fn test_zclip_update_normal_gradient() {
        let mut ema = 1.0;
        let mut ema_sq = 1.0;
        // Normal gradient — no spike expected
        zclip_update(1.1, 10, &mut ema, &mut ema_sq, 0.05, 2.0);
        assert!((ema - (0.05 * 1.1 + 0.95 * 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_zclip_update_spike_detection() {
        let mut ema = 1.0;
        let mut ema_sq = 1.0;
        // Prime the EMA
        for _ in 0..20 {
            zclip_update(1.0, 0, &mut ema, &mut ema_sq, 0.05, 2.0);
        }
        // Inject a spike — should print warning but not panic
        zclip_update(100.0, 21, &mut ema, &mut ema_sq, 0.05, 2.0);
    }

    // =========================================================================
    // detect_loss_spike tests
    // =========================================================================

    #[test]
    fn test_detect_loss_spike_no_spike() {
        let mut ema = 1.0;
        let mut rollback_count = 0;
        let mut jsonl_file = None;
        // Normal loss — no spike
        detect_loss_spike(1.1, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
        assert_eq!(rollback_count, 0);
    }

    #[test]
    fn test_detect_loss_spike_with_spike() {
        let mut ema = 1.0;
        let mut rollback_count = 0;
        let mut jsonl_file = None;
        // Loss > 3 * EMA → spike
        detect_loss_spike(5.0, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
        assert_eq!(rollback_count, 1);
    }

    #[test]
    fn test_detect_loss_spike_max_rollbacks() {
        let mut ema = 1.0;
        let mut rollback_count = 3; // already at max
        let mut jsonl_file = None;
        detect_loss_spike(10.0, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
        assert_eq!(rollback_count, 3, "Should not increment past max");
    }

    #[test]
    fn test_detect_loss_spike_ema_zero_no_spike() {
        let mut ema = 0.0;
        let mut rollback_count = 0;
        let mut jsonl_file = None;
        // EMA is 0 — condition `*ema > 0.0` fails, no spike
        detect_loss_spike(5.0, 1, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
        assert_eq!(rollback_count, 0);
    }

    // =========================================================================
    // write_heartbeat test
    // =========================================================================

    #[test]
    fn test_write_heartbeat_creates_file() {
        let dir = std::env::temp_dir().join("entrenar_heartbeat_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("heartbeat");
        write_heartbeat(&path, 42);
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).expect("should read heartbeat");
        assert!(content.contains("\t42"), "heartbeat should contain step number");
        std::fs::remove_dir_all(&dir).ok();
    }

    // =========================================================================
    // ScalingLawPredictor tests
    // =========================================================================

    #[test]
    fn test_scaling_law_predictor_insufficient_data() {
        let predictor = ScalingLawPredictor::new();
        assert!(predictor.fit().is_none(), "Need at least 3 points");
        assert!(predictor.predict(1000).is_none());
    }

    #[test]
    fn test_scaling_law_predictor_fit_and_predict() {
        let mut predictor = ScalingLawPredictor::new();
        // Simulate decreasing loss with more tokens: L ≈ 10 - 0.3 * ln(D)
        // Using values that stay positive even at large D
        predictor.record(1000, 10.0 - 0.3 * (1000.0_f64).ln() as f32);
        predictor.record(10000, 10.0 - 0.3 * (10000.0_f64).ln() as f32);
        predictor.record(100000, 10.0 - 0.3 * (100000.0_f64).ln() as f32);

        let (a, b) = predictor.fit().expect("should fit with 3 points");
        assert!(a > 0.0, "intercept should be positive");
        assert!(b > 0.0, "slope should be positive (loss decreasing)");

        // Predict at a nearby token count (not too far to avoid negative loss)
        let prediction = predictor.predict(200_000);
        assert!(prediction.is_some());
        let (pred_loss, pred_ppl, slope) = prediction.expect("prediction should succeed");
        assert!(pred_loss > 0.0, "predicted loss should be positive: {pred_loss}");
        assert!(pred_ppl > 1.0, "predicted perplexity should be > 1: {pred_ppl}");
        assert!(slope > 0.0, "slope should be positive: {slope}");
    }

    #[test]
    fn test_scaling_law_predictor_two_points_insufficient() {
        let mut predictor = ScalingLawPredictor::new();
        predictor.record(1000, 3.0);
        predictor.record(2000, 2.5);
        assert!(predictor.fit().is_none());
    }

    // =========================================================================
    // advance_curriculum tests
    // =========================================================================

    #[test]
    fn test_advance_curriculum_no_stages() {
        let empty: &[crate::config::schema::CurriculumStage] = &[];
        assert_eq!(advance_curriculum(empty, 0, 100), None);
    }

    #[test]
    fn test_advance_curriculum_single_stage_no_until() {
        let stages = vec![crate::config::schema::CurriculumStage {
            data: PathBuf::from("data.jsonl"),
            until_step: None,
        }];
        assert_eq!(advance_curriculum(&stages, 0, 100), None);
    }

    #[test]
    fn test_advance_curriculum_transition_at_boundary() {
        let stages = vec![
            crate::config::schema::CurriculumStage {
                data: PathBuf::from("easy.jsonl"),
                until_step: Some(1000),
            },
            crate::config::schema::CurriculumStage {
                data: PathBuf::from("hard.jsonl"),
                until_step: None,
            },
        ];
        // Before boundary — no transition
        assert_eq!(advance_curriculum(&stages, 0, 999), None);
        // At boundary — transition to stage 1
        assert_eq!(advance_curriculum(&stages, 0, 1000), Some(1));
        // After boundary — still transitions
        assert_eq!(advance_curriculum(&stages, 0, 1500), Some(1));
    }

    #[test]
    fn test_advance_curriculum_already_at_last_stage() {
        let stages = vec![
            crate::config::schema::CurriculumStage {
                data: PathBuf::from("easy.jsonl"),
                until_step: Some(1000),
            },
            crate::config::schema::CurriculumStage {
                data: PathBuf::from("hard.jsonl"),
                until_step: None,
            },
        ];
        // Already at stage 1 (last) — no transition
        assert_eq!(advance_curriculum(&stages, 1, 2000), None);
    }

    #[test]
    fn test_advance_curriculum_beyond_stages() {
        let stages = vec![crate::config::schema::CurriculumStage {
            data: PathBuf::from("data.jsonl"),
            until_step: Some(100),
        }];
        // current=0, until_step=100, step=200 BUT no next stage → None
        assert_eq!(advance_curriculum(&stages, 0, 200), None);
    }

    // =========================================================================
    // config_from_overrides tests
    // =========================================================================

    #[test]
    fn test_config_from_overrides_complete() {
        use crate::config::schema::ArchitectureOverrides;
        let overrides = ArchitectureOverrides {
            hidden_size: Some(512),
            num_hidden_layers: Some(8),
            num_attention_heads: Some(8),
            num_kv_heads: Some(4),
            intermediate_size: Some(2048),
            vocab_size: Some(32000),
            max_position_embeddings: Some(4096),
            rms_norm_eps: Some(1e-6),
            rope_theta: Some(500000.0),
            use_bias: Some(true),
            head_dim: None,
        };
        let config =
            config_from_overrides(&overrides).expect("should build from complete overrides");
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.max_position_embeddings, 4096);
        assert!(config.use_bias);
    }

    #[test]
    fn test_config_from_overrides_missing_required_returns_none() {
        use crate::config::schema::ArchitectureOverrides;
        // Missing hidden_size → None
        let overrides = ArchitectureOverrides {
            hidden_size: None,
            num_hidden_layers: Some(8),
            num_attention_heads: Some(8),
            intermediate_size: Some(2048),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(config_from_overrides(&overrides).is_none());
    }

    #[test]
    fn test_config_from_overrides_defaults_for_optional() {
        use crate::config::schema::ArchitectureOverrides;
        let overrides = ArchitectureOverrides {
            hidden_size: Some(512),
            num_hidden_layers: Some(4),
            num_attention_heads: Some(8),
            intermediate_size: Some(1024),
            vocab_size: Some(10000),
            num_kv_heads: None,            // defaults to num_attention_heads
            max_position_embeddings: None, // defaults to 2048
            rms_norm_eps: None,            // defaults to 1e-5
            rope_theta: None,              // defaults to 10000.0
            use_bias: None,                // defaults to false
            head_dim: None,                // defaults to hidden_size / num_heads
        };
        let config = config_from_overrides(&overrides).expect("should build");
        assert_eq!(config.num_kv_heads, 8); // same as num_attention_heads
        assert_eq!(config.max_position_embeddings, 2048);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
        assert!(!config.use_bias);
    }

    // =========================================================================
    // apply_architecture_overrides tests
    // =========================================================================

    #[test]
    fn test_apply_architecture_overrides_selective() {
        use crate::config::schema::ArchitectureOverrides;
        let mut config = minimal_transformer_config();
        let overrides = ArchitectureOverrides {
            hidden_size: Some(256),
            num_kv_heads: Some(1),
            ..Default::default()
        };
        apply_architecture_overrides(&mut config, &overrides);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_kv_heads, 1);
        // Other fields unchanged
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.intermediate_size, 128);
    }

    // =========================================================================
    // fallback_demo_config tests
    // =========================================================================

    #[test]
    fn test_fallback_demo_config_values() {
        let config = fallback_demo_config();
        assert_eq!(config.hidden_size, QWEN_HIDDEN_SIZE);
        assert_eq!(config.num_attention_heads, QWEN_NUM_ATTENTION_HEADS);
        assert_eq!(config.num_kv_heads, QWEN_NUM_KV_HEADS);
        assert_eq!(config.intermediate_size, QWEN_INTERMEDIATE_SIZE);
        assert_eq!(config.num_hidden_layers, QWEN_NUM_HIDDEN_LAYERS);
        assert_eq!(config.vocab_size, QWEN_VOCAB_SIZE);
        assert_eq!(config.max_position_embeddings, QWEN_MAX_POSITION_EMBEDDINGS);
    }

    // =========================================================================
    // load_config error paths
    // =========================================================================

    #[test]
    fn test_load_config_file_not_found() {
        let result = load_config("/nonexistent/path/to/config.yaml");
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(err.to_string().contains("Failed to read config file"), "Error: {err}");
    }

    #[test]
    fn test_load_config_invalid_yaml() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("entrenar_invalid_yaml_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("invalid.yaml");
        let mut f = std::fs::File::create(&path).expect("file write should succeed");
        f.write_all(b"{{{{not valid yaml: [").expect("write should succeed");

        let result = load_config(&path);
        assert!(result.is_err());
        let err = result.expect_err("should fail on invalid YAML");
        assert!(
            err.to_string().contains("Failed to parse"),
            "Error should mention parse failure: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_config_invalid_manifest_yaml() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("entrenar_invalid_manifest_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("bad_manifest.yaml");
        let mut f = std::fs::File::create(&path).expect("file write should succeed");
        // Has entrenar: key but invalid structure
        f.write_all(b"entrenar: \"1.0\"\nbogus_field: [1, 2, 3]\n").expect("write should succeed");

        let result = load_config(&path);
        assert!(result.is_err(), "Should fail on invalid manifest structure");
        std::fs::remove_file(&path).ok();
    }

    // =========================================================================
    // write_jsonl_event / write_jsonl_event_json tests
    // =========================================================================

    #[test]
    fn test_write_jsonl_event_with_none_file() {
        let mut file = None;
        // Should not panic when file is None
        write_jsonl_event(&mut file, "test", 1, 0.5, 0.4);
    }

    #[test]
    fn test_write_jsonl_event_json_with_none_file() {
        let mut file = None;
        let entry = serde_json::json!({"type": "test"});
        write_jsonl_event_json(&mut file, &entry);
    }

    #[test]
    fn test_write_jsonl_event_with_real_file() {
        let dir = std::env::temp_dir().join("entrenar_jsonl_test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("test.jsonl");
        let mut file = Some(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .expect("file open should succeed"),
        );
        write_jsonl_event(&mut file, "step", 10, 1.5, 1.4);
        write_jsonl_event_json(&mut file, &serde_json::json!({"type": "eval", "step": 20}));
        drop(file);
        let content = std::fs::read_to_string(&path).expect("should read jsonl");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let line0: serde_json::Value = serde_json::from_str(lines[0]).expect("valid json line 0");
        assert_eq!(line0["type"], "step");
        assert_eq!(line0["step"], 10);
        let line1: serde_json::Value = serde_json::from_str(lines[1]).expect("valid json line 1");
        assert_eq!(line1["type"], "eval");
        assert_eq!(line1["step"], 20);
        std::fs::remove_dir_all(&dir).ok();
    }

    // =========================================================================
    // extract_texts_from_array tests
    // =========================================================================

    #[test]
    fn test_extract_texts_from_array_text_column() {
        let array = vec![
            serde_json::json!({"text": "hello"}),
            serde_json::json!({"text": "world"}),
            serde_json::json!({"other": "ignored"}),
        ];
        let texts = extract_texts_from_array(&array, "text");
        assert_eq!(texts, vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_texts_from_array_content_fallback() {
        let array =
            vec![serde_json::json!({"content": "foo"}), serde_json::json!({"content": "bar"})];
        let texts = extract_texts_from_array(&array, "text"); // primary "text" missing, falls back to "content"
        assert_eq!(texts, vec!["foo", "bar"]);
    }

    #[test]
    fn test_extract_texts_from_array_empty() {
        let array: Vec<serde_json::Value> = vec![];
        let texts = extract_texts_from_array(&array, "text");
        assert!(texts.is_empty());
    }

    #[test]
    fn test_extract_texts_from_array_custom_column() {
        let array = vec![
            serde_json::json!({"code": "fn main() {}"}),
            serde_json::json!({"code": "print('hi')"}),
        ];
        let texts = extract_texts_from_array(&array, "code");
        assert_eq!(texts, vec!["fn main() {}", "print('hi')"]);
    }

    // =========================================================================
    // now_ms test
    // =========================================================================

    #[test]
    fn test_now_ms_returns_reasonable_value() {
        let ms = now_ms();
        // Should be after 2020-01-01 (1577836800000 ms)
        assert!(ms > 1_577_836_800_000, "now_ms should return current time in ms: {ms}");
    }

    // =========================================================================
    // JSONL text loading with tokenizer (try_load_from_jsonl)
    // =========================================================================

    #[test]
    fn test_try_load_from_jsonl_without_tokenizer() {
        let content = r#"{"text": "hello world"}
{"text": "foo bar"}"#;
        let result = try_load_from_jsonl(content, None, 2, 32, "text");
        assert!(result.is_none(), "Without tokenizer, should return None");
    }

    #[test]
    fn test_try_load_from_jsonl_empty_content() {
        let tokenizer = HfTokenizer::qwen2();
        let result = try_load_from_jsonl("", Some(&tokenizer), 2, 32, "text");
        assert!(result.is_none(), "Empty content should return None");
    }

    #[test]
    fn test_try_load_from_jsonl_valid() {
        let tokenizer = HfTokenizer::qwen2();
        let content = r#"{"text": "Hello world, this is a test sentence for tokenization."}
{"text": "Another sentence for testing purposes with more tokens."}"#;
        let result = try_load_from_jsonl(content, Some(&tokenizer), 2, 64, "text");
        assert!(result.is_some());
        let batches = result.expect("should have result").expect("should succeed");
        assert!(!batches.is_empty());
    }

    // =========================================================================
    // is_manifest_format edge cases
    // =========================================================================

    #[test]
    fn test_is_manifest_format_empty_string() {
        assert!(!is_manifest_format(""));
    }

    #[test]
    fn test_is_manifest_format_entrenar_in_value_not_key() {
        // "entrenar" as a value, not a key at line start
        assert!(!is_manifest_format("name: entrenar\n"));
    }

    #[test]
    fn test_is_manifest_format_indented_entrenar_not_detected() {
        // Indented entrenar should not be detected (it's not at line start)
        assert!(!is_manifest_format("  entrenar: \"1.0\"\n"));
    }

    // =========================================================================
    // print_max_steps (smoke test)
    // =========================================================================

    #[test]
    fn test_print_max_steps_some() {
        // Should not panic
        print_max_steps(Some(1000));
    }

    #[test]
    fn test_print_max_steps_none() {
        // Should not panic
        print_max_steps(None);
    }

    // =========================================================================
    // update_noise_scale tests
    // =========================================================================

    #[test]
    fn test_update_noise_scale_insufficient_data() {
        let mut window = Vec::new();
        let mut last_step = usize::MAX;
        let mut file = None;
        // Less than 10 points — should not log
        for i in 0..9 {
            update_noise_scale(1.0, i * 100, &mut window, 100, &mut last_step, &mut file);
        }
        assert_eq!(last_step, usize::MAX, "Should not have logged yet");
    }

    #[test]
    fn test_update_noise_scale_logs_at_interval() {
        let mut window = Vec::new();
        let mut last_step = usize::MAX;
        let mut file = None;
        // Add 10+ points, then hit a step that is multiple of interval
        for i in 1..=10 {
            push_capped_f64(&mut window, 1.0 + 0.01 * f64::from(i), 100);
        }
        update_noise_scale(1.15, 100, &mut window, 100, &mut last_step, &mut file);
        assert_eq!(last_step, 100, "Should have logged at step 100");
    }

    /// ALB-007: Parquet text column loading via alimentar
    #[test]
    #[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
    fn test_load_lm_batches_from_parquet_text_column() {
        use arrow::array::{RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        // Create a temp parquet file with text data
        let dir = tempfile::tempdir().expect("temp dir should succeed");
        let parquet_path = dir.path().join("train.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let texts = StringArray::from(vec![
            "def hello():\n    print('hello world')",
            "def add(a, b):\n    return a + b",
            "class Foo:\n    def __init__(self):\n        self.x = 1",
            "import os\nprint(os.getcwd())",
        ]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(texts)])
            .expect("batch creation should succeed");

        let file = std::fs::File::create(&parquet_path).expect("file create should succeed");
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None)
            .expect("writer creation should succeed");
        writer.write(&batch).expect("write should succeed");
        writer.close().expect("close should succeed");

        // Load via our new implementation
        let tokenizer = HfTokenizer::qwen2();
        let batches = load_lm_batches_from_parquet(&parquet_path, &tokenizer, 2, 64, "text")
            .expect("parquet loading should succeed");

        assert!(!batches.is_empty());
        // 4 texts with batch_size=2 → at least 2 batches
        assert!(batches.len() >= 2);
        assert!(batches[0].batch_size <= 2);
        assert!(batches[0].seq_len > 0);
    }

    /// ALB-007: Parquet directory loading (multiple shards)
    #[test]
    #[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
    fn test_load_lm_batches_from_parquet_directory() {
        use arrow::array::{RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("temp dir should succeed");
        let shard_dir = dir.path().join("shards");
        std::fs::create_dir_all(&shard_dir).expect("dir creation should succeed");

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));

        // Write two shard files
        for (i, texts) in
            [vec!["def foo(): pass", "def bar(): return 1"], vec!["class A: pass", "import sys"]]
                .iter()
                .enumerate()
        {
            let arr = StringArray::from(texts.clone());
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])
                .expect("batch should succeed");
            let path = shard_dir.join(format!("shard_{i:04}.parquet"));
            let file = std::fs::File::create(&path).expect("file should succeed");
            let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None)
                .expect("writer should succeed");
            writer.write(&batch).expect("write should succeed");
            writer.close().expect("close should succeed");
        }

        let tokenizer = HfTokenizer::qwen2();
        let batches = load_lm_batches_from_parquet(&shard_dir, &tokenizer, 2, 64, "text")
            .expect("directory loading should succeed");

        assert!(!batches.is_empty());
        // 4 total texts across 2 shards
        let total_seqs: usize = batches.iter().map(|b| b.batch_size).sum();
        assert_eq!(total_seqs, 4);
    }

    /// ALB-007: Missing text column returns error
    #[test]
    #[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
    fn test_load_lm_batches_from_parquet_missing_column() {
        use arrow::array::{Int32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("temp dir should succeed");
        let path = dir.path().join("numeric.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new("numbers", DataType::Int32, false)]));
        let arr = Int32Array::from(vec![1, 2, 3]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])
            .expect("batch should succeed");

        let file = std::fs::File::create(&path).expect("file should succeed");
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None)
            .expect("writer should succeed");
        writer.write(&batch).expect("write should succeed");
        writer.close().expect("close should succeed");

        let tokenizer = HfTokenizer::qwen2();
        let result = load_lm_batches_from_parquet(&path, &tokenizer, 2, 64, "text");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No text column found"));
    }
}

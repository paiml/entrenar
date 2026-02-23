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
use crate::tokenizer::HfTokenizer;
use crate::trace::TRACER;
use crate::train::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{load_safetensors_weights, Architecture, Transformer, TransformerConfig};
use crate::yaml_mode;
use std::fs;
use std::path::{Path, PathBuf};

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

/// Train a transformer model (LLM) from spec
///
/// Uses TransformerTrainer with CausalLMLoss for language modeling.
fn train_transformer_from_spec(spec: &TrainSpec) -> Result<()> {
    println!("✓ Config loaded and validated (Transformer mode)");
    println!("  Model: {}", spec.model.path.display());
    println!(
        "  Optimizer: {} (lr={})",
        spec.optimizer.name, spec.optimizer.lr
    );
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);
    println!("  Training mode: {:?}", spec.training.mode);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
    }
    println!();

    // Build TransformerConfig from spec
    let model_config = build_transformer_config_from_spec(spec)?;

    // Resolve model path (downloads from HF Hub if repo ID)
    let resolved_path = resolve_model_path(&spec.model.path)?;

    // Try to load model weights if path exists (ENT-117)
    let transformer = load_transformer_model(&resolved_path, &model_config)?;

    // Build TransformerTrainConfig
    let mut train_config = TransformerTrainConfig::new(model_config)
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
        train_config = train_config.with_grad_clip(clip);
    }

    if let Some(accum) = spec.training.gradient_accumulation {
        train_config = train_config.with_accumulation_steps(accum);
    }

    // Enable mixed precision if specified
    if let Some(ref precision) = spec.training.mixed_precision {
        match precision.as_str() {
            "bf16" => train_config = train_config.with_bf16(),
            "fp16" => train_config = train_config.with_fp16(),
            "fp32" => {} // fp32 is the default, no action needed
            other => {
                eprintln!("Warning: unknown mixed_precision value '{other}', defaulting to fp32");
            }
        }
    }

    // Create TransformerTrainer (with loaded weights if available)
    let mut trainer = if let Some(loaded_model) = transformer {
        TransformerTrainer::with_model(loaded_model, train_config)
    } else {
        TransformerTrainer::new(train_config)
    };
    println!("✓ TransformerTrainer initialized");
    println!("  Mixed precision: {}", trainer.is_mixed_precision());
    println!("  Checkpointing: {}", trainer.is_checkpointing());
    println!();

    // Load training data as LMBatches (supports tokenizer + text data)
    println!("Loading training data...");
    let batches = load_lm_batches(spec)?;
    println!("✓ {} LM batches created", batches.len());
    println!();

    // Training loop
    println!("Starting transformer training...");
    println!();

    // Enable tracing for overhead analysis
    TRACER.enable();
    TRACER.clear();

    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(&batches);
        let ppl = crate::train::perplexity(avg_loss);
        println!(
            "Epoch {}/{}: loss={:.6}, perplexity={:.2}",
            epoch + 1,
            spec.training.epochs,
            avg_loss,
            ppl
        );
    }

    // Print trace report
    println!("{}", TRACER.report());

    println!();
    println!("✓ Transformer training complete");
    println!(
        "  Final loss: {:.6}",
        trainer.metrics.losses.last().copied().unwrap_or(0.0)
    );
    println!(
        "  Best loss: {:.6}",
        trainer.metrics.best_loss().unwrap_or(0.0)
    );
    println!("  Steps completed: {}", trainer.step());
    println!();

    // Save the trained model
    std::fs::create_dir_all(&spec.training.output_dir).ok();

    // Save model weights to SafeTensors
    let weights_path = spec.training.output_dir.join("model.safetensors");
    println!("Saving model weights to {}...", weights_path.display());
    trainer.save(&weights_path, "rust-cli-docs-model", "Qwen2ForCausalLM")?;
    println!(
        "✓ Model weights saved ({} bytes)",
        std::fs::metadata(&weights_path)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // Save training metadata
    let metadata_path = spec.training.output_dir.join("final_model.json");
    println!("Saving metadata to {}...", metadata_path.display());
    let metadata = serde_json::json!({
        "model_path": spec.model.path,
        "weights_path": weights_path,
        "mode": "transformer",
        "training_mode": format!("{:?}", spec.training.mode),
        "epochs_completed": spec.training.epochs,
        "final_loss": trainer.metrics.losses.last().copied().unwrap_or(0.0),
        "best_loss": trainer.metrics.best_loss().unwrap_or(0.0),
        "steps": trainer.step(),
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
    println!(
        "  Optimizer: {} (lr={})",
        spec.optimizer.name, spec.optimizer.lr
    );
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
        println!(
            "Epoch {}/{}: loss={:.6}",
            epoch + 1,
            spec.training.epochs,
            avg_loss
        );
    }

    println!();
    println!("✓ Training complete");
    println!(
        "  Final loss: {:.6}",
        trainer.metrics.losses.last().copied().unwrap_or(0.0)
    );
    println!(
        "  Best loss: {:.6}",
        trainer.metrics.best_loss().unwrap_or(0.0)
    );
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

/// Load transformer model from SafeTensors weights if available
///
/// # Arguments
///
/// * `model_path` - Path to model directory or SafeTensors file
/// * `config` - Transformer configuration
///
/// # Returns
///
/// Transformer model (with loaded weights or randomly initialized).
fn load_transformer_model(
    model_path: &Path,
    config: &TransformerConfig,
) -> Result<Option<Transformer>> {
    // Check if path exists and is a valid SafeTensors location
    if !model_path.exists() {
        println!("  Model path not found, using random initialization");
        return Ok(None);
    }

    // Try loading SafeTensors weights
    println!("Loading model weights from {}...", model_path.display());

    match load_safetensors_weights(model_path, Architecture::Auto) {
        Ok(weights) => {
            println!("  Found {} weight tensors", weights.len());

            // Try to build transformer from loaded weights
            if let Some(transformer) = Transformer::from_params(config, &weights) {
                println!("✓ Loaded pre-trained weights successfully");
                return Ok(Some(transformer));
            }
            eprintln!("Warning: Weight shapes don't match config, using random initialization");
            Ok(None)
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not load weights from {}: {}",
                model_path.display(),
                e
            );
            eprintln!("Using random initialization instead");
            Ok(None)
        }
    }
}

/// Build TransformerConfig from TrainSpec
///
/// Uses config file if specified, otherwise defaults to a small model.
fn build_transformer_config_from_spec(spec: &TrainSpec) -> Result<TransformerConfig> {
    // Check if config file is specified
    if let Some(config_path) = &spec.model.config {
        let config_file = std::path::Path::new(config_path);
        if config_file.exists() {
            let config_content = std::fs::read_to_string(config_file)
                .map_err(|e| Error::ConfigError(format!("Failed to read model config: {e}")))?;

            if let Ok(hf_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                return parse_hf_config(&hf_config);
            }
        }
    }

    // R-05 (Meyer DbC): Explicit Qwen2-0.5B demo config — NOT a generic default.
    // This path is ONLY for testing without a model. Production callers must provide config.json.
    eprintln!("WARNING: No model config found — using Qwen2-0.5B demo config (NOT suitable for production training)");
    Ok(TransformerConfig {
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
    })
}

/// Parse HuggingFace config.json into `TransformerConfig`.
///
/// C-10/C-11 (Meyer DbC): Required fields (hidden_size, num_attention_heads,
/// num_hidden_layers, vocab_size, intermediate_size) must be present — no silent defaults.
/// R-04: Optional fields use generic defaults with warnings for likely-wrong values.
fn parse_hf_config(hf_config: &serde_json::Value) -> Result<TransformerConfig> {
    let hidden_size = hf_config["hidden_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-11: config.json missing 'hidden_size' — cannot train without model dimensions".into()
        ))? as usize;
    let num_attention_heads = hf_config["num_attention_heads"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-11: config.json missing 'num_attention_heads'".into()
        ))? as usize;
    let num_hidden_layers = hf_config["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-11: config.json missing 'num_hidden_layers'".into()
        ))? as usize;
    let vocab_size = hf_config["vocab_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-10: config.json missing 'vocab_size' — training with wrong vocab corrupts embeddings".into()
        ))? as usize;
    let intermediate_size = hf_config["intermediate_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-11: config.json missing 'intermediate_size'".into()
        ))? as usize;

    // R-04 (Meyer DbC): Optional fields with generic defaults.
    // num_kv_heads → num_attention_heads is the correct GQA→MHA fallback.
    // max_position_embeddings → 2048 is a conservative safe minimum.
    // rms_norm_eps → 1e-6 is the most common default.
    // rope_theta → 10000 is the LLaMA/Mistral standard (WRONG for Qwen at 1M).
    // use_bias → false is correct for most modern architectures.
    let num_kv_heads = hf_config["num_key_value_heads"]
        .as_u64()
        .unwrap_or(num_attention_heads as u64)
        as usize;

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
            eprintln!("Warning: config.json missing 'rope_theta', defaulting to 10000.0 \
                (Qwen models use 1000000.0 — check your config)");
            10_000.0
        }
    };

    let rms_norm_eps = hf_config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let use_bias = hf_config["attention_bias"].as_bool().unwrap_or(false);

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
    let seq_len = spec.data.seq_len.unwrap_or(512);
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

/// Attempt to load LM batches from the training data file
fn try_load_lm_from_file(
    spec: &TrainSpec,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
) -> Option<Result<Vec<LMBatch>>> {
    if !spec.data.train.exists() {
        return None;
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
            e.get(text_col)
                .or_else(|| e.get("content"))
                .and_then(|v| v.as_str())
                .map(String::from)
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

    println!(
        "  Loaded {} text examples from {label}, tokenizing...",
        texts.len()
    );
    Some(tokenize_texts_to_batches(
        &texts, tokenizer, batch_size, seq_len,
    ))
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
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|obj| {
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

    println!(
        "  Loaded {} text examples from JSONL, tokenizing...",
        texts.len()
    );
    Some(tokenize_texts_to_batches(
        &texts, tokenizer, batch_size, seq_len,
    ))
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
        if let Some(result) = try_load_from_array(
            array,
            tokenizer,
            batch_size,
            seq_len,
            text_col,
            "JSON array",
        ) {
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
            let seq: Vec<u32> = tokens
                .iter()
                .filter_map(|t| t.as_u64().map(|v| v as u32))
                .collect();
            if !seq.is_empty() {
                all_sequences.push(seq);
            }
        }
    }

    if !all_sequences.is_empty() {
        println!(
            "  Loaded {} pre-tokenized sequences from JSON",
            all_sequences.len()
        );
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

/// Load LM batches from Parquet file with text column
///
/// Note: For now, parquet with text requires converting to JSON first.
/// Use `alimentar` CLI: `alimentar export input.parquet -o output.jsonl`
fn load_lm_batches_from_parquet(
    path: &std::path::Path,
    _tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    // Parquet text loading requires the alimentar runtime.
    // For now, recommend converting to JSONL first.
    eprintln!(
        "Warning: Parquet text loading requires conversion. \
         Use: alimentar export {} -o train.jsonl --text-column {}",
        path.display(),
        text_column
    );
    eprintln!("Using demo LM batches for now.");
    create_demo_lm_batches(batch_size, seq_len)
}

/// Create LMBatches from tokenized sequences
fn create_lm_batches_from_sequences(
    sequences: &[Vec<u32>],
    batch_size: usize,
    _seq_len: usize,
) -> Result<Vec<LMBatch>> {
    let mut batches = Vec::new();
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
    yaml.lines()
        .any(|line| line.starts_with("entrenar:") || line.starts_with("entrenar :"))
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
        let batches = create_demo_lm_batches(4, 32).unwrap();
        assert_eq!(batches.len(), 4);
        // Each batch should have valid data
        for batch in &batches {
            assert!(!batch.input_ids.is_empty());
        }
    }

    #[test]
    fn test_create_demo_lm_batches_small() {
        let batches = create_demo_lm_batches(1, 16).unwrap();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_create_demo_lm_batches_large_seq_len() {
        let batches = create_demo_lm_batches(2, 512).unwrap();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_create_lm_batches_from_sequences() {
        let sequences = vec![
            vec![1u32, 2, 3, 4, 5],
            vec![6u32, 7, 8, 9, 10],
            vec![11u32, 12, 13, 14, 15],
        ];
        let batches = create_lm_batches_from_sequences(&sequences, 2, 32).unwrap();
        assert_eq!(batches.len(), 2); // 3 sequences with batch_size 2 = 2 batches
    }

    #[test]
    fn test_create_lm_batches_from_sequences_single_batch() {
        let sequences = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
        let batches = create_lm_batches_from_sequences(&sequences, 4, 32).unwrap();
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_create_lm_batches_from_sequences_empty() {
        let sequences: Vec<Vec<u32>> = vec![];
        let batches = create_lm_batches_from_sequences(&sequences, 4, 32).unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_load_pretokenized_json_valid() {
        let examples: Vec<serde_json::Value> = vec![
            serde_json::json!({"input_ids": [1, 2, 3, 4, 5]}),
            serde_json::json!({"input_ids": [6, 7, 8, 9, 10]}),
        ];
        let batches = load_pretokenized_json(&examples, 2, 32).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_pretokenized_json_empty() {
        let examples: Vec<serde_json::Value> = vec![];
        // Falls back to demo batches
        let batches = load_pretokenized_json(&examples, 2, 32).unwrap();
        assert!(!batches.is_empty()); // Demo batches
    }

    #[test]
    fn test_load_pretokenized_json_no_input_ids() {
        let examples: Vec<serde_json::Value> = vec![
            serde_json::json!({"text": "hello"}),
            serde_json::json!({"text": "world"}),
        ];
        // Falls back to demo batches
        let batches = load_pretokenized_json(&examples, 2, 32).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_pretokenized() {
        let json = r#"{"examples": [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]}"#;
        let batches = load_lm_batches_from_json(json, None, 2, 32, None).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_array_pretokenized() {
        let json = r#"[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]"#;
        let batches = load_lm_batches_from_json(json, None, 2, 32, None).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_invalid() {
        let json = "not valid json";
        // Falls back to demo batches
        let batches = load_lm_batches_from_json(json, None, 2, 32, None).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_load_lm_batches_from_json_empty_examples() {
        let json = r#"{"examples": []}"#;
        // Falls back to demo batches
        let batches = load_lm_batches_from_json(json, None, 2, 32, None).unwrap();
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
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 1e-4,
                params: HashMap::new(),
            },
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

        let config = build_transformer_config_from_spec(&spec).unwrap();
        // Default Qwen2.5-like dimensions
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.intermediate_size, 4864);
    }

    #[test]
    fn test_load_lm_batches_from_parquet_fallback() {
        use std::path::Path;
        let tokenizer = HfTokenizer::qwen2();
        // Non-existent path triggers fallback
        let batches = load_lm_batches_from_parquet(
            Path::new("/nonexistent.parquet"),
            &tokenizer,
            4,
            32,
            "text",
        )
        .unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_tokenize_texts_to_batches_empty() {
        let tokenizer = HfTokenizer::qwen2();
        let texts: Vec<String> = vec![];
        // Falls back to demo batches
        let batches = tokenize_texts_to_batches(&texts, &tokenizer, 4, 32).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_tokenize_texts_to_batches_valid() {
        let tokenizer = HfTokenizer::qwen2();
        let texts = vec!["Hello world".to_string(), "This is a test".to_string()];
        let batches = tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).unwrap();
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_tokenize_texts_to_batches_single_token() {
        let tokenizer = HfTokenizer::qwen2();
        // Very short text that results in single token gets filtered
        let texts = vec!["a".to_string()];
        let batches = tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).unwrap();
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
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest_test.yaml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(manifest_yaml.as_bytes()).unwrap();

        let spec = load_config(&path).unwrap();
        assert_eq!(
            spec.model.path,
            std::path::PathBuf::from("./models/test.safetensors")
        );
        assert_eq!(
            spec.data.train,
            std::path::PathBuf::from("./data/train.parquet")
        );
        assert_eq!(spec.data.batch_size, 16);
        assert_eq!(spec.optimizer.name, "adam");
        assert!((spec.optimizer.lr - 0.0001).abs() < 1e-6);
        assert_eq!(spec.training.epochs, 5);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_config_legacy_format() {
        use std::io::Write;
        let legacy_yaml = r#"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
"#;
        let dir = std::env::temp_dir().join("entrenar_bridge_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("legacy_test.yaml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(legacy_yaml.as_bytes()).unwrap();

        let spec = load_config(&path).unwrap();
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
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(config_json.as_bytes()).unwrap();

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

        let config = build_transformer_config_from_spec(&spec).unwrap();
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
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config_no_hidden.json");
        let mut f = std::fs::File::create(&config_path).unwrap();
        f.write_all(config_json.as_bytes()).unwrap();

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
        assert!(
            err.to_string().contains("hidden_size"),
            "Error must mention 'hidden_size': {err}"
        );

        std::fs::remove_file(&config_path).ok();
    }

    #[test]
    fn test_resolve_model_path_local_file() {
        let local_path = Path::new("model.safetensors");
        let resolved = resolve_model_path(local_path).unwrap();
        assert_eq!(resolved, PathBuf::from("model.safetensors"));
    }

    #[test]
    fn test_resolve_model_path_local_dir() {
        let local_path = Path::new("./output/model.gguf");
        let resolved = resolve_model_path(local_path).unwrap();
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
}

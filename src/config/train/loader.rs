//! Main entry points for YAML-based training

use super::batches::load_training_batches;
use crate::config::schema::{ModelMode, TrainSpec};
use crate::config::validate::validate_config;
use crate::error::{Error, Result};
use crate::tokenizer::HfTokenizer;
use crate::trace::TRACER;
use crate::train::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{load_safetensors_weights, Architecture, Transformer, TransformerConfig};
use std::fs;
use std::path::Path;

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
    // Step 1: Load YAML file
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    // Step 2: Parse YAML
    let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

    // Step 3: Validate configuration
    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

    // Step 4: Dispatch based on model mode
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

    // Try to load model weights if path exists (ENT-117)
    let transformer = load_transformer_model(&spec.model.path, &model_config)?;

    // Build TransformerTrainConfig
    let mut train_config = TransformerTrainConfig::new(model_config)
        .with_lr(spec.optimizer.lr)
        .with_warmup_steps(spec.training.warmup_steps)
        .with_max_seq_len(spec.data.seq_len.unwrap_or(512));

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
            _ => {} // fp32 is default
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
        .map_err(|e| Error::ConfigError(format!("Failed to serialize metadata: {}", e)))?;
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
            } else {
                eprintln!("Warning: Weight shapes don't match config, using random initialization");
                Ok(None)
            }
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
                .map_err(|e| Error::ConfigError(format!("Failed to read model config: {}", e)))?;

            // Try parsing as HuggingFace config.json format
            if let Ok(hf_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                return Ok(TransformerConfig {
                    hidden_size: hf_config["hidden_size"].as_u64().unwrap_or(896) as usize,
                    num_attention_heads: hf_config["num_attention_heads"].as_u64().unwrap_or(14)
                        as usize,
                    num_kv_heads: hf_config["num_key_value_heads"].as_u64().unwrap_or(2) as usize,
                    intermediate_size: hf_config["intermediate_size"].as_u64().unwrap_or(4864)
                        as usize,
                    num_hidden_layers: hf_config["num_hidden_layers"].as_u64().unwrap_or(24)
                        as usize,
                    vocab_size: hf_config["vocab_size"].as_u64().unwrap_or(151936) as usize,
                    max_position_embeddings: hf_config["max_position_embeddings"]
                        .as_u64()
                        .unwrap_or(32768) as usize,
                    rms_norm_eps: hf_config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
                    rope_theta: hf_config["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
                    use_bias: hf_config["attention_bias"].as_bool().unwrap_or(false),
                });
            }
        }
    }

    // Default: Use a small demo config for testing
    // Qwen2.5-Coder-0.5B-like dimensions (scaled down)
    eprintln!("Warning: No model config found, using demo transformer config");
    Ok(TransformerConfig {
        hidden_size: 896,
        num_attention_heads: 14,
        num_kv_heads: 2,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        use_bias: false,
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

    // Try to load tokenizer if specified
    let tokenizer = load_tokenizer(spec)?;

    // Check if training data file exists
    if spec.data.train.exists() {
        if let Some(ext) = spec.data.train.extension() {
            if ext == "json" || ext == "jsonl" {
                if let Ok(content) = std::fs::read_to_string(&spec.data.train) {
                    // Try to load data from JSON
                    return load_lm_batches_from_json(
                        &content,
                        tokenizer.as_ref(),
                        batch_size,
                        seq_len,
                        spec.data.input_column.as_deref(),
                    );
                }
            } else if ext == "parquet" {
                // For parquet, we need to extract text column and tokenize
                if let Some(ref tokenizer) = tokenizer {
                    return load_lm_batches_from_parquet(
                        &spec.data.train,
                        tokenizer,
                        batch_size,
                        seq_len,
                        spec.data.input_column.as_deref().unwrap_or("text"),
                    );
                }
            }
        }
    }

    // Fallback: Create demo batches for testing
    eprintln!(
        "Warning: Training data not found at '{}', using demo LM batches",
        spec.data.train.display()
    );
    create_demo_lm_batches(batch_size, seq_len)
}

/// Load HfTokenizer from spec if tokenizer path is specified
fn load_tokenizer(spec: &TrainSpec) -> Result<Option<HfTokenizer>> {
    if let Some(ref tokenizer_path) = spec.data.tokenizer {
        if tokenizer_path.exists() {
            println!("  Loading tokenizer from: {}", tokenizer_path.display());
            let tokenizer = HfTokenizer::from_file(tokenizer_path)
                .map_err(|e| Error::ConfigError(format!("Failed to load tokenizer: {}", e)))?;
            println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());
            return Ok(Some(tokenizer));
        } else {
            eprintln!(
                "Warning: Tokenizer not found at '{}', using default Qwen2 tokenizer",
                tokenizer_path.display()
            );
        }
    }

    // No tokenizer specified - use default for transformer mode
    println!("  Using default Qwen2 tokenizer");
    Ok(Some(HfTokenizer::qwen2()))
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
    // Determine the text column name
    let text_col = input_column.unwrap_or("text");

    // Try parsing as single JSON object or array
    if let Ok(data) = serde_json::from_str::<serde_json::Value>(content) {
        // Check for pre-tokenized format first
        if let Some(examples) = data.get("examples").and_then(|e| e.as_array()) {
            // Check if first example has input_ids (pre-tokenized)
            if examples.first().and_then(|e| e.get("input_ids")).is_some() {
                return load_pretokenized_json(examples, batch_size, seq_len);
            }

            // Otherwise, extract text and tokenize
            if let Some(tokenizer) = tokenizer {
                let texts: Vec<String> = examples
                    .iter()
                    .filter_map(|e| {
                        e.get(text_col)
                            .or_else(|| e.get("content"))
                            .and_then(|v| v.as_str())
                            .map(String::from)
                    })
                    .collect();

                if !texts.is_empty() {
                    println!(
                        "  Loaded {} text examples from JSON, tokenizing...",
                        texts.len()
                    );
                    return tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len);
                }
            }
        }

        // Try as array of objects
        if let Some(array) = data.as_array() {
            // Check for pre-tokenized
            if array.first().and_then(|e| e.get("input_ids")).is_some() {
                return load_pretokenized_json(array, batch_size, seq_len);
            }

            // Extract text and tokenize
            if let Some(tokenizer) = tokenizer {
                let texts: Vec<String> = array
                    .iter()
                    .filter_map(|e| {
                        e.get(text_col)
                            .or_else(|| e.get("content"))
                            .and_then(|v| v.as_str())
                            .map(String::from)
                    })
                    .collect();

                if !texts.is_empty() {
                    println!(
                        "  Loaded {} text examples from JSON array, tokenizing...",
                        texts.len()
                    );
                    return tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len);
                }
            }
        }
    }

    // Try parsing as JSONL (newline-delimited JSON)
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if !lines.is_empty() {
        if let Some(tokenizer) = tokenizer {
            let texts: Vec<String> = lines
                .iter()
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

            if !texts.is_empty() {
                println!(
                    "  Loaded {} text examples from JSONL, tokenizing...",
                    texts.len()
                );
                return tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len);
            }
        }
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

/// Load training spec from YAML file (without running training)
///
/// Useful for testing config parsing and validation separately from training.
pub fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TrainSpec> {
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

    Ok(spec)
}

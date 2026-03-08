//! Merge command implementation

use crate::autograd::Tensor;
use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{MergeArgs, MergeMethod};
use crate::merge::{
    dare_merge, ensemble_merge, slerp_merge, ties_merge, DareConfig, EnsembleConfig, Model,
    SlerpConfig, TiesConfig,
};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub fn run_merge(args: MergeArgs, level: LogLevel) -> Result<(), String> {
    // ENT-LoRA-017: LoRA adapter merge path
    if args.method == MergeMethod::LoraAdapter {
        return run_lora_adapter_merge(&args, level);
    }

    log_merge_start(&args, level);
    validate_model_count(&args)?;

    let models = load_all_models(&args.models, level)?;
    let merged = perform_merge(&models, &args)?;
    export_merged_model(&merged, &args)?;

    log_merge_complete(&merged, &args, level);
    Ok(())
}

/// Log merge operation start
fn log_merge_start(args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merging {} models using {:?}", args.models.len(), args.method),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(level, LogLevel::Verbose, &format!("  Model {}: {}", i + 1, model.display()));
    }
    log(level, LogLevel::Verbose, &format!("  Output: {}", args.output.display()));
}

/// Validate we have enough models
fn validate_model_count(args: &MergeArgs) -> Result<(), String> {
    if args.models.len() < 2 {
        return Err("Need at least 2 models to merge".to_string());
    }
    Ok(())
}

/// Load all models from paths
fn load_all_models(paths: &[std::path::PathBuf], level: LogLevel) -> Result<Vec<Model>, String> {
    let mut models: Vec<Model> = Vec::new();
    for path in paths {
        let model = load_single_model(path)?;
        let tensor_count = model.len();
        models.push(model);

        log(
            level,
            LogLevel::Verbose,
            &format!("  Loaded {} tensors from {}", tensor_count, path.display()),
        );
    }
    Ok(models)
}

/// Load a single model from a SafeTensors file
fn load_single_model(path: &Path) -> Result<Model, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;

    let mut model: Model = HashMap::new();
    for name in tensors.names() {
        if let Some(tensor) = extract_f32_tensor(&tensors, name)? {
            model.insert((*name).to_string(), tensor);
        }
    }
    Ok(model)
}

/// Extract a tensor as f32 values (returns None for non-F32 tensors)
fn extract_f32_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<Option<Tensor>, String> {
    let tensor = tensors.tensor(name).map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

    if tensor.dtype() != safetensors::tensor::Dtype::F32 {
        return Ok(None);
    }

    let bytes = tensor.data();
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Some(Tensor::from_vec(values, false)))
}

/// Perform the merge based on the specified method
fn perform_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    match args.method {
        MergeMethod::Ties => perform_ties_merge(models, args),
        MergeMethod::Dare => perform_dare_merge(models, args),
        MergeMethod::Slerp => perform_slerp_merge(models, args),
        MergeMethod::Average => perform_average_merge(models, args),
        MergeMethod::LoraAdapter => {
            // Handled by early return in run_merge; shouldn't reach here
            Err("LoRA adapter merge uses dedicated path".to_string())
        }
    }
}

/// TIES merge: first model is base, rest are task-specific
fn perform_ties_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = TiesConfig { density: args.density.unwrap_or(0.2) };
    let base = &models[0];
    ties_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("TIES merge failed: {e}"))
}

/// DARE merge with dropout
fn perform_dare_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = DareConfig { drop_prob: 1.0 - args.density.unwrap_or(0.5), seed: None };
    let base = &models[0];
    dare_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("DARE merge failed: {e}"))
}

/// SLERP merge (requires exactly 2 models)
fn perform_slerp_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    if models.len() != 2 {
        return Err("SLERP requires exactly 2 models".to_string());
    }
    let config = SlerpConfig { t: args.weight.unwrap_or(0.5) };
    slerp_merge(&models[0], &models[1], &config).map_err(|e| format!("SLERP merge failed: {e}"))
}

/// Average/ensemble merge with optional weights
fn perform_average_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = build_ensemble_config(args)?;
    ensemble_merge(models, &config).map_err(|e| format!("Average merge failed: {e}"))
}

/// Build ensemble config from args
fn build_ensemble_config(args: &MergeArgs) -> Result<EnsembleConfig, String> {
    if let Some(w_str) = &args.weights {
        let weights: Vec<f32> = w_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid weights: {e}"))?;
        Ok(EnsembleConfig::weighted_average(weights))
    } else {
        Ok(EnsembleConfig::uniform_average())
    }
}

/// Export merged model to file
fn export_merged_model(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_ext = args.output.extension().and_then(|s| s.to_str()).unwrap_or("json");

    if output_ext == "safetensors" {
        export_safetensors(merged, args)
    } else {
        export_json(merged, args)
    }
}

/// Export to SafeTensors format
fn export_safetensors(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    use safetensors::tensor::{Dtype, TensorView};

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = merged
        .iter()
        .map(|(name, tensor)| {
            let data = tensor.data();
            let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap_or(&[])).to_vec();
            let shape = vec![tensor.len()];
            (name.clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .filter_map(|(name, bytes, shape)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).ok().map(|view| (name.as_str(), view))
        })
        .collect();

    let metadata = build_safetensor_metadata(merged, args);
    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("Failed to serialize SafeTensors: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Failed to write output: {e}"))
}

/// Build SafeTensors metadata
fn build_safetensor_metadata(merged: &Model, args: &MergeArgs) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), "merged-model".to_string());
    metadata.insert("merge_method".to_string(), format!("{:?}", args.method));
    metadata.insert("tensor_count".to_string(), merged.len().to_string());
    metadata
}

/// Export to JSON format
fn export_json(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_data: HashMap<String, Vec<f32>> =
        merged.iter().map(|(name, tensor)| (name.clone(), tensor.data().to_vec())).collect();

    let json_data =
        serde_json::to_vec_pretty(&output_data).map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &json_data).map_err(|e| format!("Failed to write output: {e}"))
}

/// Log merge completion
fn log_merge_complete(merged: &Model, args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merge complete: {} tensors written to {}", merged.len(), args.output.display()),
    );
}

/// Merge LoRA adapter into base model (ENT-LoRA-017)
///
/// Computes W_merged = W_base + scale * B @ A for each adapted module,
/// producing a standard safetensors model with no LoRA tensors.
fn run_lora_adapter_merge(args: &MergeArgs, level: LogLevel) -> Result<(), String> {
    let base_path = args.base.as_ref().ok_or("--base required for lora-adapter merge")?;
    let adapter_dir = args.adapter.as_ref().ok_or("--adapter required for lora-adapter merge")?;

    let config_path = adapter_dir.join("adapter_config.json");
    let adapter_path = adapter_dir.join("adapter_model.safetensors");

    if !base_path.exists() {
        return Err(format!("Base model not found: {}", base_path.display()));
    }
    if !config_path.exists() {
        return Err(format!("adapter_config.json not found in {}", adapter_dir.display()));
    }
    if !adapter_path.exists() {
        return Err(format!("adapter_model.safetensors not found in {}", adapter_dir.display()));
    }

    log(level, LogLevel::Normal, "LoRA adapter merge:");
    log(level, LogLevel::Normal, &format!("  Base: {}", base_path.display()));
    log(level, LogLevel::Normal, &format!("  Adapter: {}", adapter_dir.display()));

    // Read adapter config
    let config_str =
        std::fs::read_to_string(&config_path).map_err(|e| format!("Read adapter config: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_str).map_err(|e| format!("Parse adapter config: {e}"))?;

    let rank = config.get("r").and_then(serde_json::Value::as_u64).unwrap_or(8) as usize;
    let alpha =
        config.get("lora_alpha").and_then(serde_json::Value::as_f64).unwrap_or(rank as f64 * 2.0);
    let scale = alpha as f32 / rank as f32;

    log(level, LogLevel::Normal, &format!("  Rank: {rank}, Alpha: {alpha}, Scale: {scale:.4}"));

    // Load base model
    let base_data = std::fs::read(base_path).map_err(|e| format!("Read base model: {e}"))?;
    let base_tensors =
        SafeTensors::deserialize(&base_data).map_err(|e| format!("Parse base model: {e}"))?;

    // Load adapter
    let adapter_data = std::fs::read(&adapter_path).map_err(|e| format!("Read adapter: {e}"))?;
    let adapter_tensors =
        SafeTensors::deserialize(&adapter_data).map_err(|e| format!("Parse adapter: {e}"))?;

    // Merge: copy all base tensors, apply LoRA delta where adapters exist
    let adapter_names: Vec<String> =
        adapter_tensors.names().iter().map(|s| (*s).to_string()).collect();
    let base_names: Vec<String> = base_tensors.names().iter().map(|s| (*s).to_string()).collect();

    // Build map of adapter A/B pairs grouped by module path
    let lora_pairs = build_lora_pairs(&adapter_names, &adapter_tensors)?;
    let mut merged_count = 0usize;

    // Prepare output tensors
    let mut output_tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for name in &base_names {
        let base_t = base_tensors.tensor(name).map_err(|e| format!("Get tensor {name}: {e}"))?;
        let shape: Vec<usize> = base_t.shape().to_vec();

        // Check if this weight has a LoRA adapter
        if let Some((a_data, b_data, a_shape, b_shape)) = lora_pairs.get(name.as_str()) {
            // W_merged = W_base + scale * B @ A
            let base_f32 = bytes_to_f32(base_t.data(), base_t.dtype());
            let a_f32 = bytes_to_f32(a_data, safetensors::tensor::Dtype::F32);
            let b_f32 = bytes_to_f32(b_data, safetensors::tensor::Dtype::F32);

            let d_out = b_shape[0];
            let r = b_shape[1];
            let d_in = a_shape[1];

            // Compute B @ A: [d_out, r] @ [r, d_in] -> [d_out, d_in]
            let mut ba = vec![0.0f32; d_out * d_in];
            for i in 0..d_out {
                for j in 0..d_in {
                    let mut sum = 0.0f32;
                    for k in 0..r {
                        sum += b_f32[i * r + k] * a_f32[k * d_in + j];
                    }
                    ba[i * d_in + j] = sum;
                }
            }

            // W_merged = W_base + scale * BA
            let mut merged: Vec<f32> = base_f32;
            for (i, val) in merged.iter_mut().enumerate() {
                *val += scale * ba[i];
            }

            let bytes: Vec<u8> = bytemuck::cast_slice(&merged).to_vec();
            output_tensors.push((name.clone(), bytes, shape));
            merged_count += 1;
        } else {
            // Pass through base tensor unchanged
            output_tensors.push((name.clone(), base_t.data().to_vec(), shape));
        }
    }

    // Serialize to safetensors
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = output_tensors
        .iter()
        .filter_map(|(name, bytes, shape)| {
            safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                shape.clone(),
                bytes,
            )
            .ok()
            .map(|view| (name.as_str(), view))
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "entrenar-merged-lora".to_string());
    metadata.insert("lora_rank".to_string(), rank.to_string());
    metadata.insert("lora_alpha".to_string(), format!("{alpha}"));

    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("Serialize merged model: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Write merged model: {e}"))?;

    let output_size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
    log(
        level,
        LogLevel::Normal,
        &format!("  Merged {merged_count} adapter weights into base model"),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Output: {} ({:.2} MB)", args.output.display(), output_size as f64 / 1e6),
    );

    Ok(())
}

/// Build a map of base weight name -> (A_data, B_data, A_shape, B_shape)
fn build_lora_pairs<'a>(
    names: &[String],
    tensors: &'a SafeTensors<'a>,
) -> Result<HashMap<&'a str, (Vec<u8>, Vec<u8>, Vec<usize>, Vec<usize>)>, String> {
    let mut pairs: HashMap<String, (Option<(Vec<u8>, Vec<usize>)>, Option<(Vec<u8>, Vec<usize>)>)> =
        HashMap::new();

    for name in names {
        // PEFT naming: base_model.model.{path}.lora_A.weight / lora_B.weight
        let (base_name, is_a) = if let Some(stripped) = name.strip_suffix(".lora_A.weight") {
            (stripped.replace("base_model.model.", "") + ".weight", true)
        } else if let Some(stripped) = name.strip_suffix(".lora_B.weight") {
            (stripped.replace("base_model.model.", "") + ".weight", false)
        } else {
            continue;
        };

        let tensor = tensors.tensor(name).map_err(|e| format!("Get adapter tensor {name}: {e}"))?;
        let data = tensor.data().to_vec();
        let shape = tensor.shape().to_vec();

        let entry = pairs.entry(base_name).or_insert((None, None));
        if is_a {
            entry.0 = Some((data, shape));
        } else {
            entry.1 = Some((data, shape));
        }
    }

    let mut result = HashMap::new();
    for (base_name, (a, b)) in &pairs {
        if let (Some((a_data, a_shape)), Some((b_data, b_shape))) = (a, b) {
            // Leak the base_name string to get a &'a str — safe in this context
            // as the result lives only for the merge duration
            let key: &str = Box::leak(base_name.clone().into_boxed_str());
            result.insert(key, (a_data.clone(), b_data.clone(), a_shape.clone(), b_shape.clone()));
        }
    }
    Ok(result)
}

/// Convert tensor bytes to f32 based on dtype
fn bytes_to_f32(data: &[u8], dtype: safetensors::tensor::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::tensor::Dtype::F32 => {
            data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        safetensors::tensor::Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        safetensors::tensor::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        _ => {
            // For other dtypes, treat as f32 (best effort)
            data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    }
}

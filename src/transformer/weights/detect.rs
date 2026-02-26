//! Architecture detection and SafeTensors file discovery

use super::Architecture;
use crate::error::Result;

/// Find SafeTensors files in a directory or return single file
pub(crate) fn find_safetensors_files(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        // Single file path
        if path.extension().is_some_and(|e| e == "safetensors") {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        // Directory - find all .safetensors files
        // Check for model.safetensors first
        let single = path.join("model.safetensors");
        if single.exists() {
            files.push(single);
        } else {
            // Look for sharded files: model-00001-of-00005.safetensors
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().is_some_and(|e| e == "safetensors") {
                        files.push(p);
                    }
                }
                // Sort for consistent ordering
                files.sort();
            }
        }
    }

    Ok(files)
}

/// Auto-detect model architecture from tensor names
pub(crate) fn detect_architecture(tensors: &safetensors::SafeTensors<'_>) -> Architecture {
    let names: Vec<String> = tensors.names().iter().map(std::string::ToString::to_string).collect();

    // Qwen2 uses "model.layers.X.self_attn.q_proj" with biases
    // LLaMA uses same pattern but no biases
    // Check for bias tensors to distinguish
    let has_attn_bias = names.iter().any(|n: &String| {
        n.contains("self_attn.q_proj.bias") || n.contains("self_attn.k_proj.bias")
    });

    if has_attn_bias {
        // Qwen2 has attention biases
        return Architecture::Qwen2;
    }

    // Check for Mistral-specific patterns
    // Mistral uses sliding window attention config but same weight names as LLaMA
    // We default to LLaMA if no Qwen2 bias markers found

    Architecture::Llama
}

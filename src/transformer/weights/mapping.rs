//! Weight name mapping between architectures

use super::Architecture;

/// Map RoBERTa/CodeBERT weight names to entrenar encoder convention (ENC-006).
fn map_roberta_weight_name(name: &str) -> String {
    // Strip roberta. or bert. prefix
    let stripped =
        name.strip_prefix("roberta.").or_else(|| name.strip_prefix("bert.")).unwrap_or(name);

    // Embeddings
    if stripped == "embeddings.word_embeddings.weight" {
        return "encoder.embed_tokens.weight".to_string();
    }
    if stripped == "embeddings.position_embeddings.weight" {
        return "encoder.position_embeddings.weight".to_string();
    }
    if stripped == "embeddings.token_type_embeddings.weight" {
        return "encoder.token_type_embeddings.weight".to_string();
    }
    if stripped == "embeddings.LayerNorm.weight" {
        return "encoder.embeddings_layernorm.weight".to_string();
    }
    if stripped == "embeddings.LayerNorm.bias" {
        return "encoder.embeddings_layernorm.bias".to_string();
    }

    // Encoder layers: encoder.layer.{i}.XXX
    if let Some(rest) = stripped.strip_prefix("encoder.layer.") {
        if let Some((num, layer_rest)) = rest.split_once('.') {
            let mapped = layer_rest
                .replace("attention.self.query", "self_attn.q_proj")
                .replace("attention.self.key", "self_attn.k_proj")
                .replace("attention.self.value", "self_attn.v_proj")
                .replace("attention.output.dense", "self_attn.o_proj")
                .replace("attention.output.LayerNorm", "input_layernorm")
                .replace("intermediate.dense", "mlp.intermediate.dense")
                .replace("output.dense", "mlp.output.dense")
                .replace("output.LayerNorm", "post_attention_layernorm");

            return format!("encoder.layers.{num}.{mapped}");
        }
    }

    // Pooler (optional)
    if stripped.starts_with("pooler.") {
        return format!("encoder.{stripped}");
    }

    // Pass through anything else
    name.to_string()
}

/// Map weight name from source architecture to standard LLaMA convention
///
/// Standard names expected by `Transformer::from_params`:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.self_attn.q_proj.weight`
/// - `model.layers.{i}.self_attn.k_proj.weight`
/// - `model.layers.{i}.self_attn.v_proj.weight`
/// - `model.layers.{i}.self_attn.o_proj.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.layers.{i}.mlp.gate_proj.weight`
/// - `model.layers.{i}.mlp.up_proj.weight`
/// - `model.layers.{i}.mlp.down_proj.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
pub(crate) fn map_weight_name(name: &str, arch: Architecture) -> String {
    match arch {
        Architecture::Qwen2 => {
            // Qwen2 weight names are nearly identical to LLaMA convention
            // Main differences:
            // - Qwen2 has bias tensors (which we preserve)
            // - Some Qwen2 models use "attn" instead of "self_attn" (rare)

            // Handle potential "attn" -> "self_attn" mapping
            if name.contains(".attn.") && !name.contains(".self_attn.") {
                name.replace(".attn.", ".self_attn.")
            } else {
                name.to_string()
            }
        }
        Architecture::Mistral | Architecture::Llama => {
            // LLaMA and Mistral use same naming convention as our standard
            name.to_string()
        }
        Architecture::RoBERTa => map_roberta_weight_name(name),
        Architecture::Auto => {
            // Should not reach here after detection
            name.to_string()
        }
    }
}

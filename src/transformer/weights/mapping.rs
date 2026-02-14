//! Weight name mapping between architectures

use super::Architecture;

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
        Architecture::Auto => {
            // Should not reach here after detection
            name.to_string()
        }
    }
}

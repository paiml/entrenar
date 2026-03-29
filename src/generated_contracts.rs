// Auto-generated contract assertions from YAML — DO NOT EDIT.
// Zero cost in release builds (debug_assert!).
// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs
// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;

// Auto-generated from contracts/backward-pass-v1.yaml — DO NOT EDIT
// Contract: backward-pass-v1

/// Preconditions for equation `chain_rule`.
/// Domain-specific. Call: `contract_pre_chain_rule!(slice_expr)`
macro_rules! contract_pre_chain_rule {
    () => {{}};
    ($input:expr) => {{
        let grad_output = &$input;
        debug_assert!(!grad_output.is_empty(),
            "Contract chain_rule: precondition violated — !grad_output.is_empty()");
    }};
}

/// Postconditions for equation `chain_rule`.
/// Call before return: `contract_post_chain_rule!(result_expr)`
macro_rules! contract_post_chain_rule {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == input.len(), "Contract chain_rule: postcondition violated — ret.len() == input.len()");
        debug_assert!(ret.iter().all(|g| g.is_finite()), "Contract chain_rule: postcondition violated — ret.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `chain_rule`.
macro_rules! contract_chain_rule {
    ($input:expr, $body:expr) => {{
        contract_pre_chain_rule!($input);
        let _contract_result = $body;
        contract_post_chain_rule!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `matmul_backward`.
/// Domain-specific. Call: `contract_pre_matmul_backward!(slice_expr)`
macro_rules! contract_pre_matmul_backward {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `matmul_backward`.
/// Call before return: `contract_post_matmul_backward!(result_expr)`
macro_rules! contract_post_matmul_backward {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(grad_a.len() == a.len(), "Contract matmul_backward: postcondition violated — grad_a.len() == a.len()");
        debug_assert!(grad_b.len() == b.len(), "Contract matmul_backward: postcondition violated — grad_b.len() == b.len()");
        debug_assert!(grad_a.iter().all(|g| g.is_finite()), "Contract matmul_backward: postcondition violated — grad_a.iter().all(|g| g.is_finite())");
        debug_assert!(grad_b.iter().all(|g| g.is_finite()), "Contract matmul_backward: postcondition violated — grad_b.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `matmul_backward`.
macro_rules! contract_matmul_backward {
    ($input:expr, $body:expr) => {{
        contract_pre_matmul_backward!($input);
        let _contract_result = $body;
        contract_post_matmul_backward!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `softmax_backward`.
/// Domain-specific. Call: `contract_pre_softmax_backward!(slice_expr)`
macro_rules! contract_pre_softmax_backward {
    () => {{}};
    ($input:expr) => {{
        let y = &$input;
        debug_assert!(!y.is_empty(),
            "Contract softmax_backward: precondition violated — !y.is_empty()");
        debug_assert!(y.iter().all(|v| *v >= 0.0 && v.is_finite()),
            "Contract softmax_backward: precondition violated — y.iter().all(|v| *v >= 0.0 && v.is_finite())");
        debug_assert!((y.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "Contract softmax_backward: precondition violated — (y.iter().sum::<f32>() - 1.0).abs() < 1e-5");
    }};
}

/// Postconditions for equation `softmax_backward`.
/// Call before return: `contract_post_softmax_backward!(result_expr)`
macro_rules! contract_post_softmax_backward {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == y.len(), "Contract softmax_backward: postcondition violated — ret.len() == y.len()");
        debug_assert!(ret.iter().all(|g| g.is_finite()), "Contract softmax_backward: postcondition violated — ret.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `softmax_backward`.
macro_rules! contract_softmax_backward {
    ($input:expr, $body:expr) => {{
        contract_pre_softmax_backward!($input);
        let _contract_result = $body;
        contract_post_softmax_backward!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/batch-v1.yaml — DO NOT EDIT
// Contract: batch-v1

/// Preconditions for equation `batch_partition`.
/// Call at function entry: `contract_pre_batch_partition!(input_expr)`
macro_rules! contract_pre_batch_partition {
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(n > 0,
            "Contract batch_partition: precondition violated — n > 0");
        debug_assert!(batch_size > 0,
            "Contract batch_partition: precondition violated — batch_size > 0");
    }};
}

/// Postconditions for equation `batch_partition`.
/// Call before return: `contract_post_batch_partition!(result_expr)`
macro_rules! contract_post_batch_partition {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret >= 1, "Contract batch_partition: postcondition violated — ret >= 1");
        debug_assert!(ret == (n + batch_size - 1) / batch_size, "Contract batch_partition: postcondition violated — ret == (n + batch_size - 1) / batch_size");
    }};
}

/// Combined pre+post contract for equation `batch_partition`.
macro_rules! contract_batch_partition {
    ($input:expr, $body:expr) => {{
        contract_pre_batch_partition!($input);
        let _contract_result = $body;
        contract_post_batch_partition!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `gradient_scaling`.
/// Domain-specific. Call: `contract_pre_gradient_scaling!(slice_expr)`
macro_rules! contract_pre_gradient_scaling {
    () => {{}};
    ($input:expr) => {{
        let gradients = &$input;
        debug_assert!(!gradients.is_empty(),
            "Contract gradient_scaling: precondition violated — !gradients.is_empty()");
    }};
}

/// Postconditions for equation `gradient_scaling`.
/// Call before return: `contract_post_gradient_scaling!(result_expr)`
macro_rules! contract_post_gradient_scaling {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == gradients[0].len(), "Contract gradient_scaling: postcondition violated — ret.len() == gradients[0].len()");
        debug_assert!(ret.iter().all(|g| g.is_finite()), "Contract gradient_scaling: postcondition violated — ret.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `gradient_scaling`.
macro_rules! contract_gradient_scaling {
    ($input:expr, $body:expr) => {{
        contract_pre_gradient_scaling!($input);
        let _contract_result = $body;
        contract_post_gradient_scaling!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/checkpoint-v1.yaml — DO NOT EDIT
// Contract: checkpoint-v1

/// Preconditions for equation `checkpoint_memory`.
/// Call at function entry: `contract_pre_checkpoint_memory!(input_expr)`
macro_rules! contract_pre_checkpoint_memory {
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(num_layers > 0,
            "Contract checkpoint_memory: precondition violated — num_layers > 0");
        debug_assert!(activation_size > 0,
            "Contract checkpoint_memory: precondition violated — activation_size > 0");
    }};
}

/// Postconditions for equation `checkpoint_memory`.
/// Call before return: `contract_post_checkpoint_memory!(result_expr)`
macro_rules! contract_post_checkpoint_memory {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(checkpoint_memory <= (num_layers as f64).sqrt().ceil() as usize * activation_size, "Contract checkpoint_memory: postcondition violated — checkpoint_memory <= (num_layers as f64).sqrt().ceil() as usize * activation_size");
    }};
}

/// Combined pre+post contract for equation `checkpoint_memory`.
macro_rules! contract_checkpoint_memory {
    ($input:expr, $body:expr) => {{
        contract_pre_checkpoint_memory!($input);
        let _contract_result = $body;
        contract_post_checkpoint_memory!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `save_restore_identity`.
/// Domain-specific. Call: `contract_pre_save_restore_identity!(slice_expr)`
macro_rules! contract_pre_save_restore_identity {
    () => {{}};
    ($input:expr) => {{
        let model_state = &$input;
    }};
}

/// Postconditions for equation `save_restore_identity`.
/// Call before return: `contract_post_save_restore_identity!(result_expr)`
macro_rules! contract_post_save_restore_identity {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(restored.weights == model_state.weights, "Contract save_restore_identity: postcondition violated — restored.weights == model_state.weights");
        debug_assert!(restored.epoch == model_state.epoch, "Contract save_restore_identity: postcondition violated — restored.epoch == model_state.epoch");
        debug_assert!(restored.optimizer_state == model_state.optimizer_state, "Contract save_restore_identity: postcondition violated — restored.optimizer_state == model_state.optimizer_state");
    }};
}

/// Combined pre+post contract for equation `save_restore_identity`.
macro_rules! contract_save_restore_identity {
    ($input:expr, $body:expr) => {{
        contract_pre_save_restore_identity!($input);
        let _contract_result = $body;
        contract_post_save_restore_identity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/gemm-v1.yaml — DO NOT EDIT
// Contract: gemm-v1

/// Preconditions for equation `bf16_precision`.
/// Domain-specific. Call: `contract_pre_bf16_precision!(slice_expr)`
macro_rules! contract_pre_bf16_precision {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
        debug_assert!(!a.is_empty(),
            "Contract bf16_precision: precondition violated — !a.is_empty()");
    }};
}

/// Postconditions for equation `bf16_precision`.
/// Call before return: `contract_post_bf16_precision!(result_expr)`
macro_rules! contract_post_bf16_precision {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Contract bf16_precision: postcondition violated — ret.iter().all(|v| v.is_finite())");
        debug_assert!(ret.len() == m * n, "Contract bf16_precision: postcondition violated — ret.len() == m * n");
    }};
}

/// Combined pre+post contract for equation `bf16_precision`.
macro_rules! contract_bf16_precision {
    ($input:expr, $body:expr) => {{
        contract_pre_bf16_precision!($input);
        let _contract_result = $body;
        contract_post_bf16_precision!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `gemm`.
/// Domain-specific. Call: `contract_pre_gemm!(slice_expr)`
macro_rules! contract_pre_gemm {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `gemm`.
/// Call before return: `contract_post_gemm!(result_expr)`
macro_rules! contract_post_gemm {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(c.len() == m * n, "Contract gemm: postcondition violated — c.len() == m * n");
        debug_assert!(c.iter().all(|v| v.is_finite()), "Contract gemm: postcondition violated — c.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `gemm`.
macro_rules! contract_gemm {
    ($input:expr, $body:expr) => {{
        contract_pre_gemm!($input);
        let _contract_result = $body;
        contract_post_gemm!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `gemm_backward`.
/// Domain-specific. Call: `contract_pre_gemm_backward!(slice_expr)`
macro_rules! contract_pre_gemm_backward {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `gemm_backward`.
/// Call before return: `contract_post_gemm_backward!(result_expr)`
macro_rules! contract_post_gemm_backward {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(grad_a.len() == m * k, "Contract gemm_backward: postcondition violated — grad_a.len() == m * k");
        debug_assert!(grad_b.len() == k * n, "Contract gemm_backward: postcondition violated — grad_b.len() == k * n");
        debug_assert!(grad_a.iter().all(|v| v.is_finite()), "Contract gemm_backward: postcondition violated — grad_a.iter().all(|v| v.is_finite())");
        debug_assert!(grad_b.iter().all(|v| v.is_finite()), "Contract gemm_backward: postcondition violated — grad_b.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `gemm_backward`.
macro_rules! contract_gemm_backward {
    ($input:expr, $body:expr) => {{
        contract_pre_gemm_backward!($input);
        let _contract_result = $body;
        contract_post_gemm_backward!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/lora-v1.yaml — DO NOT EDIT
// Contract: lora-v1

/// Preconditions for equation `lora_decomposition`.
/// Domain-specific. Call: `contract_pre_lora_decomposition!(slice_expr)`
macro_rules! contract_pre_lora_decomposition {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `lora_decomposition`.
/// Call before return: `contract_post_lora_decomposition!(result_expr)`
macro_rules! contract_post_lora_decomposition {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == d * k, "Contract lora_decomposition: postcondition violated — ret.len() == d * k");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Contract lora_decomposition: postcondition violated — ret.iter().all(|v| v.is_finite())");
        debug_assert!(w_frozen_after == w_frozen_before, "Contract lora_decomposition: postcondition violated — w_frozen_after == w_frozen_before");
    }};
}

/// Combined pre+post contract for equation `lora_decomposition`.
macro_rules! contract_lora_decomposition {
    ($input:expr, $body:expr) => {{
        contract_pre_lora_decomposition!($input);
        let _contract_result = $body;
        contract_post_lora_decomposition!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/matmul-v1.yaml — DO NOT EDIT
// Contract: matmul-v1

/// Preconditions for equation `backward`.
/// Domain-specific. Call: `contract_pre_backward!(slice_expr)`
macro_rules! contract_pre_backward {
    () => {{}};
    ($input:expr) => {{
        let grad_output = &$input;
    }};
}

/// Postconditions for equation `backward`.
/// Call before return: `contract_post_backward!(result_expr)`
macro_rules! contract_post_backward {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(grad_a.len() == m * k, "Contract backward: postcondition violated — grad_a.len() == m * k");
        debug_assert!(grad_b.len() == k * n, "Contract backward: postcondition violated — grad_b.len() == k * n");
        debug_assert!(grad_a.iter().all(|g| g.is_finite()), "Contract backward: postcondition violated — grad_a.iter().all(|g| g.is_finite())");
        debug_assert!(grad_b.iter().all(|g| g.is_finite()), "Contract backward: postcondition violated — grad_b.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `backward`.
macro_rules! contract_backward {
    ($input:expr, $body:expr) => {{
        contract_pre_backward!($input);
        let _contract_result = $body;
        contract_post_backward!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `matmul`.
/// Domain-specific. Call: `contract_pre_matmul!(slice_expr)`
macro_rules! contract_pre_matmul {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `matmul`.
/// Call before return: `contract_post_matmul!(result_expr)`
macro_rules! contract_post_matmul {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == m * n, "Contract matmul: postcondition violated — ret.len() == m * n");
        debug_assert!(ret.data().iter().all(|v| v.is_finite()), "Contract matmul: postcondition violated — ret.data().iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `matmul`.
macro_rules! contract_matmul {
    ($input:expr, $body:expr) => {{
        contract_pre_matmul!($input);
        let _contract_result = $body;
        contract_post_matmul!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `matmul_nt`.
/// Domain-specific. Call: `contract_pre_matmul_nt!(slice_expr)`
macro_rules! contract_pre_matmul_nt {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `matmul_nt`.
/// Call before return: `contract_post_matmul_nt!(result_expr)`
macro_rules! contract_post_matmul_nt {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == m * n, "Contract matmul_nt: postcondition violated — ret.len() == m * n");
        debug_assert!(ret.data().iter().all(|v| v.is_finite()), "Contract matmul_nt: postcondition violated — ret.data().iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `matmul_nt`.
macro_rules! contract_matmul_nt {
    ($input:expr, $body:expr) => {{
        contract_pre_matmul_nt!($input);
        let _contract_result = $body;
        contract_post_matmul_nt!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/optimizer-v1.yaml — DO NOT EDIT
// Contract: optimizer-v1

/// Preconditions for equation `adamw_update`.
/// Domain-specific. Call: `contract_pre_adamw_update!(slice_expr)`
macro_rules! contract_pre_adamw_update {
    () => {{}};
    ($input:expr) => {{
        let x = &$input;
    }};
}

/// Postconditions for equation `adamw_update`.
/// Call before return: `contract_post_adamw_update!(result_expr)`
macro_rules! contract_post_adamw_update {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(params.iter().all(|p| p.is_finite()), "Contract adamw_update: postcondition violated — params.iter().all(|p| p.is_finite())");
        debug_assert!(params.len() == params_before.len(), "Contract adamw_update: postcondition violated — params.len() == params_before.len()");
    }};
}

/// Combined pre+post contract for equation `adamw_update`.
macro_rules! contract_adamw_update {
    ($input:expr, $body:expr) => {{
        contract_pre_adamw_update!($input);
        let _contract_result = $body;
        contract_post_adamw_update!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `convergence`.
/// Domain-specific. Call: `contract_pre_convergence!(slice_expr)`
macro_rules! contract_pre_convergence {
    () => {{}};
    ($input:expr) => {{
        let x = &$input;
    }};
}

/// Postconditions for equation `convergence`.
/// Call before return: `contract_post_convergence!(result_expr)`
macro_rules! contract_post_convergence {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(loss_final.is_finite(), "Contract convergence: postcondition violated — loss_final.is_finite()");
        debug_assert!(loss_final <= loss_initial, "Contract convergence: postcondition violated — loss_final <= loss_initial");
    }};
}

/// Combined pre+post contract for equation `convergence`.
macro_rules! contract_convergence {
    ($input:expr, $body:expr) => {{
        contract_pre_convergence!($input);
        let _contract_result = $body;
        contract_post_convergence!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `sgd_momentum_update`.
/// Domain-specific. Call: `contract_pre_sgd_momentum_update!(slice_expr)`
macro_rules! contract_pre_sgd_momentum_update {
    () => {{}};
    ($input:expr) => {{
        let grad = &$input;
    }};
}

/// Postconditions for equation `sgd_momentum_update`.
/// Call before return: `contract_post_sgd_momentum_update!(result_expr)`
macro_rules! contract_post_sgd_momentum_update {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(params.iter().all(|p| p.is_finite()), "Contract sgd_momentum_update: postcondition violated — params.iter().all(|p| p.is_finite())");
        debug_assert!(velocity.len() == params.len(), "Contract sgd_momentum_update: postcondition violated — velocity.len() == params.len()");
    }};
}

/// Combined pre+post contract for equation `sgd_momentum_update`.
macro_rules! contract_sgd_momentum_update {
    ($input:expr, $body:expr) => {{
        contract_pre_sgd_momentum_update!($input);
        let _contract_result = $body;
        contract_post_sgd_momentum_update!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/quantization-v1.yaml — DO NOT EDIT
// Contract: quantization-v1

/// Preconditions for equation `compression_ratio`.
/// Domain-specific. Call: `contract_pre_compression_ratio!(slice_expr)`
macro_rules! contract_pre_compression_ratio {
    () => {{}};
    ($input:expr) => {{
        let self = &$input;
        debug_assert!(self.len > 0,
            "Contract compression_ratio: precondition violated — self.len > 0");
    }};
}

/// Postconditions for equation `compression_ratio`.
/// Call before return: `contract_post_compression_ratio!(result_expr)`
macro_rules! contract_post_compression_ratio {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret > 1.0, "Contract compression_ratio: postcondition violated — ret > 1.0");
        debug_assert!(ret.is_finite(), "Contract compression_ratio: postcondition violated — ret.is_finite()");
    }};
}

/// Combined pre+post contract for equation `compression_ratio`.
macro_rules! contract_compression_ratio {
    ($input:expr, $body:expr) => {{
        contract_pre_compression_ratio!($input);
        let _contract_result = $body;
        contract_post_compression_ratio!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `quantization_error`.
/// Domain-specific. Call: `contract_pre_quantization_error!(slice_expr)`
macro_rules! contract_pre_quantization_error {
    () => {{}};
    ($input:expr) => {{
        let x = &$input;
        debug_assert!(!x.is_empty(),
            "Contract quantization_error: precondition violated — !x.is_empty()");
    }};
}

/// Postconditions for equation `quantization_error`.
/// Call before return: `contract_post_quantization_error!(result_expr)`
macro_rules! contract_post_quantization_error {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(mse.is_finite(), "Contract quantization_error: postcondition violated — mse.is_finite()");
        debug_assert!(mse >= 0.0, "Contract quantization_error: postcondition violated — mse >= 0.0");
        debug_assert!(mse <= (scale / 7.0).powi(2) / 4.0, "Contract quantization_error: postcondition violated — mse <= (scale / 7.0).powi(2) / 4.0");
    }};
}

/// Combined pre+post contract for equation `quantization_error`.
macro_rules! contract_quantization_error {
    ($input:expr, $body:expr) => {{
        contract_pre_quantization_error!($input);
        let _contract_result = $body;
        contract_post_quantization_error!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `symmetric_4bit`.
/// Domain-specific. Call: `contract_pre_symmetric_4bit!(slice_expr)`
macro_rules! contract_pre_symmetric_4bit {
    () => {{}};
    ($input:expr) => {{
        let input = &$input;
        debug_assert!(!input.is_empty(),
            "Contract symmetric_4bit: precondition violated — !input.is_empty()");
        debug_assert!(input.iter().all(|v| v.is_finite()),
            "Contract symmetric_4bit: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `symmetric_4bit`.
/// Call before return: `contract_post_symmetric_4bit!(result_expr)`
macro_rules! contract_post_symmetric_4bit {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(quantized.iter().all(|q| *q >= -7 && *q <= 7), "Contract symmetric_4bit: postcondition violated — quantized.iter().all(|q| *q >= -7 && *q <= 7)");
        debug_assert!(scale >= 0.0, "Contract symmetric_4bit: postcondition violated — scale >= 0.0");
    }};
}

/// Combined pre+post contract for equation `symmetric_4bit`.
macro_rules! contract_symmetric_4bit {
    ($input:expr, $body:expr) => {{
        contract_pre_symmetric_4bit!($input);
        let _contract_result = $body;
        contract_post_symmetric_4bit!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/softmax-v1.yaml — DO NOT EDIT
// Contract: softmax-v1

/// Preconditions for equation `log_sum_exp`.
/// Domain-specific. Call: `contract_pre_log_sum_exp!(slice_expr)`
macro_rules! contract_pre_log_sum_exp {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `log_sum_exp`.
/// Call before return: `contract_post_log_sum_exp!(result_expr)`
macro_rules! contract_post_log_sum_exp {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.is_finite(), "Contract log_sum_exp: postcondition violated — ret.is_finite()");
    }};
}

/// Combined pre+post contract for equation `log_sum_exp`.
macro_rules! contract_log_sum_exp {
    ($input:expr, $body:expr) => {{
        contract_pre_log_sum_exp!($input);
        let _contract_result = $body;
        contract_post_log_sum_exp!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `softmax`.
/// Domain-specific. Call: `contract_pre_softmax!(slice_expr)`
macro_rules! contract_pre_softmax {
    () => {{}};
    ($input:expr) => {{
        let a = &$input;
    }};
}

/// Postconditions for equation `softmax`.
/// Call before return: `contract_post_softmax!(result_expr)`
macro_rules! contract_post_softmax {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(ret.len() == a.len(), "Contract softmax: postcondition violated — ret.len() == a.len()");
        debug_assert!(ret.data().iter().all(|v| *v >= 0.0), "Contract softmax: postcondition violated — ret.data().iter().all(|v| *v >= 0.0)");
        debug_assert!((ret.data().iter().copied().sum::<f32>() - 1.0).abs() < 1e-6, "Contract softmax: postcondition violated — (ret.data().iter().copied().sum::<f32>() - 1.0).abs() < 1e-6");
    }};
}

/// Combined pre+post contract for equation `softmax`.
macro_rules! contract_softmax {
    ($input:expr, $body:expr) => {{
        contract_pre_softmax!($input);
        let _contract_result = $body;
        contract_post_softmax!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/tokenizer-v1.yaml — DO NOT EDIT
// Contract: tokenizer-v1

/// Preconditions for equation `bpe_merge`.
/// Call at function entry: `contract_pre_bpe_merge!(input_expr)`
macro_rules! contract_pre_bpe_merge {
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(!corpus.is_empty(),
            "Contract bpe_merge: precondition violated — !corpus.is_empty()");
        debug_assert!(k > 0,
            "Contract bpe_merge: precondition violated — k > 0");
        debug_assert!(k <= vocab_size - base_vocab_size,
            "Contract bpe_merge: precondition violated — k <= vocab_size - base_vocab_size");
    }};
}

/// Postconditions for equation `bpe_merge`.
/// Call before return: `contract_post_bpe_merge!(result_expr)`
macro_rules! contract_post_bpe_merge {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(vocab.len() == base_vocab_size + k, "Contract bpe_merge: postcondition violated — vocab.len() == base_vocab_size + k");
        debug_assert!(total_tokens_after <= total_tokens_before, "Contract bpe_merge: postcondition violated — total_tokens_after <= total_tokens_before");
    }};
}

/// Combined pre+post contract for equation `bpe_merge`.
macro_rules! contract_bpe_merge {
    ($input:expr, $body:expr) => {{
        contract_pre_bpe_merge!($input);
        let _contract_result = $body;
        contract_post_bpe_merge!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `encode_decode_identity`.
/// Domain-specific. Call: `contract_pre_encode_decode_identity!(slice_expr)`
macro_rules! contract_pre_encode_decode_identity {
    () => {{}};
    ($input:expr) => {{
        let tokenizer = &$input;
        debug_assert!(tokenizer.is_trained(),
            "Contract encode_decode_identity: precondition violated — tokenizer.is_trained()");
    }};
}

/// Postconditions for equation `encode_decode_identity`.
/// Call before return: `contract_post_encode_decode_identity!(result_expr)`
macro_rules! contract_post_encode_decode_identity {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(!ret.is_empty(), "Contract encode_decode_identity: postcondition violated — !ret.is_empty()");
        debug_assert!(ret.iter().all(|id| *id < vocab_size), "Contract encode_decode_identity: postcondition violated — ret.iter().all(|id| *id < vocab_size)");
    }};
}

/// Combined pre+post contract for equation `encode_decode_identity`.
macro_rules! contract_encode_decode_identity {
    ($input:expr, $body:expr) => {{
        contract_pre_encode_decode_identity!($input);
        let _contract_result = $body;
        contract_post_encode_decode_identity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/training-loop-v1.yaml — DO NOT EDIT
// Contract: training-loop-v1

/// Preconditions for equation `gradient_norm`.
/// Domain-specific. Call: `contract_pre_gradient_norm!(slice_expr)`
macro_rules! contract_pre_gradient_norm {
    () => {{}};
    ($input:expr) => {{
        let gradients = &$input;
        debug_assert!(!gradients.is_empty(),
            "Contract gradient_norm: precondition violated — !gradients.is_empty()");
    }};
}

/// Postconditions for equation `gradient_norm`.
/// Call before return: `contract_post_gradient_norm!(result_expr)`
macro_rules! contract_post_gradient_norm {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(grad_norm <= clip_value, "Contract gradient_norm: postcondition violated — grad_norm <= clip_value");
        debug_assert!(gradients.iter().all(|g| g.is_finite()), "Contract gradient_norm: postcondition violated — gradients.iter().all(|g| g.is_finite())");
    }};
}

/// Combined pre+post contract for equation `gradient_norm`.
macro_rules! contract_gradient_norm {
    ($input:expr, $body:expr) => {{
        contract_pre_gradient_norm!($input);
        let _contract_result = $body;
        contract_post_gradient_norm!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `loss_decrease`.
/// Domain-specific. Call: `contract_pre_loss_decrease!(slice_expr)`
macro_rules! contract_pre_loss_decrease {
    () => {{}};
    ($input:expr) => {{
        let x = &$input;
    }};
}

/// Postconditions for equation `loss_decrease`.
/// Call before return: `contract_post_loss_decrease!(result_expr)`
macro_rules! contract_post_loss_decrease {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(loss.is_finite(), "Contract loss_decrease: postcondition violated — loss.is_finite()");
        debug_assert!(!loss.is_nan(), "Contract loss_decrease: postcondition violated — !loss.is_nan()");
    }};
}

/// Combined pre+post contract for equation `loss_decrease`.
macro_rules! contract_loss_decrease {
    ($input:expr, $body:expr) => {{
        contract_pre_loss_decrease!($input);
        let _contract_result = $body;
        contract_post_loss_decrease!(_contract_result);
        _contract_result
    }};
}

// Total: 19 preconditions, 60 postconditions from 11 contracts

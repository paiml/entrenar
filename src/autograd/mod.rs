//! Tape-based autograd engine
//!
//! Provides automatic differentiation using a computational graph with gradient tape.
//!
//! ## Gradient Checkpointing
//!
//! For memory-efficient training of large models, use the `checkpoint` module:
//!
//! ```ignore
//! use entrenar::autograd::checkpoint::{checkpoint, CheckpointConfig};
//!
//! let output = checkpoint(|x| layer.forward(x), &input);
//! ```

mod backward;
pub mod checkpoint;
mod context;
mod ops;
pub mod precision;
mod tensor;

#[cfg(test)]
mod tests;

pub use backward::BackwardOp;
pub use checkpoint::{
    checkpoint, checkpoint_if, estimate_memory_savings, optimal_checkpoints, CheckpointConfig,
    CheckpointManager, CheckpointedSegment,
};
pub use context::Context;
pub use ops::*;
pub use precision::{
    bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_to_f32, GradScaler, MixedPrecisionConfig, Precision,
};
pub use tensor::Tensor;

/// Perform backward pass on a tensor
pub fn backward(tensor: &mut Tensor, grad_output: Option<ndarray::Array1<f32>>) {
    if let Some(grad) = grad_output {
        tensor.set_grad(grad);
    } else {
        // Initialize with ones for scalar loss
        let ones = ndarray::Array1::ones(tensor.data().len());
        tensor.set_grad(ones);
    }

    if let Some(op) = tensor.backward_op() {
        op.backward();
    }
}

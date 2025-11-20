//! Tape-based autograd engine
//!
//! Provides automatic differentiation using a computational graph with gradient tape.

mod backward;
mod context;
mod ops;
mod tensor;

#[cfg(test)]
mod tests;

pub use backward::BackwardOp;
pub use context::Context;
pub use ops::*;
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

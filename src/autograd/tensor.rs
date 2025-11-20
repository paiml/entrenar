//! Tensor type with gradient tracking

use super::BackwardOp;
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Tensor with automatic differentiation support
#[derive(Clone)]
pub struct Tensor {
    data: Array1<f32>,
    grad: Rc<RefCell<Option<Array1<f32>>>>,
    backward_op: Option<Rc<dyn BackwardOp>>,
    requires_grad: bool,
}

impl Tensor {
    /// Create a new tensor with data
    pub fn new(data: Array1<f32>, requires_grad: bool) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            backward_op: None,
            requires_grad,
        }
    }

    /// Create a tensor from a vector
    pub fn from_vec(data: Vec<f32>, requires_grad: bool) -> Self {
        Self::new(Array1::from(data), requires_grad)
    }

    /// Create a tensor filled with zeros
    pub fn zeros(size: usize, requires_grad: bool) -> Self {
        Self::new(Array1::zeros(size), requires_grad)
    }

    /// Create a tensor filled with ones
    pub fn ones(size: usize, requires_grad: bool) -> Self {
        Self::new(Array1::ones(size), requires_grad)
    }

    /// Get reference to data
    pub fn data(&self) -> &Array1<f32> {
        &self.data
    }

    /// Get mutable reference to data
    pub fn data_mut(&mut self) -> &mut Array1<f32> {
        &mut self.data
    }

    /// Get gradient (if computed)
    pub fn grad(&self) -> Option<Array1<f32>> {
        self.grad.borrow().clone()
    }

    /// Set gradient
    pub fn set_grad(&self, grad: Array1<f32>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    /// Accumulate gradient (for when tensor is used multiple times)
    pub fn accumulate_grad(&self, grad: Array1<f32>) {
        let mut grad_ref = self.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_mut() {
            *existing = &*existing + &grad;
        } else {
            *grad_ref = Some(grad);
        }
    }

    /// Zero out gradient
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Check if requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get reference to gradient cell (for backward operations)
    pub fn grad_cell(&self) -> Rc<RefCell<Option<Array1<f32>>>> {
        self.grad.clone()
    }

    /// Set backward operation
    pub fn set_backward_op(&mut self, op: Rc<dyn BackwardOp>) {
        self.backward_op = Some(op);
    }

    /// Get backward operation
    pub fn backward_op(&self) -> Option<Rc<dyn BackwardOp>> {
        self.backward_op.clone()
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("grad", &self.grad.borrow())
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

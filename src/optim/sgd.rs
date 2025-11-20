//! Stochastic Gradient Descent optimizer

use super::Optimizer;
use crate::Tensor;
use ndarray::Array1;

/// SGD optimizer with optional momentum
pub struct SGD {
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<Array1<f32>>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocities: Vec::new(),
        }
    }

    /// Initialize velocities if needed
    fn ensure_velocities(&mut self, params: &[Tensor]) {
        if self.velocities.is_empty() {
            self.velocities = params.iter().map(|_| None).collect();
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Tensor]) {
        self.ensure_velocities(params);

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Use SIMD for large tensors (>= 16 elements for meaningful speedup)
                if grad.len() >= 16 {
                    let grad_slice = grad.as_slice().unwrap();
                    let param_slice = param.data_mut().as_slice_mut().unwrap();

                    if self.momentum > 0.0 {
                        // Initialize velocity if needed
                        if self.velocities[i].is_none() {
                            self.velocities[i] = Some(ndarray::Array1::zeros(grad.len()));
                        }

                        let velocity = self.velocities[i].as_mut().unwrap();
                        let velocity_slice = velocity.as_slice_mut().unwrap();

                        // v = momentum * v - lr * grad (using SIMD)
                        // First scale velocity by momentum
                        for v in velocity_slice.iter_mut() {
                            *v *= self.momentum;
                        }

                        // Then add -lr * grad using SIMD axpy
                        super::simd::simd_axpy(-self.lr, grad_slice, velocity_slice);

                        // param = param + velocity (using SIMD axpy with a=1.0)
                        super::simd::simd_axpy(1.0, velocity_slice, param_slice);
                    } else {
                        // Simple SGD: param -= lr * grad (using SIMD axpy)
                        super::simd::simd_axpy(-self.lr, grad_slice, param_slice);
                    }
                } else {
                    // Fallback to scalar implementation for small tensors
                    if self.momentum > 0.0 {
                        // v = momentum * v - lr * grad
                        let velocity = if let Some(v) = &self.velocities[i] {
                            v * self.momentum - &grad * self.lr
                        } else {
                            &grad * (-self.lr)
                        };

                        *param.data_mut() = param.data() + &velocity;
                        self.velocities[i] = Some(velocity);
                    } else {
                        // Simple SGD: param -= lr * grad
                        *param.data_mut() = param.data() - &(&grad * self.lr);
                    }
                }
            }
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

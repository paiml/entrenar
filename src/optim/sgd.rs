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

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

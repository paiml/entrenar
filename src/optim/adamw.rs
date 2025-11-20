//! AdamW optimizer (Adam with decoupled Weight decay)

use super::Optimizer;
use crate::Tensor;
use ndarray::Array1;

/// AdamW optimizer
///
/// AdamW decouples weight decay from the gradient-based update, making it more
/// effective than L2 regularization. Instead of adding weight decay to the gradient,
/// it applies weight decay directly to the parameters.
///
/// Standard Adam with L2: θ_t = θ_{t-1} - lr * (m_t / (√v_t + ε) + λ * θ_{t-1})
/// AdamW: θ_t = (1 - lr * λ) * θ_{t-1} - lr * m_t / (√v_t + ε)
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: u64,
    m: Vec<Option<Array1<f32>>>, // First moment
    v: Vec<Option<Array1<f32>>>, // Second moment
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Create AdamW with default parameters (weight_decay = 0.01)
    pub fn default_params(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Initialize moments if needed
    fn ensure_moments(&mut self, params: &[Tensor]) {
        if self.m.is_empty() {
            self.m = params.iter().map(|_| None).collect();
            self.v = params.iter().map(|_| None).collect();
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [Tensor]) {
        self.ensure_moments(params);
        self.t += 1;

        // Bias correction factors
        let lr_t = self.lr
            * ((1.0 - self.beta2.powi(self.t as i32)).sqrt()
                / (1.0 - self.beta1.powi(self.t as i32)));

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // m_t = β1 * m_{t-1} + (1 - β1) * g
                let m_t = if let Some(m) = &self.m[i] {
                    m * self.beta1 + &grad * (1.0 - self.beta1)
                } else {
                    &grad * (1.0 - self.beta1)
                };

                // v_t = β2 * v_{t-1} + (1 - β2) * g²
                let grad_sq = &grad * &grad;
                let v_t = if let Some(v) = &self.v[i] {
                    v * self.beta2 + &grad_sq * (1.0 - self.beta2)
                } else {
                    &grad_sq * (1.0 - self.beta2)
                };

                // AdamW update with decoupled weight decay:
                // θ_t = (1 - lr * λ) * θ_{t-1} - lr_t * m_t / (√v_t + ε)
                let adaptive_update = &m_t / &(v_t.mapv(|x| x.sqrt()) + self.epsilon) * lr_t;

                // Apply weight decay directly to parameters (decoupled)
                let weight_decay_factor = 1.0 - self.lr * self.weight_decay;
                *param.data_mut() = param.data() * weight_decay_factor - &adaptive_update;

                self.m[i] = Some(m_t);
                self.v[i] = Some(v_t);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adamw_quadratic_convergence() {
        // Test convergence on f(x) = x²
        let mut params = vec![Tensor::from_vec(vec![5.0, -3.0, 2.0], true)];
        let mut optimizer = AdamW::default_params(0.1);

        for _ in 0..100 {
            // Compute gradient: ∇(x²) = 2x
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);

            optimizer.step(&mut params);
        }

        // Should converge close to 0
        for &val in params[0].data().iter() {
            assert!(val.abs() < 0.5, "Value {} did not converge", val);
        }
    }

    #[test]
    fn test_adamw_weight_decay() {
        // Test that weight decay is properly applied
        let mut params = vec![Tensor::from_vec(vec![1.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);

        // Zero gradient - only weight decay should apply
        let grad = ndarray::arr1(&[0.0]);
        params[0].set_grad(grad);

        let initial_value = params[0].data()[0];
        optimizer.step(&mut params);
        let after_step = params[0].data()[0];

        // With zero gradient, weight decay should reduce the parameter
        // θ_t = (1 - lr * λ) * θ_{t-1} = (1 - 0.1 * 0.1) * 1.0 = 0.99
        assert!(after_step < initial_value);
        assert_abs_diff_eq!(after_step, 0.99, epsilon = 1e-6);
    }

    #[test]
    fn test_adamw_vs_adam_difference() {
        // AdamW and Adam should behave differently with weight decay
        let mut params_adamw = vec![Tensor::from_vec(vec![2.0, -2.0], true)];
        let mut params_adam = vec![Tensor::from_vec(vec![2.0, -2.0], true)];

        let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut adam = super::super::Adam::default_params(0.1);

        for _ in 0..10 {
            // Same gradient for both
            let grad = ndarray::arr1(&[1.0, -1.0]);

            params_adamw[0].set_grad(grad.clone());
            params_adam[0].set_grad(grad.clone());

            adamw.step(&mut params_adamw);
            adam.step(&mut params_adam);
        }

        // With weight decay, AdamW should have smaller absolute values
        // (weight decay shrinks parameters toward zero)
        assert!(params_adamw[0].data()[0].abs() < params_adam[0].data()[0].abs());
        assert!(params_adamw[0].data()[1].abs() < params_adam[0].data()[1].abs());
    }
}

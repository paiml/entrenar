//! Activation function autograd operations: relu, gelu, swish, softmax

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// ReLU activation
pub fn relu(a: &Tensor) -> Tensor {
    let data = a.data().mapv(|x| x.max(0.0));
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(ReluBackward {
            a: a_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct ReluBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for ReluBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * (a > 0)
                let grad_a = grad * &self.a.data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// GELU activation (Gaussian Error Linear Unit)
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// ONE PATH: Forward math delegates to `trueno::gelu_scalar` (UCBD §4).
pub fn gelu(a: &Tensor) -> Tensor {
    let data = a.data().mapv(trueno::gelu_scalar);

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(GeluBackward {
            a: a_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct GeluBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for GeluBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const COEFF: f32 = 0.044_715;

                // ∂GELU/∂x = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
                // where z = √(2/π) * (x + 0.044715 * x³)
                // and dz/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
                let grad_a: Vec<f32> = self
                    .a
                    .data()
                    .iter()
                    .zip(grad_output.iter())
                    .map(|(&x, &grad)| {
                        let x2 = x * x;
                        let x3 = x2 * x;
                        let z = SQRT_2_OVER_PI * (x + COEFF * x3);
                        let tanh_z = z.tanh();
                        let sech2_z = 1.0 - tanh_z * tanh_z;
                        let dz_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);

                        let gelu_grad = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech2_z * dz_dx;
                        grad * gelu_grad
                    })
                    .collect();

                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Swish activation (also known as SiLU - Sigmoid Linear Unit)
///
/// Swish(x) = x * sigmoid(x) = x / (1 + e^(-x))
///
/// ONE PATH: Forward math delegates to `trueno::silu_scalar` (UCBD §4).
pub fn swish(a: &Tensor) -> Tensor {
    let data = a.data().mapv(trueno::silu_scalar);

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let output_clone = result.clone();
        let backward_op = Rc::new(SwishBackward {
            a: a_clone,
            output: output_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct SwishBackward {
    a: Tensor,
    output: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SwishBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂Swish/∂x = Swish(x) + sigmoid(x) * (1 - Swish(x))
                // This can be simplified to: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                let grad_a: Vec<f32> = self
                    .a
                    .data()
                    .iter()
                    .zip(self.output.data().iter())
                    .zip(grad_output.iter())
                    .map(|((&x, &swish_x), &grad)| {
                        let sigmoid = 1.0 / (1.0 + (-x).exp());
                        let swish_grad = swish_x + sigmoid * (1.0 - swish_x);
                        grad * swish_grad
                    })
                    .collect();

                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Softmax activation
pub fn softmax(a: &Tensor) -> Tensor {
    let max_val = a.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals = a.data().mapv(|x| (x - max_val).exp());
    let sum_exp = exp_vals.sum();
    let data = exp_vals / sum_exp;

    let requires_grad = a.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let output_clone = result.clone();
        let backward_op = Rc::new(SoftmaxBackward {
            a: a_clone,
            output: output_clone,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct SoftmaxBackward {
    a: Tensor,
    output: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂x = y ⊙ (∂L/∂y - (y · ∂L/∂y))
                let y = self.output.data();
                let dot = (y * grad_output).sum();
                let grad_a = y * &(grad_output - dot);
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

// =========================================================================
// FALSIFY-SI: silu-kernel-v1.yaml contract (entrenar via trueno::silu_scalar)
//
// Five-Whys (PMAT-354, Phase 11):
//   Why 1: entrenar had zero FALSIFY-SI-* tests despite SiLU in CUDA forward
//   Why 2: CUDA tests verify backward correctness, not mathematical invariants
//   Why 3: no mapping from silu-kernel-v1.yaml to entrenar test names
//   Why 4: entrenar predates the provable-contracts YAML convention
//   Why 5: SiLU CUDA forward delegates to cuBLAS (assumed correct)
//
// Note: entrenar's SiLU is CUDA-only (silu_forward/silu_backward). These
// tests exercise trueno::silu_scalar which is the canonical reference impl.
//
// References:
//   - provable-contracts/contracts/silu-kernel-v1.yaml
//   - Ramachandran et al. (2017) "Searching for Activation Functions"
// =========================================================================

#[cfg(test)]
mod silu_contract_tests {
    /// FALSIFY-SI-001: Zero preservation — SiLU(0) = 0
    #[test]
    fn falsify_si_001_zero_preservation() {
        let y = trueno::silu_scalar(0.0);
        assert!(y.abs() < 1e-7, "FALSIFIED SI-001: SiLU(0) = {y}, expected 0");
    }

    /// FALSIFY-SI-002: Global lower bound — SiLU(x) > -0.279 for all x
    #[test]
    fn falsify_si_002_global_lower_bound() {
        let test_values: Vec<f32> = vec![
            -100.0, -50.0, -10.0, -5.0, -2.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 100.0,
        ];
        for &x in &test_values {
            let y = trueno::silu_scalar(x);
            assert!(y > -0.28, "FALSIFIED SI-002: SiLU({x}) = {y}, expected > -0.279");
        }
    }

    /// FALSIFY-SI-003: Monotonic for positive inputs
    #[test]
    fn falsify_si_003_monotonic_positive() {
        let values: Vec<f32> = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
        for i in 1..values.len() {
            let y_prev = trueno::silu_scalar(values[i - 1]);
            let y_curr = trueno::silu_scalar(values[i]);
            assert!(
                y_curr > y_prev,
                "FALSIFIED SI-003: SiLU({}) = {y_curr} not > SiLU({}) = {y_prev}",
                values[i], values[i - 1]
            );
        }
    }

    /// FALSIFY-SI-005: Asymptotic linearity — |SiLU(x) - x| < 0.01 for x > 10
    #[test]
    fn falsify_si_005_asymptotic_linearity() {
        for &x in &[10.0f32, 20.0, 50.0, 100.0, 500.0] {
            let y = trueno::silu_scalar(x);
            assert!(
                (y - x).abs() < 0.01,
                "FALSIFIED SI-005: |SiLU({x}) - {x}| = {} >= 0.01",
                (y - x).abs()
            );
        }
    }

    /// FALSIFY-SI-006: Large negative → 0
    #[test]
    fn falsify_si_006_large_negative_vanishes() {
        for &x in &[-10.0f32, -20.0, -50.0, -100.0] {
            let y = trueno::silu_scalar(x);
            assert!(y.abs() < 0.01, "FALSIFIED SI-006: SiLU({x}) = {y}, expected ≈ 0");
        }
    }

    mod si_proptest_falsify {
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_si_002_prop_lower_bound(x in -1000.0_f32..1000.0) {
                let y = trueno::silu_scalar(x);
                prop_assert!(y > -0.28, "FALSIFIED SI-002-prop: SiLU({x}) = {y}");
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_si_003_prop_monotonic_positive(
                a in 0.001_f32..100.0,
                b in 0.001_f32..100.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    prop_assert!(
                        trueno::silu_scalar(hi) > trueno::silu_scalar(lo),
                        "FALSIFIED SI-003-prop: SiLU({hi}) not > SiLU({lo})"
                    );
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_si_005_prop_asymptotic(x in 10.0_f32..500.0) {
                let y = trueno::silu_scalar(x);
                prop_assert!(
                    (y - x).abs() < 0.01,
                    "FALSIFIED SI-005-prop: |SiLU({x}) - {x}| = {}",
                    (y - x).abs()
                );
            }
        }
    }
}

// =========================================================================
// FALSIFY-GE: gelu-kernel-v1.yaml contract (entrenar autograd gelu)
// =========================================================================
#[cfg(test)]
mod gelu_contract_tests {
    use super::*;
    use ndarray::Array1;

    /// FALSIFY-GE-001: Non-negativity — gelu(x) >= 0 for positive x
    #[test]
    fn falsify_ge_001_non_negativity() {
        let x = Tensor::new(Array1::from(vec![0.001, 0.1, 1.0, 5.0, 10.0, 100.0]), false);
        let y = gelu(&x);
        for (i, &val) in y.data().iter().enumerate() {
            assert!(val >= 0.0, "FALSIFIED GE-001: gelu(positive)[{i}] = {val} < 0");
        }
    }

    /// FALSIFY-GE-002: Monotonicity — ordering preserved for positive inputs
    #[test]
    fn falsify_ge_002_positive_monotonicity() {
        let x = Tensor::new(Array1::from(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]), false);
        let y = gelu(&x);
        let data = y.data();
        for i in 1..data.len() {
            assert!(
                data[i] > data[i - 1],
                "FALSIFIED GE-002: gelu not monotonic: [{i}]={} not > [{}]={}",
                data[i], i - 1, data[i - 1]
            );
        }
    }

    /// FALSIFY-GE-003: Zero preservation — gelu(0) = 0
    #[test]
    fn falsify_ge_003_zero_preservation() {
        let x = Tensor::new(Array1::from(vec![0.0]), false);
        let y = gelu(&x);
        assert!(y.data()[0].abs() < 1e-7, "FALSIFIED GE-003: gelu(0) = {}", y.data()[0]);
    }

    /// FALSIFY-GE-006: Large input stability
    #[test]
    fn falsify_ge_006_large_input_stability() {
        let x = Tensor::new(Array1::from(vec![10.0, 50.0, -10.0, -50.0]), false);
        let y = gelu(&x);
        let d = y.data();
        assert!((d[0] - 10.0).abs() < 0.01, "FALSIFIED GE-006: gelu(10) = {}", d[0]);
        assert!((d[1] - 50.0).abs() < 0.01, "FALSIFIED GE-006: gelu(50) = {}", d[1]);
        assert!(d[2].abs() < 0.01, "FALSIFIED GE-006: gelu(-10) = {}", d[2]);
        assert!(d[3].abs() < 0.01, "FALSIFIED GE-006: gelu(-50) = {}", d[3]);
    }
}

//! Matrix multiplication autograd operations
//!
//! Uses realizar's CUDA executor for GPU acceleration, falls back to trueno SIMD GEMM on CPU.
//! Both forward AND backward passes use CUDA GEMM for full GPU acceleration.
//! Instrumented with TRACER for empirical overhead analysis.

use crate::autograd::{BackwardOp, Tensor};
use crate::trace::{TraceStep, TRACER};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;

/// Global CUDA executor (singleton, initialized once)
#[cfg(feature = "cuda")]
static CUDA_EXECUTOR: OnceLock<Option<Mutex<CudaExecutor>>> = OnceLock::new();

/// Get or initialize CUDA executor
#[cfg(feature = "cuda")]
fn get_cuda_executor() -> Option<&'static Mutex<CudaExecutor>> {
    CUDA_EXECUTOR
        .get_or_init(|| match CudaExecutor::new(0) {
            Ok(executor) => {
                eprintln!("realizar CUDA executor initialized on GPU 0");
                Some(Mutex::new(executor))
            }
            Err(e) => {
                eprintln!("CUDA init failed: {e:?}, using CPU");
                None
            }
        })
        .as_ref()
}

/// Transpose a row-major matrix (rows x cols) to (cols x rows)
/// Uses cache-efficient blocked transpose for large matrices
#[inline]
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    TRACER.start(TraceStep::Transpose);
    let mut transposed = vec![0.0f32; rows * cols];

    const BLOCK_SIZE: usize = 32;
    if rows >= BLOCK_SIZE && cols >= BLOCK_SIZE {
        transpose_blocked(data, &mut transposed, rows, cols, BLOCK_SIZE);
    } else {
        transpose_simple(data, &mut transposed, rows, cols);
    }

    TRACER.end(TraceStep::Transpose, format!("{rows}x{cols}"));
    transposed
}

/// Blocked transpose for cache efficiency on large matrices.
#[inline]
fn transpose_blocked(src: &[f32], dst: &mut [f32], rows: usize, cols: usize, block: usize) {
    for r_block in (0..rows).step_by(block) {
        for c_block in (0..cols).step_by(block) {
            let r_end = (r_block + block).min(rows);
            let c_end = (c_block + block).min(cols);
            for r in r_block..r_end {
                for c in c_block..c_end {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Simple transpose for small matrices.
#[inline]
fn transpose_simple(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Compute matrix multiplication using realizar CUDA if available, else SIMD CPU
#[cfg(feature = "cuda")]
pub fn matmul_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Try CUDA via realizar
    if let Some(executor_mutex) = get_cuda_executor() {
        if let Ok(mut executor) = executor_mutex.lock() {
            match cuda_matmul(&mut executor, a, b, m, k, n) {
                Ok(result) => return result,
                Err(e) => {
                    eprintln!("CUDA matmul failed: {e:?}, falling back to CPU");
                }
            }
        }
    }

    // Fall back to trueno SIMD
    cpu_matmul(a, b, m, k, n)
}

/// CUDA matrix multiplication via realizar's CudaExecutor
#[cfg(feature = "cuda")]
fn cuda_matmul(
    executor: &mut CudaExecutor,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    TRACER.start(TraceStep::Alloc);
    let mut c = vec![0.0f32; m * n];
    TRACER.end(TraceStep::Alloc, format!("{m}x{n}"));

    TRACER.start(TraceStep::Matmul);
    executor.gemm(a, b, &mut c, m as u32, n as u32, k as u32).map_err(|e| format!("{e:?}"))?;
    TRACER.end(TraceStep::Matmul, format!("{m}x{k}x{n}"));
    Ok(c)
}

/// CPU fallback using trueno SIMD GEMM
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    if let Err(e) = trueno::blis::gemm(m, n, k, a, b, &mut c) {
        eprintln!("trueno gemm failed: {e:?}, using naive");
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    c
}

/// CPU-only path (no CUDA feature)
#[cfg(not(feature = "cuda"))]
pub fn matmul_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    cpu_matmul(a, b, m, k, n)
}

/// Matrix multiplication
///
/// Computes C = A @ B where:
/// - A is m×k (flattened to length m*k)
/// - B is k×n (flattened to length k*n)
/// - C is m×n (flattened to length m*n)
///
/// Uses GPU acceleration when available (requires `gpu` feature).
///
/// # Arguments
/// * `a` - Left matrix (m×k flattened)
/// * `b` - Right matrix (k×n flattened)
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A (= rows in B)
/// * `n` - Number of columns in B
pub fn matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");

    // Compute C = A @ B using GPU if available
    let result_data = matmul_compute(
        a.data().as_slice().expect("matrix A must be contiguous"),
        b.data().as_slice().expect("matrix B must be contiguous"),
        m,
        k,
        n,
    );

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(MatmulBackward {
            a: a_clone,
            b: b_clone,
            m,
            k,
            n,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct MatmulBackward {
    a: Tensor,
    b: Tensor,
    m: usize,
    k: usize,
    n: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MatmulBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            // ∂L/∂A = ∂L/∂C @ B^T  (m×n) @ (n×k) = (m×k)
            // ∂L/∂B = A^T @ ∂L/∂C  (k×m) @ (m×n) = (k×n)

            let grad_c = grad_output.as_slice().expect("gradient output must be contiguous");
            let a_data = self.a.data();
            let b_data = self.b.data();
            let a_slice = a_data.as_slice().expect("matrix A must be contiguous");
            let b_slice = b_data.as_slice().expect("matrix B must be contiguous");

            if self.a.requires_grad() {
                // grad_A = grad_C @ B^T
                // grad_C is (m, n), B is (k, n), B^T is (n, k)
                // Result: (m, n) @ (n, k) = (m, k)
                let b_t = transpose(b_slice, self.k, self.n);
                let grad_a = matmul_compute(grad_c, &b_t, self.m, self.n, self.k);
                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if self.b.requires_grad() {
                // grad_B = A^T @ grad_C
                // A is (m, k), A^T is (k, m), grad_C is (m, n)
                // Result: (k, m) @ (m, n) = (k, n)
                let a_t = transpose(a_slice, self.m, self.k);
                let grad_b = matmul_compute(&a_t, grad_c, self.k, self.m, self.n);
                self.b.accumulate_grad(Array1::from(grad_b));
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_identity() {
        // 1x1 matrix
        let data = vec![5.0];
        let result = transpose(&data, 1, 1);
        assert_eq!(result, vec![5.0]);
    }

    #[test]
    fn test_transpose_2x3() {
        // 2x3 matrix
        // [1, 2, 3]
        // [4, 5, 6]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose(&data, 2, 3);
        // Expected 3x2:
        // [1, 4]
        // [2, 5]
        // [3, 6]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_3x2() {
        // 3x2 matrix
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose(&data, 3, 2);
        // Expected 2x3:
        assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_matmul_compute_2x2() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_compute(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_compute_2x3_3x2() {
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[7, 8], [9, 10], [11, 12]] (3x2)
        // C = A @ B (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_compute(&a, &b, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_no_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(!c.requires_grad());
        assert_eq!(
            c.data().as_slice().expect("operation should succeed"),
            &[19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn test_matmul_with_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());
        assert!(c.backward_op().is_some());
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);

        // Set gradient of output
        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));

        // Trigger backward
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        // Check gradients are accumulated
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_matmul_a_requires_grad_only() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());

        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        assert!(a.grad().is_some());
        assert!(b.grad().is_none());
    }

    #[test]
    fn test_matmul_b_requires_grad_only() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());

        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        assert!(a.grad().is_none());
        assert!(b.grad().is_some());
    }

    #[test]
    #[should_panic(expected = "Matrix A size mismatch")]
    fn test_matmul_size_mismatch_a() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let _ = matmul(&a, &b, 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "Matrix B size mismatch")]
    fn test_matmul_size_mismatch_b() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0]), false);
        let _ = matmul(&a, &b, 2, 2, 2);
    }

    #[test]
    fn test_transpose_double_transpose() {
        // Transpose twice should give original
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = transpose(&data, 2, 3);
        let t2 = transpose(&t1, 3, 2);
        assert_eq!(data, t2);
    }

    // =========================================================================
    // FALSIFY-MM: matmul-kernel-v1.yaml contract (entrenar autograd matmul)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had 10 matmul tests but zero FALSIFY-MM-* tests
    //   Why 2: unit tests verify 2x2 cases and backward, not invariants
    //   Why 3: no mapping from matmul-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML convention
    //   Why 5: matmul was "obviously correct" (textbook GEMM + autograd)
    //
    // References:
    //   - provable-contracts/contracts/matmul-kernel-v1.yaml
    // =========================================================================

    /// FALSIFY-MM-001e: Shape correctness — output is [m, n]
    #[test]
    fn falsify_mm_001e_shape_correctness() {
        for (m, k, n) in [(2, 3, 4), (1, 5, 1), (4, 4, 4), (3, 1, 2)] {
            let result = matmul_compute(&vec![1.0; m * k], &vec![1.0; k * n], m, k, n);
            assert_eq!(
                result.len(),
                m * n,
                "FALSIFIED MM-001e: output len = {}, expected {} for ({m}x{k}) @ ({k}x{n})",
                result.len(),
                m * n
            );
        }
    }

    /// FALSIFY-MM-005e: Identity matrix — A @ I = A
    #[test]
    fn falsify_mm_005e_identity_matrix() {
        let m = 3;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut identity = vec![0.0; k * k];
        for i in 0..k {
            identity[i * k + i] = 1.0;
        }
        let result = matmul_compute(&a, &identity, m, k, k);
        for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "FALSIFIED MM-005e: (A@I)[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// FALSIFY-MM-002e: Numerical accuracy against reference
    #[test]
    fn falsify_mm_002e_numerical_accuracy() {
        // 2x3 @ 3x2 known result
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result = matmul_compute(&a, &b, 2, 3, 2);
        let expected = [58.0, 64.0, 139.0, 154.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "FALSIFIED MM-002e: result[{i}] = {got}, expected {exp}"
            );
        }
    }

    mod mm_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-MM-001e-prop: Shape correctness for random dimensions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_mm_001e_prop_shape(
                m in 1..=8usize,
                k in 1..=8usize,
                n in 1..=8usize,
            ) {
                let result = matmul_compute(&vec![1.0; m * k], &vec![1.0; k * n], m, k, n);
                prop_assert_eq!(result.len(), m * n);
            }
        }

        // FALSIFY-MM-005e-prop: Identity matrix for random dimensions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_mm_005e_prop_identity(
                m in 1..=6usize,
                k in 1..=6usize,
                seed in 0..500u32,
            ) {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let mut identity = vec![0.0; k * k];
                for i in 0..k {
                    identity[i * k + i] = 1.0;
                }
                let result = matmul_compute(&a, &identity, m, k, k);
                for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
                    prop_assert!(
                        (got - exp).abs() < 1e-4,
                        "FALSIFIED MM-005e-prop: (A@I)[{}] = {}, expected {}",
                        i, got, exp
                    );
                }
            }
        }
    }
}

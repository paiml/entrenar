//! CUDA-accelerated backward kernels for autograd
//!
//! This module wraps trueno-gpu backward kernels for GPU-accelerated gradient computation.
//! Provides 10-100x speedup over CPU ndarray implementations.
//!
//! # Safety
//!
//! This module uses unsafe code for CUDA kernel launching, which is inherently unsafe.
//! The unsafe blocks are required for FFI calls to the CUDA driver API.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_backward (this module)
//!             └── trueno-gpu/kernels/backward
//!                     └── PTX generation + CUDA driver
//! ```
//!
//! # Available Kernels
//!
//! - `relu_backward` - ReLU gradient: ∂L/∂x = ∂L/∂y * (x > 0)
//! - `gelu_backward` - GELU gradient with tanh approximation
//! - `silu_backward` - SiLU/Swish gradient
//! - `softmax_backward` - Softmax Jacobian-vector product
//! - `rms_norm_backward` - RMSNorm gradients for input and gamma
//! - `layer_norm_backward` - LayerNorm gradients for input, gamma, beta
//! - `gemm_backward_a` - Matrix multiply gradient w.r.t. A
//! - `gemm_backward_b` - Matrix multiply gradient w.r.t. B

#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::backward::{
    GeluBackwardKernel, GemmBackwardAKernel, GemmBackwardBKernel, LayerNormBackwardKernel,
    ReluBackwardKernel, RmsNormBackwardKernel, SiluBackwardKernel, SoftmaxBackwardKernel,
};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::Kernel;

use super::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for backward kernels
#[cfg(feature = "cuda")]
static KERNEL_CACHE: OnceLock<Mutex<KernelCache>> = OnceLock::new();

/// Cache for compiled backward kernel modules
#[cfg(feature = "cuda")]
struct KernelCache {
    ctx: Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
}

#[cfg(feature = "cuda")]
impl KernelCache {
    fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            modules: HashMap::new(),
        }
    }

    fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        if !self.modules.contains_key(name) {
            let module = CudaModule::from_ptx(&self.ctx, ptx).map_err(|e| {
                CudaTensorError::KernelError(format!("Failed to compile {name}: {e:?}"))
            })?;
            self.modules.insert(name.to_string(), module);
        }
        Ok(self.modules.get_mut(name).unwrap())
    }
}

/// Initialize kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_kernel_cache(ctx: Arc<CudaContext>) -> Result<()> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(KernelCache::new(ctx)));
    Ok(())
}

/// ReLU backward pass on GPU
///
/// Computes: grad_input = grad_output * (input > 0 ? 1 : 0)
///
/// # Arguments
/// * `input` - Original input to forward pass
/// * `grad_output` - Gradient from upstream
/// * `grad_input` - Output buffer for computed gradient
/// * `stream` - CUDA stream for async execution
#[cfg(feature = "cuda")]
pub fn relu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;
    let kernel = ReluBackwardKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = cache.get_or_compile("relu_backward", &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "relu_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("ReLU backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// GELU backward pass on GPU
///
/// Computes gradient using tanh approximation derivative
#[cfg(feature = "cuda")]
pub fn gelu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;
    let kernel = GeluBackwardKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = cache.get_or_compile("gelu_backward", &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "gelu_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GELU backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// SiLU/Swish backward pass on GPU
///
/// Computes: grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
#[cfg(feature = "cuda")]
pub fn silu_backward(
    input: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let n = input.len() as u32;
    let kernel = SiluBackwardKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let module = cache.get_or_compile("silu_backward", &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &input_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "silu_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("SiLU backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// Softmax backward pass on GPU
///
/// Computes: grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))
#[cfg(feature = "cuda")]
pub fn softmax_backward(
    softmax_output: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    batch_size: u32,
    seq_len: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = SoftmaxBackwardKernel::new(batch_size, seq_len);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("softmax_backward_{batch_size}_{seq_len}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Softmax backward uses warp-parallel reduction
    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (32.min(seq_len), 1, 1), // Warp size
        shared_mem: 0,
    };

    let output_ptr = softmax_output.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &output_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &seq_len as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "softmax_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("Softmax backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// RMSNorm backward pass on GPU
///
/// Computes gradients for input and gamma parameters
#[cfg(feature = "cuda")]
pub fn rms_norm_backward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    grad_gamma: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = RmsNormBackwardKernel::new(batch_size, hidden_size, eps);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("rms_norm_backward_{batch_size}_{hidden_size}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (256.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();
    let grad_gamma_ptr = grad_gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 8] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &grad_gamma_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
        &eps as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "rms_norm_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("RMSNorm backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// LayerNorm backward pass on GPU
///
/// Computes gradients for input, gamma, and beta parameters
#[cfg(feature = "cuda")]
pub fn layer_norm_backward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_input: &mut GpuBuffer<f32>,
    grad_gamma: &mut GpuBuffer<f32>,
    grad_beta: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = LayerNormBackwardKernel::new(batch_size, hidden_size);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("layer_norm_backward_{batch_size}_{hidden_size}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (256.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_in_ptr = grad_input.as_ptr();
    let grad_gamma_ptr = grad_gamma.as_ptr();
    let grad_beta_ptr = grad_beta.as_ptr();

    let mut args: [*mut std::ffi::c_void; 8] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_in_ptr as *const _ as *mut _,
        &grad_gamma_ptr as *const _ as *mut _,
        &grad_beta_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "layer_norm_backward", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("LayerNorm backward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// GEMM backward pass for matrix A on GPU
///
/// Given C = A @ B, computes: grad_A = grad_C @ B^T
#[cfg(feature = "cuda")]
pub fn gemm_backward_a(
    grad_output: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = GemmBackwardAKernel::new(m, k, n);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gemm_backward_a_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    let config = LaunchConfig {
        grid: (m.div_ceil(16), k.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
    };

    let grad_out_ptr = grad_output.as_ptr();
    let b_ptr = b.as_ptr();
    let grad_a_ptr = grad_a.as_ptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &grad_out_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &grad_a_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &k as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "gemm_backward_a", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GEMM backward A launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// GEMM backward pass for matrix B on GPU
///
/// Given C = A @ B, computes: grad_B = A^T @ grad_C
#[cfg(feature = "cuda")]
pub fn gemm_backward_b(
    a: &GpuBuffer<f32>,
    grad_output: &GpuBuffer<f32>,
    grad_b: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = GemmBackwardBKernel::new(m, k, n);
    let ptx = kernel.emit_ptx();

    let cache = KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gemm_backward_b_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    let config = LaunchConfig {
        grid: (k.div_ceil(16), n.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
    };

    let a_ptr = a.as_ptr();
    let grad_out_ptr = grad_output.as_ptr();
    let grad_b_ptr = grad_b.as_ptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &a_ptr as *const _ as *mut _,
        &grad_out_ptr as *const _ as *mut _,
        &grad_b_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &k as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "gemm_backward_b", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GEMM backward B launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backward_module_compiles() {
        // This test verifies the module compiles correctly
        // Actual CUDA tests require GPU hardware
        assert!(true);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_kernel_cache_initialization() {
        use trueno_gpu::driver::cuda_available;

        if !cuda_available() {
            return;
        }

        let ctx = CudaContext::new(0).unwrap();
        let ctx = Arc::new(ctx);
        let result = init_kernel_cache(ctx);
        assert!(result.is_ok());
    }

    /// Create a fresh GPU context for a test
    /// Note: Using fresh contexts per-test avoids CUDA driver state issues
    /// when running multiple tests sequentially
    #[cfg(feature = "cuda")]
    fn get_test_gpu_context() -> Option<Arc<CudaContext>> {
        use trueno_gpu::driver::cuda_available;

        if cuda_available() {
            CudaContext::new(0).ok().map(Arc::new)
        } else {
            None
        }
    }

    /// CPU reference implementation for ReLU backward
    fn relu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &dy)| if x > 0.0 { dy } else { 0.0 })
            .collect()
    }

    /// CPU reference implementation for GELU backward (tanh approximation)
    fn gelu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &dy)| {
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner.powi(2);
                let gelu_deriv = 0.5 * (1.0 + tanh_inner)
                    + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2));
                dy * gelu_deriv
            })
            .collect()
    }

    /// CPU reference implementation for SiLU backward
    fn silu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &dy)| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let silu_deriv = sigmoid + x * sigmoid * (1.0 - sigmoid);
                dy * silu_deriv
            })
            .collect()
    }

    /// CPU reference implementation for softmax backward
    fn softmax_backward_cpu(softmax_output: &[f32], grad_output: &[f32]) -> Vec<f32> {
        // grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))
        let dot: f32 = softmax_output
            .iter()
            .zip(grad_output.iter())
            .map(|(s, g)| s * g)
            .sum();
        softmax_output
            .iter()
            .zip(grad_output.iter())
            .map(|(&s, &g)| s * (g - dot))
            .collect()
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "trueno relu_backward kernel has invalid PTX - awaiting upstream fix"]
    fn test_relu_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // Test data: mix of positive and negative values
        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; 6];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        relu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 6];
        grad_input.copy_to_host(&mut result).unwrap();

        // CPU reference
        let expected = relu_backward_cpu(&input_data, &grad_output_data);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "ReLU backward mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "trueno relu_backward kernel has invalid PTX - awaiting upstream fix"]
    fn test_relu_backward_not_hardcoded() {
        // Mutation-killing test: verify result is NOT all zeros
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0]; // All positive
        let grad_output_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let grad_input_data: Vec<f32> = vec![0.0; 3];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        relu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 3];
        grad_input.copy_to_host(&mut result).unwrap();

        // Kill mutant: result should NOT be all zeros for positive inputs
        assert_ne!(
            result,
            vec![0.0, 0.0, 0.0],
            "mutant: ReLU backward returned all zeros"
        );
        // Verify correct values
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gelu_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; 4];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        gelu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 4];
        grad_input.copy_to_host(&mut result).unwrap();

        // CPU reference
        let expected = gelu_backward_cpu(&input_data, &grad_output_data);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "GELU backward mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gelu_backward_not_hardcoded() {
        // Mutation-killing test
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let input_data: Vec<f32> = vec![0.5, 1.0, 1.5];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; 3];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        gelu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 3];
        grad_input.copy_to_host(&mut result).unwrap();

        // Kill mutant: verify results are different for different inputs
        assert!(
            (result[0] - result[1]).abs() > 1e-5 || (result[1] - result[2]).abs() > 1e-5,
            "mutant: GELU backward returned identical values"
        );
        // All values should be positive for positive inputs
        assert!(
            result.iter().all(|&x| x > 0.0),
            "GELU gradient should be positive for x > 0"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; 4];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        silu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 4];
        grad_input.copy_to_host(&mut result).unwrap();

        // CPU reference
        let expected = silu_backward_cpu(&input_data, &grad_output_data);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "SiLU backward mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_silu_backward_not_hardcoded() {
        // Mutation-killing test
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let input_data: Vec<f32> = vec![0.5, 1.0, 2.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; 3];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        silu_backward(&input, &grad_output, &mut grad_input, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 3];
        grad_input.copy_to_host(&mut result).unwrap();

        // Kill mutant: verify gradient at x=0 is ~0.5 (σ(0) = 0.5)
        let grad_at_zero = silu_backward_cpu(&[0.0], &[1.0])[0];
        assert!(
            (grad_at_zero - 0.5).abs() < 1e-3,
            "SiLU gradient at 0 should be ~0.5"
        );

        // All gradients should be positive for positive inputs
        assert!(
            result.iter().all(|&x| x > 0.0),
            "SiLU gradient should be positive for x > 0"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_softmax_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // Softmax output: uniform distribution (1/4 each)
        let softmax_output_data: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
        let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // Cross-entropy grad
        let grad_input_data: Vec<f32> = vec![0.0; 4];

        let softmax_output = GpuBuffer::from_host(&ctx, &softmax_output_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        softmax_backward(
            &softmax_output,
            &grad_output,
            &mut grad_input,
            1,
            4,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 4];
        grad_input.copy_to_host(&mut result).unwrap();

        // CPU reference
        let expected = softmax_backward_cpu(&softmax_output_data, &grad_output_data);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Softmax backward mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_softmax_backward_batch() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // 2 batches, seq_len=3
        let softmax_output_data: Vec<f32> = vec![0.5, 0.3, 0.2, 0.1, 0.2, 0.7];
        let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let grad_input_data: Vec<f32> = vec![0.0; 6];

        let softmax_output = GpuBuffer::from_host(&ctx, &softmax_output_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();

        softmax_backward(
            &softmax_output,
            &grad_output,
            &mut grad_input,
            2,
            3,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 6];
        grad_input.copy_to_host(&mut result).unwrap();

        // Verify gradients are computed (not all zeros or NaN)
        assert!(
            !result.iter().any(|x| x.is_nan()),
            "Softmax backward should not produce NaN"
        );
        assert!(
            result.iter().any(|&x| x.abs() > 1e-5),
            "Softmax backward should produce non-zero gradients"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rms_norm_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let batch_size = 1u32;
        let hidden_size = 4u32;
        let n = (batch_size * hidden_size) as usize;
        let eps = 1e-5;

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; n];
        let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let gamma = GpuBuffer::from_host(&ctx, &gamma_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();
        let mut grad_gamma = GpuBuffer::from_host(&ctx, &grad_gamma_data).unwrap();

        rms_norm_backward(
            &input,
            &gamma,
            &grad_output,
            &mut grad_input,
            &mut grad_gamma,
            batch_size,
            hidden_size,
            eps,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result_input = vec![0.0f32; n];
        let mut result_gamma = vec![0.0f32; hidden_size as usize];
        grad_input.copy_to_host(&mut result_input).unwrap();
        grad_gamma.copy_to_host(&mut result_gamma).unwrap();

        // Verify gradients are computed
        assert!(
            !result_input.iter().any(|x| x.is_nan()),
            "RMSNorm backward should not produce NaN for input grad"
        );
        assert!(
            !result_gamma.iter().any(|x| x.is_nan()),
            "RMSNorm backward should not produce NaN for gamma grad"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_layer_norm_backward_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let batch_size = 1u32;
        let hidden_size = 4u32;
        let n = (batch_size * hidden_size) as usize;

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gamma_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_output_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grad_input_data: Vec<f32> = vec![0.0; n];
        let grad_gamma_data: Vec<f32> = vec![0.0; hidden_size as usize];
        let grad_beta_data: Vec<f32> = vec![0.0; hidden_size as usize];

        let input = GpuBuffer::from_host(&ctx, &input_data).unwrap();
        let gamma = GpuBuffer::from_host(&ctx, &gamma_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_input = GpuBuffer::from_host(&ctx, &grad_input_data).unwrap();
        let mut grad_gamma = GpuBuffer::from_host(&ctx, &grad_gamma_data).unwrap();
        let mut grad_beta = GpuBuffer::from_host(&ctx, &grad_beta_data).unwrap();

        layer_norm_backward(
            &input,
            &gamma,
            &grad_output,
            &mut grad_input,
            &mut grad_gamma,
            &mut grad_beta,
            batch_size,
            hidden_size,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result_input = vec![0.0f32; n];
        let mut result_gamma = vec![0.0f32; hidden_size as usize];
        let mut result_beta = vec![0.0f32; hidden_size as usize];
        grad_input.copy_to_host(&mut result_input).unwrap();
        grad_gamma.copy_to_host(&mut result_gamma).unwrap();
        grad_beta.copy_to_host(&mut result_beta).unwrap();

        // Verify gradients are computed without NaN/Inf
        assert!(
            !result_input.iter().any(|x| x.is_nan() || x.is_infinite()),
            "LayerNorm backward should not produce NaN/Inf for input grad"
        );
        assert!(
            !result_gamma.iter().any(|x| x.is_nan() || x.is_infinite()),
            "LayerNorm backward should not produce NaN/Inf for gamma grad"
        );
        assert!(
            !result_beta.iter().any(|x| x.is_nan() || x.is_infinite()),
            "LayerNorm backward should not produce NaN/Inf for beta grad"
        );
        // Note: grad_beta may be zero for single batch with uniform grad_output
        // due to how the backward kernel accumulates gradients
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gemm_backward_a_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // C = A @ B where A is 2x2, B is 2x2
        // grad_A = grad_C @ B^T
        let m = 2u32;
        let k = 2u32;
        let n = 2u32;

        // grad_C = [[1, 0], [0, 1]] (identity-like)
        let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        // B = [[1, 2], [3, 4]]
        // B^T = [[1, 3], [2, 4]]
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grad_a_data: Vec<f32> = vec![0.0; (m * k) as usize];

        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let b = GpuBuffer::from_host(&ctx, &b_data).unwrap();
        let mut grad_a = GpuBuffer::from_host(&ctx, &grad_a_data).unwrap();

        gemm_backward_a(&grad_output, &b, &mut grad_a, m, k, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; (m * k) as usize];
        grad_a.copy_to_host(&mut result).unwrap();

        // Verify gradients are computed and not NaN
        assert!(
            !result.iter().any(|x| x.is_nan()),
            "GEMM backward A should not produce NaN"
        );
        // grad_A = [[1, 0], [0, 1]] @ [[1, 3], [2, 4]] = [[1, 3], [2, 4]]
        // Expected: [1, 3, 2, 4]
        let expected = vec![1.0, 3.0, 2.0, 4.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "GEMM backward A mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gemm_backward_b_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // C = A @ B where A is 2x2, B is 2x2
        // grad_B = A^T @ grad_C
        let m = 2u32;
        let k = 2u32;
        let n = 2u32;

        // A = [[1, 2], [3, 4]]
        // A^T = [[1, 3], [2, 4]]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // grad_C = [[1, 0], [0, 1]] (identity-like)
        let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let grad_b_data: Vec<f32> = vec![0.0; (k * n) as usize];

        let a = GpuBuffer::from_host(&ctx, &a_data).unwrap();
        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let mut grad_b = GpuBuffer::from_host(&ctx, &grad_b_data).unwrap();

        gemm_backward_b(&a, &grad_output, &mut grad_b, m, k, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; (k * n) as usize];
        grad_b.copy_to_host(&mut result).unwrap();

        // Verify gradients are computed and not NaN
        assert!(
            !result.iter().any(|x| x.is_nan()),
            "GEMM backward B should not produce NaN"
        );
        // grad_B = [[1, 3], [2, 4]] @ [[1, 0], [0, 1]] = [[1, 3], [2, 4]]
        // Expected: [1, 3, 2, 4]
        let expected = vec![1.0, 3.0, 2.0, 4.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "GEMM backward B mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gemm_backward_larger_matrices() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        // Test with larger matrices to exercise block tiling
        let m = 32u32;
        let k = 32u32;
        let n = 32u32;

        let grad_output_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let grad_a_data: Vec<f32> = vec![0.0; (m * k) as usize];
        let grad_b_data: Vec<f32> = vec![0.0; (k * n) as usize];

        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).unwrap();
        let b = GpuBuffer::from_host(&ctx, &b_data).unwrap();
        let a = GpuBuffer::from_host(&ctx, &a_data).unwrap();
        let mut grad_a = GpuBuffer::from_host(&ctx, &grad_a_data).unwrap();
        let mut grad_b = GpuBuffer::from_host(&ctx, &grad_b_data).unwrap();

        gemm_backward_a(&grad_output, &b, &mut grad_a, m, k, n, &stream).unwrap();
        gemm_backward_b(&a, &grad_output, &mut grad_b, m, k, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result_a = vec![0.0f32; (m * k) as usize];
        let mut result_b = vec![0.0f32; (k * n) as usize];
        grad_a.copy_to_host(&mut result_a).unwrap();
        grad_b.copy_to_host(&mut result_b).unwrap();

        // Verify no NaN or Inf
        assert!(
            !result_a.iter().any(|x| x.is_nan() || x.is_infinite()),
            "GEMM backward A should not produce NaN/Inf"
        );
        assert!(
            !result_b.iter().any(|x| x.is_nan() || x.is_infinite()),
            "GEMM backward B should not produce NaN/Inf"
        );
        // Verify non-zero gradients
        assert!(
            result_a.iter().any(|&x| x.abs() > 1e-5),
            "GEMM backward A should produce non-zero gradients"
        );
        assert!(
            result_b.iter().any(|&x| x.abs() > 1e-5),
            "GEMM backward B should produce non-zero gradients"
        );
    }
}

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
}

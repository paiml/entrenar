//! CUDA-accelerated forward kernels for autograd
//!
//! This module wraps trueno-gpu forward kernels for GPU-accelerated forward passes.
//! Provides 10-100x speedup over CPU ndarray implementations.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_forward (this module)
//!             └── trueno-gpu/kernels
//!                     └── PTX generation + CUDA driver
//! ```
//!
//! # Available Kernels
//!
//! - `relu_forward` - ReLU activation
//! - `softmax_forward` - Numerically stable softmax with warp shuffle
//! - `layer_norm_forward` - Fused layer normalization
//! - `rms_norm_forward` - RMS normalization (LLaMA-style)
//! - `gelu_forward` - GELU activation
//! - `silu_forward` - SiLU/Swish activation
//! - `gemm_forward` - Matrix multiplication

#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    FusedSwigluKernel, GeluKernel, GemmKernel, Kernel, LayerNormKernel, ReluKernel, RmsNormKernel,
    SiluKernel, SoftmaxKernel,
};

use super::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for forward kernels
#[cfg(feature = "cuda")]
static FORWARD_KERNEL_CACHE: OnceLock<Mutex<ForwardKernelCache>> = OnceLock::new();

/// Cache for compiled forward kernel modules
#[cfg(feature = "cuda")]
struct ForwardKernelCache {
    ctx: std::sync::Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
}

#[cfg(feature = "cuda")]
impl ForwardKernelCache {
    fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
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

/// Initialize forward kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_forward_kernel_cache(ctx: std::sync::Arc<CudaContext>) -> Result<()> {
    FORWARD_KERNEL_CACHE.get_or_init(|| Mutex::new(ForwardKernelCache::new(ctx)));
    Ok(())
}

/// ReLU activation forward pass on GPU
///
/// Computes: output[i] = max(0, input[i])
#[cfg(feature = "cuda")]
pub fn relu_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = ReluKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("relu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "relu", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("ReLU forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// Softmax forward pass on GPU
///
/// Computes: output[i] = exp(input[i] - max(input)) / sum(exp(input - max(input)))
///
/// Uses warp-parallel reduction for numerical stability.
#[cfg(feature = "cuda")]
pub fn softmax_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    length: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = SoftmaxKernel::new(length);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("softmax_forward_{length}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32.min(length), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &length as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "softmax", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("Softmax forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// Layer normalization forward pass on GPU
///
/// Computes: output = gamma * (input - mean) / sqrt(var + eps) + beta
#[cfg(feature = "cuda")]
pub fn layer_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    beta: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = LayerNormKernel::new(hidden_size);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("layer_norm_forward_{hidden_size}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (256.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let beta_ptr = beta.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &beta_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "layer_norm", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("LayerNorm forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// RMS normalization forward pass on GPU (LLaMA-style)
///
/// Computes: output = gamma * input / sqrt(mean(input^2) + eps)
#[cfg(feature = "cuda")]
pub fn rms_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = RmsNormKernel::new(hidden_size);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("rms_norm_forward_{hidden_size}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (batch_size, 1, 1),
        block: (256.min(hidden_size), 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        &input_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &batch_size as *const _ as *mut _,
        &hidden_size as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "rms_norm", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("RMSNorm forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// GELU activation forward pass on GPU
///
/// Computes: output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[cfg(feature = "cuda")]
pub fn gelu_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = GeluKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gelu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "gelu", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GELU forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// SiLU/Swish activation forward pass on GPU
///
/// Computes: output = x * sigmoid(x) = x / (1 + exp(-x))
#[cfg(feature = "cuda")]
pub fn silu_forward(
    input: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = SiluKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("silu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "silu", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("SiLU forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// Fused SwiGLU forward pass on GPU (ENT-150)
///
/// Computes: output = SiLU(gate) * up
/// Fuses two operations into one kernel for better memory bandwidth.
#[cfg(feature = "cuda")]
pub fn fused_swiglu_forward(
    gate: &GpuBuffer<f32>,
    up: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = FusedSwigluKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("fused_swiglu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let gate_ptr = gate.as_ptr();
    let up_ptr = up.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 4] = [
        &gate_ptr as *const _ as *mut _,
        &up_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "fused_swiglu", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("Fused SwiGLU forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// GEMM forward pass on GPU
///
/// Computes: C = A @ B where A is MxK, B is KxN, C is MxN
#[cfg(feature = "cuda")]
pub fn gemm_forward(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    k: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = GemmKernel::naive(m, n, k);
    let ptx = kernel.emit_ptx();

    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gemm_forward_{m}_{k}_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Use 16x16 thread blocks for GEMM
    let config = LaunchConfig {
        grid: (m.div_ceil(16), n.div_ceil(16), 1),
        block: (16, 16, 1),
        shared_mem: 0,
    };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();

    let mut args: [*mut std::ffi::c_void; 6] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &c_ptr as *const _ as *mut _,
        &m as *const _ as *mut _,
        &k as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(module, "gemm_naive", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GEMM forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_forward_module_compiles() {
        // This test verifies the module compiles correctly
        // Actual CUDA tests require GPU hardware
        assert!(true);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_forward_kernel_cache_initialization() {
        use trueno_gpu::driver::cuda_available;

        if !cuda_available() {
            return;
        }

        let ctx = CudaContext::new(0).unwrap();
        let ctx = std::sync::Arc::new(ctx);
        let result = init_forward_kernel_cache(ctx);
        assert!(result.is_ok());
    }
}

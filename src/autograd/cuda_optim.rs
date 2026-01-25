//! CUDA-accelerated optimizer kernels for autograd
//!
//! This module wraps trueno-gpu optimizer kernels for GPU-resident weight updates.
//! Eliminates CPU↔GPU synchronization by keeping all optimizer state on GPU.
//!
//! # Architecture (SPEC-FT-001 v3.1.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_optim (this module)
//!             └── trueno-gpu/kernels/optimizer
//!                     └── AdamWStepKernel, AdamStepKernel, GradientClipKernel
//! ```
//!
//! # Available Functions
//!
//! - `adamw_step_cuda` - Fused AdamW with weight decay
//! - `adam_step_cuda` - Vanilla Adam without weight decay
//! - `gradient_clip_cuda` - Apply gradient clipping scale

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
use trueno_gpu::kernels::{AdamStepKernel, AdamWStepKernel, GradientClipKernel, Kernel};

use super::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for optimizer kernels
#[cfg(feature = "cuda")]
static OPTIM_KERNEL_CACHE: OnceLock<Mutex<OptimKernelCache>> = OnceLock::new();

/// Cache for compiled optimizer kernel modules
#[cfg(feature = "cuda")]
struct OptimKernelCache {
    ctx: std::sync::Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
}

#[cfg(feature = "cuda")]
impl OptimKernelCache {
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

/// Initialize optimizer kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_optim_kernel_cache(ctx: std::sync::Arc<CudaContext>) -> Result<()> {
    OPTIM_KERNEL_CACHE.get_or_init(|| Mutex::new(OptimKernelCache::new(ctx)));
    Ok(())
}

/// Fused AdamW optimizer step on GPU
///
/// Performs in-place weight update with momentum, adaptive learning rate, and weight decay.
///
/// # Arguments
/// - `params`: weight tensor (updated in-place)
/// - `grads`: gradient tensor
/// - `m`: first moment state (updated in-place)
/// - `v`: second moment state (updated in-place)
/// - `lr`: learning rate
/// - `beta1`: first moment decay (typically 0.9)
/// - `beta2`: second moment decay (typically 0.999)
/// - `eps`: numerical stability (typically 1e-8)
/// - `weight_decay`: L2 penalty coefficient
/// - `step`: current step number (for bias correction)
/// - `n`: number of parameters
/// - `stream`: CUDA stream
#[cfg(feature = "cuda")]
pub fn adamw_step_cuda(
    params: &mut GpuBuffer<f32>,
    grads: &GpuBuffer<f32>,
    m: &mut GpuBuffer<f32>,
    v: &mut GpuBuffer<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = AdamWStepKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = OPTIM_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("adamw_step_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    // Pre-compute bias correction factors
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(step as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(step as i32));

    let params_ptr = params.as_ptr();
    let grads_ptr = grads.as_ptr();
    let m_ptr = m.as_ptr();
    let v_ptr = v.as_ptr();

    let mut args: [*mut std::ffi::c_void; 12] = [
        &params_ptr as *const _ as *mut _,
        &grads_ptr as *const _ as *mut _,
        &m_ptr as *const _ as *mut _,
        &v_ptr as *const _ as *mut _,
        &lr as *const _ as *mut _,
        &beta1 as *const _ as *mut _,
        &beta2 as *const _ as *mut _,
        &eps as *const _ as *mut _,
        &weight_decay as *const _ as *mut _,
        &bias_correction1 as *const _ as *mut _,
        &bias_correction2 as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "adamw_step", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("AdamW step launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// Fused Adam optimizer step on GPU (no weight decay)
///
/// Same as `adamw_step_cuda` but without the decoupled weight decay term.
#[cfg(feature = "cuda")]
pub fn adam_step_cuda(
    params: &mut GpuBuffer<f32>,
    grads: &GpuBuffer<f32>,
    m: &mut GpuBuffer<f32>,
    v: &mut GpuBuffer<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: u32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = AdamStepKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = OPTIM_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("adam_step_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    // Pre-compute bias correction factors
    let bias_correction1 = 1.0 / (1.0 - beta1.powi(step as i32));
    let bias_correction2 = 1.0 / (1.0 - beta2.powi(step as i32));

    let params_ptr = params.as_ptr();
    let grads_ptr = grads.as_ptr();
    let m_ptr = m.as_ptr();
    let v_ptr = v.as_ptr();

    let mut args: [*mut std::ffi::c_void; 11] = [
        &params_ptr as *const _ as *mut _,
        &grads_ptr as *const _ as *mut _,
        &m_ptr as *const _ as *mut _,
        &v_ptr as *const _ as *mut _,
        &lr as *const _ as *mut _,
        &beta1 as *const _ as *mut _,
        &beta2 as *const _ as *mut _,
        &eps as *const _ as *mut _,
        &bias_correction1 as *const _ as *mut _,
        &bias_correction2 as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "adam_step", &config, &mut args)
            .map_err(|e| CudaTensorError::KernelError(format!("Adam step launch failed: {e:?}")))?;
    }

    Ok(())
}

/// Apply gradient clipping on GPU
///
/// Scales gradients by a pre-computed factor to enforce maximum norm.
///
/// # Arguments
/// - `grads`: gradient tensor (updated in-place)
/// - `scale`: clipping scale factor (pre-computed as `min(1.0, max_norm / grad_norm)`)
/// - `n`: number of gradient elements
/// - `stream`: CUDA stream
///
/// # Usage
/// ```ignore
/// // Compute gradient norm on host
/// let grad_norm = compute_l2_norm(&grads);
/// let scale = (max_norm / grad_norm).min(1.0);
///
/// // Apply clipping on GPU
/// gradient_clip_cuda(&mut grads, scale, n, &stream)?;
/// ```
#[cfg(feature = "cuda")]
pub fn gradient_clip_cuda(
    grads: &mut GpuBuffer<f32>,
    scale: f32,
    n: u32,
    stream: &CudaStream,
) -> Result<()> {
    // Skip kernel launch if no clipping needed
    if (scale - 1.0).abs() < 1e-7 {
        return Ok(());
    }

    let kernel = GradientClipKernel::new(n);
    let ptx = kernel.emit_ptx();

    let cache = OPTIM_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let key = format!("gradient_clip_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };

    let grads_ptr = grads.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &grads_ptr as *const _ as *mut _,
        &scale as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "gradient_clip", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("Gradient clip launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_optim_module_compiles() {
        // This test verifies the module compiles correctly
        // Actual CUDA tests require GPU hardware
        assert!(true);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_optim_kernel_cache_initialization() {
        use trueno_gpu::driver::cuda_available;

        if !cuda_available() {
            return;
        }

        let ctx = CudaContext::new(0).unwrap();
        let ctx = std::sync::Arc::new(ctx);
        let result = init_optim_kernel_cache(ctx);
        assert!(result.is_ok());
    }
}

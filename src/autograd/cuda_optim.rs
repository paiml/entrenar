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
    let mut cache = cache.lock().map_err(|_err| {
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
    let mut cache = cache.lock().map_err(|_err| {
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
    let mut cache = cache.lock().map_err(|_err| {
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

    /// Create a fresh GPU context for a test
    /// Note: Using fresh contexts per-test avoids CUDA driver state issues
    /// when running multiple tests sequentially
    #[cfg(feature = "cuda")]
    fn get_test_gpu_context() -> Option<std::sync::Arc<CudaContext>> {
        use trueno_gpu::driver::cuda_available;

        if cuda_available() {
            CudaContext::new(0).ok().map(std::sync::Arc::new)
        } else {
            None
        }
    }

    /// CPU reference implementation for AdamW step
    fn adamw_step_cpu(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let bias_correction1 = 1.0 / (1.0 - beta1.powi(step as i32));
        let bias_correction2 = 1.0 / (1.0 - beta2.powi(step as i32));

        for i in 0..params.len() {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = m[i] * bias_correction1;
            let v_hat = v[i] * bias_correction2;

            // AdamW update: weight decay is applied directly to params
            params[i] = params[i] * (1.0 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// CPU reference implementation for Adam step (no weight decay)
    fn adam_step_cpu(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: u32,
    ) {
        let bias_correction1 = 1.0 / (1.0 - beta1.powi(step as i32));
        let bias_correction2 = 1.0 / (1.0 - beta2.powi(step as i32));

        for i in 0..params.len() {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Compute bias-corrected estimates
            let m_hat = m[i] * bias_correction1;
            let v_hat = v[i] * bias_correction2;

            // Adam update (no weight decay)
            params[i] = params[i] - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// CPU reference implementation for gradient clipping
    fn gradient_clip_cpu(grads: &mut [f32], scale: f32) {
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_step_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let step = 1u32;

        // Initial values
        let mut params_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let mut m_data: Vec<f32> = vec![0.0; n as usize];
        let mut v_data: Vec<f32> = vec![0.0; n as usize];

        // CPU reference
        let mut cpu_params = params_data.clone();
        let mut cpu_m = m_data.clone();
        let mut cpu_v = v_data.clone();
        adamw_step_cpu(
            &mut cpu_params,
            &grads_data,
            &mut cpu_m,
            &mut cpu_v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        );

        // GPU execution
        let mut params = GpuBuffer::from_host(&ctx, &params_data).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        params.copy_to_host(&mut params_data).unwrap();
        m.copy_to_host(&mut m_data).unwrap();
        v.copy_to_host(&mut v_data).unwrap();

        // Compare GPU vs CPU results
        for i in 0..n as usize {
            assert!(
                (params_data[i] - cpu_params[i]).abs() < 1e-4,
                "AdamW params mismatch at {i}: GPU={}, CPU={}",
                params_data[i],
                cpu_params[i]
            );
            assert!(
                (m_data[i] - cpu_m[i]).abs() < 1e-5,
                "AdamW m mismatch at {i}: GPU={}, CPU={}",
                m_data[i],
                cpu_m[i]
            );
            assert!(
                (v_data[i] - cpu_v[i]).abs() < 1e-5,
                "AdamW v mismatch at {i}: GPU={}, CPU={}",
                v_data[i],
                cpu_v[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_step_not_hardcoded() {
        // Mutation-killing test: verify params actually change
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let initial_params: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5]; // Non-zero gradients
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params = GpuBuffer::from_host(&ctx, &initial_params).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.01, // Larger LR to see effect
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result_params = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result_params).unwrap();

        // Kill mutant: params should have changed
        assert_ne!(
            result_params, initial_params,
            "mutant: AdamW params unchanged after step"
        );
        // Verify params decreased (negative gradient update)
        for (i, (&new, &old)) in result_params.iter().zip(initial_params.iter()).enumerate() {
            assert!(
                new < old,
                "AdamW params[{i}] should decrease with positive gradients"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adamw_weight_decay() {
        // Test that weight decay is actually applied
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let params_data: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0]; // Large weights
        let grads_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0]; // Zero gradients
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params = GpuBuffer::from_host(&ctx, &params_data).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        // With zero gradients, only weight decay should affect params
        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.01, // LR
            0.9,
            0.999,
            1e-8,
            0.1, // High weight decay
            1,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result).unwrap();

        // With zero gradients, params should decay: p = p * (1 - lr * wd)
        let expected = 10.0 * (1.0 - 0.01 * 0.1);
        for (i, &p) in result.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-3,
                "Weight decay not applied correctly at {i}: got {p}, expected {expected}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adam_step_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let step = 1u32;

        // Initial values
        let mut params_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grads_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let mut m_data: Vec<f32> = vec![0.0; n as usize];
        let mut v_data: Vec<f32> = vec![0.0; n as usize];

        // CPU reference
        let mut cpu_params = params_data.clone();
        let mut cpu_m = m_data.clone();
        let mut cpu_v = v_data.clone();
        adam_step_cpu(
            &mut cpu_params,
            &grads_data,
            &mut cpu_m,
            &mut cpu_v,
            lr,
            beta1,
            beta2,
            eps,
            step,
        );

        // GPU execution
        let mut params = GpuBuffer::from_host(&ctx, &params_data).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        adam_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            step,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        params.copy_to_host(&mut params_data).unwrap();
        m.copy_to_host(&mut m_data).unwrap();
        v.copy_to_host(&mut v_data).unwrap();

        // Compare GPU vs CPU results
        for i in 0..n as usize {
            assert!(
                (params_data[i] - cpu_params[i]).abs() < 1e-4,
                "Adam params mismatch at {i}: GPU={}, CPU={}",
                params_data[i],
                cpu_params[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_adam_step_multiple_iterations() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let lr = 0.01f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;

        let mut params_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let grads_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params = GpuBuffer::from_host(&ctx, &params_data).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        // Run 10 steps
        for step in 1..=10 {
            adam_step_cuda(
                &mut params,
                &grads,
                &mut m,
                &mut v,
                lr,
                beta1,
                beta2,
                eps,
                step,
                n,
                &stream,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        params.copy_to_host(&mut params_data).unwrap();

        // Params should have decreased significantly after 10 steps
        for &p in &params_data {
            assert!(p < 1.0, "Params should decrease after multiple Adam steps");
            assert!(p > 0.0, "Params should remain positive");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_basic() {
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let grads_data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        let scale = 0.5f32; // Scale down by half

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();

        gradient_clip_cuda(&mut grads, scale, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).unwrap();

        // CPU reference
        let mut expected = grads_data.clone();
        gradient_clip_cpu(&mut expected, scale);

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Gradient clip mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_no_op() {
        // Test that scale=1.0 is a no-op
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let grads_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let scale = 1.0f32; // No scaling

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();

        // This should be a no-op (kernel not even launched)
        gradient_clip_cuda(&mut grads, scale, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).unwrap();

        // Gradients should be unchanged
        for (i, (&got, &exp)) in result.iter().zip(grads_data.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "Gradient clip with scale=1 should not modify values at {i}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_clip_not_hardcoded() {
        // Mutation-killing test
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 4u32;
        let grads_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let scale = 0.1f32;

        let mut grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();

        gradient_clip_cuda(&mut grads, scale, n, &stream).unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; n as usize];
        grads.copy_to_host(&mut result).unwrap();

        // Kill mutant: result should NOT equal original
        assert_ne!(result, grads_data, "mutant: gradient clip had no effect");

        // Verify scaled values
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_optimizer_large_scale() {
        // Test with larger parameter count
        let ctx = match get_test_gpu_context() {
            Some(c) => c,
            None => return,
        };
        init_optim_kernel_cache(ctx.clone()).unwrap();
        let stream = CudaStream::new(&ctx).unwrap();

        let n = 1024u32;
        let params_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let grads_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let m_data: Vec<f32> = vec![0.0; n as usize];
        let v_data: Vec<f32> = vec![0.0; n as usize];

        let mut params = GpuBuffer::from_host(&ctx, &params_data).unwrap();
        let grads = GpuBuffer::from_host(&ctx, &grads_data).unwrap();
        let mut m = GpuBuffer::from_host(&ctx, &m_data).unwrap();
        let mut v = GpuBuffer::from_host(&ctx, &v_data).unwrap();

        adamw_step_cuda(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; n as usize];
        params.copy_to_host(&mut result).unwrap();

        // Verify no NaN or Inf
        assert!(
            !result.iter().any(|x| x.is_nan() || x.is_infinite()),
            "Large-scale optimizer should not produce NaN/Inf"
        );
    }
}

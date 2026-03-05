#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{BatchedVectorizedRmsNormKernel, Kernel, LayerNormKernel};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

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
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = LayerNormKernel::new(hidden_size);
    let kernel_name = kernel.name();

    let key = format!("layer_norm_forward_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

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

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("LayerNorm forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// RMS normalization forward pass on GPU (LLaMA-style)
///
/// Computes: output = gamma * input / sqrt(mean(input^2) + eps)
///
/// Uses BatchedVectorizedRmsNormKernel: single kernel launch processes all
/// batch_size rows in parallel via grid.y = batch_size, 256 threads per block.
///
/// ALB-076: Previously launched one 32-thread kernel per row (2048 launches for
/// batch=4, seq=512). nsys profiling showed this was 97.1% of all GPU time.
/// Single batched launch eliminates 100K+ kernel launches per step.
#[cfg(feature = "cuda")]
pub fn rms_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, batch_size);

    let key = format!("batched_rmsnorm_fwd_{hidden_size}");
    let module = match cache.get_cached(&key) {
        Some(m) => m,
        None => {
            let ptx = kernel.emit_ptx_for_target(cache.sm_target());
            cache.get_or_compile(&key, &ptx)?
        }
    };

    // Grid: (1, batch_size, 1) — one block per row, all rows in parallel
    // Block: (256, 1, 1) — 8 warps per block for parallel reduction
    let config = LaunchConfig {
        grid: (1, batch_size, 1),
        block: (256, 1, 1),
        shared_mem: 8 * 4, // 8 warp partial sums (f32)
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();
    let gamma_ptr = gamma.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &gamma_ptr as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. input has batch_size * hidden_size elements,
    // output has batch_size * hidden_size elements, gamma has hidden_size elements.
    // Parameters match PTX signature (u64 input_ptr, u64 output_ptr, u64 gamma_ptr).
    unsafe {
        stream
            .launch_kernel(module, "batched_rmsnorm_vectorized", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("RMSNorm forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

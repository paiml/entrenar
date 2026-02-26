#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{Kernel, LayerNormKernel, RmsNormKernel};

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
    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = LayerNormKernel::new(hidden_size);
    let kernel_name = kernel.name();
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

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

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, kernel_name, &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("LayerNorm forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

/// RMS normalization forward pass on GPU (LLaMA-style)
///
/// Computes: output = gamma * input / sqrt(mean(input^2) + eps)
///
/// Note: The kernel uses warp shuffle and requires 32 threads per block.
/// For batched input, each row is processed sequentially.
#[cfg(feature = "cuda")]
pub fn rms_norm_forward(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    batch_size: u32,
    hidden_size: u32,
    stream: &CudaStream,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = RmsNormKernel::new(hidden_size);
    let kernel_name = kernel.name();
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("rms_norm_forward_{hidden_size}");
    let module = cache.get_or_compile(&key, &ptx)?;

    // Kernel uses warp shuffle and expects exactly 32 threads (one warp)
    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };

    // Process each batch row sequentially (kernel handles single row)
    for batch_idx in 0..batch_size {
        let row_offset = u64::from(batch_idx * hidden_size);
        let byte_offset = row_offset * std::mem::size_of::<f32>() as u64;

        // Calculate pointer offsets for this batch row
        let input_ptr = input.as_ptr() + byte_offset;
        let output_ptr = output.as_ptr() + byte_offset;
        let gamma_ptr = gamma.as_ptr();

        // Kernel signature: (input_ptr, output_ptr, gamma_ptr)
        let mut args: [*mut std::ffi::c_void; 3] = [
            &input_ptr as *const _ as *mut _,
            &output_ptr as *const _ as *mut _,
            &gamma_ptr as *const _ as *mut _,
        ];

        // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
        // matching sizes, and the kernel parameters match the expected PTX signature.
        unsafe {
            stream
                .launch_kernel(module, kernel_name, &config, &mut args)
                .map_err(|e| {
                    CudaTensorError::KernelError(format!("RMSNorm forward launch failed: {e:?}"))
                })?;
        }
    }

    Ok(())
}

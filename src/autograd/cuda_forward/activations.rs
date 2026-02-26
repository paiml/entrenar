#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{GeluKernel, Kernel, ReluKernel, SiluKernel, SoftmaxKernel};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

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
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = ReluKernel::new(n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("relu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "relu", &config, &mut args).map_err(|e| {
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
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = SoftmaxKernel::new(length);
    let kernel_name = kernel.name();
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("softmax_forward_{length}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig { grid: (1, 1, 1), block: (32.min(length), 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &length as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, kernel_name, &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("Softmax forward launch failed: {e:?}"))
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
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = GeluKernel::new(n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("gelu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "gelu", &config, &mut args).map_err(|e| {
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
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let kernel = SiluKernel::new(n);
    let ptx = kernel.emit_ptx_for_target(cache.sm_target());

    let key = format!("silu_forward_{n}");
    let module = cache.get_or_compile(&key, &ptx)?;

    let config = LaunchConfig { grid: (n.div_ceil(256), 1, 1), block: (256, 1, 1), shared_mem: 0 };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: [*mut std::ffi::c_void; 3] = [
        &input_ptr as *const _ as *mut _,
        &output_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream.launch_kernel(module, "silu", &config, &mut args).map_err(|e| {
            CudaTensorError::KernelError(format!("SiLU forward launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

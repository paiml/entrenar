#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{FusedSwigluKernel, GemmKernel, Kernel};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

#[cfg(feature = "cuda")]
use super::cache::FORWARD_KERNEL_CACHE;

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
    let mut cache = cache.lock().map_err(|_err| {
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

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
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
    let mut cache = cache.lock().map_err(|_err| {
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

    // SAFETY: Kernel launch requires FFI. All buffers are valid GPU allocations with
    // matching sizes, and the kernel parameters match the expected PTX signature.
    unsafe {
        stream
            .launch_kernel(module, "gemm_naive", &config, &mut args)
            .map_err(|e| {
                CudaTensorError::KernelError(format!("GEMM forward launch failed: {e:?}"))
            })?;
    }

    Ok(())
}

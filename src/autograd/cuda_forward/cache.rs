#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    Batched4DGemmKernel, BatchedSoftmaxKernel, BatchedToInterleavedKernel,
    BatchedTransposeKernel, FusedSwigluKernel, GemmKernel, InterleavedToBatchedKernel, Kernel,
    Nf4GemmKernel, ResidualAddKernel, RmsNormKernel, ScaleKernel,
};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for forward kernels
#[cfg(feature = "cuda")]
pub(super) static FORWARD_KERNEL_CACHE: OnceLock<Mutex<ForwardKernelCache>> = OnceLock::new();

/// Cache for compiled forward kernel modules
///
/// Stores the device's SM target (e.g. "sm_89") detected at init time.
/// All PTX must be emitted for this target before compilation.
///
/// # Contract: F-PTX-001 (Target Parity)
///
/// PTX `.target` directive MUST match the device compute capability.
/// The cache validates this at compile time and rejects mismatched PTX.
#[cfg(feature = "cuda")]
pub(super) struct ForwardKernelCache {
    ctx: std::sync::Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
    /// Device SM target string (e.g. "sm_89" for RTX 4090)
    sm_target: String,
}

#[cfg(feature = "cuda")]
impl ForwardKernelCache {
    pub(super) fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
        // Detect device compute capability at construction time.
        // Falls back to sm_70 if detection fails (should never happen
        // since we already have a valid CudaContext).
        let sm_target = ctx.sm_target().unwrap_or_else(|_| "sm_70".to_string());
        eprintln!("[CUDA] Kernel cache initialized for target: {sm_target}");
        Self { ctx, modules: HashMap::new(), sm_target }
    }

    /// Get the device SM target for PTX emission.
    ///
    /// Consumers MUST use this to emit PTX via `kernel.emit_ptx_for_target(cache.sm_target())`.
    pub(super) fn sm_target(&self) -> &str {
        &self.sm_target
    }

    /// Compile PTX and cache the resulting module.
    ///
    /// # Contract: F-PTX-001 (Target Parity)
    ///
    /// Validates that the PTX `.target` directive matches the device's compute
    /// capability. Rejects PTX compiled for the wrong architecture.
    pub(super) fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        use std::collections::hash_map::Entry;

        // F-PTX-001: Validate PTX target matches device
        if let Some(target_line) = ptx.lines().find(|l| l.starts_with(".target ")) {
            let ptx_target = target_line.trim().trim_start_matches(".target ");
            if ptx_target != self.sm_target {
                return Err(CudaTensorError::KernelError(format!(
                    "F-PTX-001 violated: PTX target '{ptx_target}' != device target '{}'. \
                     Use kernel.emit_ptx_for_target(\"{}\") instead of emit_ptx().",
                    self.sm_target, self.sm_target
                )));
            }
        }

        match self.modules.entry(name.to_string()) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let module = CudaModule::from_ptx(&self.ctx, ptx).map_err(|err| {
                    CudaTensorError::KernelError(format!("Failed to compile {name}: {err:?}"))
                })?;
                Ok(e.insert(module))
            }
        }
    }

    /// Pre-warm all kernels needed for transformer forward pass.
    ///
    /// # Contract: C-PREWARM-001 (JIT Before Payload)
    ///
    /// - **Precondition**: Kernel cache initialized, GPU VRAM mostly free (no blocks uploaded yet)
    /// - **Postcondition**: All forward-pass PTX modules JIT-compiled and cached
    /// - **Invariant**: Subsequent `get_or_compile()` calls for these keys hit cache (zero JIT)
    ///
    /// CUDA's `cuModuleLoadDataEx` JIT compiler needs device memory for compilation.
    /// If called after uploading 36 transformer blocks (~22 GB), the near-OOM state causes
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` during JIT (trueno#107). Pre-warming compiles all PTX
    /// while VRAM is free, avoiding this failure mode entirely.
    pub(super) fn pre_warm_for_model(
        &mut self,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<()> {
        let s = max_seq_len as u32;
        let h = hidden_size as u32;
        let kv_h = (num_kv_heads * head_dim) as u32;
        let i = intermediate_size as u32;
        let nh = num_heads as u32;
        let nkv = num_kv_heads as u32;
        let hd = head_dim as u32;
        let sh = s * h;   // seq_len * hidden_size
        let si = s * i;   // seq_len * intermediate_size

        let mut count = 0u32;
        let target = self.sm_target.clone();

        // Helper: generate PTX and compile
        macro_rules! warm {
            ($key:expr, $kernel:expr) => {{
                let ptx = $kernel.emit_ptx_for_target(&target);
                self.get_or_compile(&$key, &ptx)?;
                count += 1;
            }};
        }

        // 1. RMSNorm (called with hidden_size)
        warm!(format!("rms_norm_forward_{h}"), RmsNormKernel::new(h));

        // 2. GEMM: Q/O projections (S, H, H)
        warm!(format!("gemm_forward_{s}_{h}_{h}"), GemmKernel::naive(s, h, h));

        // 3. GEMM: K/V projections (S, H, kv_hidden)
        if kv_h != h {
            warm!(format!("gemm_forward_{s}_{h}_{kv_h}"), GemmKernel::naive(s, kv_h, h));
        }

        // 4. GEMM: gate/up projections (S, H, I)
        warm!(format!("gemm_forward_{s}_{h}_{i}"), GemmKernel::naive(s, i, h));

        // 5. GEMM: down projection (S, I, H)
        warm!(format!("gemm_forward_{s}_{i}_{h}"), GemmKernel::naive(s, h, i));

        // 6. Fused SwiGLU
        warm!(format!("fused_swiglu_forward_{si}"), FusedSwigluKernel::new(si));

        // 7. Residual add (seq * hidden)
        warm!(format!("residual_add_forward_{sh}"), ResidualAddKernel::new(sh));

        // 8. Interleaved-to-batched: Q (S, NH, HD) and K/V (S, NKV, HD)
        warm!(
            format!("interleaved_to_batched_{s}_{nh}_{hd}"),
            InterleavedToBatchedKernel::new(s, nh, hd)
        );
        if nkv != nh {
            warm!(
                format!("interleaved_to_batched_{s}_{nkv}_{hd}"),
                InterleavedToBatchedKernel::new(s, nkv, hd)
            );
        }

        // 9. Batched transpose: K^T (NH, S, HD)
        warm!(
            format!("batched_transpose_{nh}_{s}_{hd}"),
            BatchedTransposeKernel::new(nh, s, hd)
        );

        // 10. Batched 4D GEMM: Q@K^T (1, NH, S, S, HD)
        warm!(
            format!("batched_4d_gemm_1_{nh}_{s}_{s}_{hd}"),
            Batched4DGemmKernel::new(1, nh, s, s, hd)
        );

        // 11. Scale: attention scores (NH * S * S)
        let score_n = nh * s * s;
        warm!(format!("scale_forward_{score_n}"), ScaleKernel::new(score_n));

        // 12. Batched softmax: (NH * S rows, S cols)
        let softmax_rows = nh * s;
        warm!(
            format!("batched_softmax_forward_{softmax_rows}_{s}"),
            BatchedSoftmaxKernel::new(softmax_rows, s)
        );

        // 13. Batched 4D GEMM: attn@V (1, NH, S, HD, S)
        warm!(
            format!("batched_4d_gemm_1_{nh}_{s}_{hd}_{s}"),
            Batched4DGemmKernel::new(1, nh, s, hd, s)
        );

        // 14. Batched-to-interleaved: attention output (S, NH, HD)
        warm!(
            format!("batched_to_interleaved_{s}_{nh}_{hd}"),
            BatchedToInterleavedKernel::new(s, nh, hd)
        );

        // 15-18. NF4 quantized GEMM variants (trueno#108: QLoRA support)
        // Same 4 GEMM shapes but with Nf4GemmKernel instead of GemmKernel.
        // Only compiled if K is divisible by 64 (NF4 block size).
        if h % 64 == 0 {
            warm!(format!("nf4_gemm_forward_{s}_{h}_{h}"), Nf4GemmKernel::new(s, h, h));
            if kv_h != h && kv_h % 64 == 0 {
                warm!(format!("nf4_gemm_forward_{s}_{h}_{kv_h}"), Nf4GemmKernel::new(s, kv_h, h));
            }
            if i % 64 == 0 {
                warm!(format!("nf4_gemm_forward_{s}_{h}_{i}"), Nf4GemmKernel::new(s, i, h));
                warm!(format!("nf4_gemm_forward_{s}_{i}_{h}"), Nf4GemmKernel::new(s, h, i));
            }
        }

        eprintln!("[CUDA] Pre-warmed {count} forward kernels (JIT compiled before block upload)");
        Ok(())
    }
}

/// Initialize forward kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_forward_kernel_cache(ctx: std::sync::Arc<CudaContext>) -> Result<()> {
    FORWARD_KERNEL_CACHE.get_or_init(|| Mutex::new(ForwardKernelCache::new(ctx)));
    Ok(())
}

/// Pre-warm forward kernels for a specific model configuration.
///
/// # Contract: C-PREWARM-001 (JIT Before Payload)
///
/// Must be called AFTER `init_forward_kernel_cache()` and BEFORE uploading
/// transformer blocks to GPU. JIT-compiles all PTX modules that the forward
/// pass will need, while VRAM is still free.
#[cfg(feature = "cuda")]
pub fn pre_warm_forward_kernels(
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE
        .get()
        .ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    cache.pre_warm_for_model(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
    )
}

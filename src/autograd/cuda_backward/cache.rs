#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule};

use super::super::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for backward kernels
#[cfg(feature = "cuda")]
pub(super) static KERNEL_CACHE: OnceLock<Mutex<KernelCache>> = OnceLock::new();

/// Cache for compiled backward kernel modules
///
/// # Contract: F-PTX-001 (Target Parity)
///
/// Same invariant as forward cache — PTX target must match device.
#[cfg(feature = "cuda")]
pub(super) struct KernelCache {
    ctx: Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
    sm_target: String,
}

#[cfg(feature = "cuda")]
impl KernelCache {
    pub(super) fn new(ctx: Arc<CudaContext>) -> Self {
        let sm_target = ctx.sm_target().unwrap_or_else(|_| "sm_70".to_string());
        Self { ctx, modules: HashMap::new(), sm_target }
    }

    pub(super) fn sm_target(&self) -> &str {
        &self.sm_target
    }

    /// Look up a previously compiled module by key (KAIZEN-058).
    pub(super) fn get_cached(&mut self, name: &str) -> Option<&mut CudaModule> {
        self.modules.get_mut(name)
    }

    pub(super) fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        use std::collections::hash_map::Entry;

        // F-PTX-001: Validate PTX target matches device
        if let Some(target_line) = ptx.lines().find(|l| l.starts_with(".target ")) {
            let ptx_target = target_line.trim().trim_start_matches(".target ");
            if ptx_target != self.sm_target {
                return Err(CudaTensorError::KernelError(format!(
                    "F-PTX-001 violated: PTX target '{ptx_target}' != device target '{}'",
                    self.sm_target
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
}

/// Initialize kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_kernel_cache(ctx: Arc<CudaContext>) -> Result<()> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(KernelCache::new(ctx)));
    Ok(())
}

/// Pre-warm backward GEMM kernels for training gradient computation (ENT-153).
///
/// Covers both LoRA-only shapes (NF4 QLoRA) and full fp32 backward shapes.
///
/// ## LoRA backward shapes
/// `gemm_backward_b` (weight grads): `(S,R,qd)`, `(S,R,kv)`, `(S,H,R)`
/// `gemm_backward_a` (input grads): `(S,qd,R)`, `(S,kv,R)`, `(S,R,H)`
///
/// ## Full fp32 backward shapes (non-NF4)
/// `gemm_backward_a`: `(S,I,H)` down, `(S,H,I)` gate/up, `(S,H,H)` Q/O, `(S,kv,H)` K/V
/// `gemm_backward_b`: `(S,I,H)` grad_w_down, `(S,H,I)` grad_w_gate/up,
///                     `(S,H,H)` grad_w_q/o, `(S,kv,H)` grad_w_k/v
///
/// All must be JIT-compiled before block upload fills VRAM (C-PREWARM-001).
#[cfg(feature = "cuda")]
pub fn pre_warm_lora_backward_kernels(
    hidden_size: usize,
    q_dim: usize,
    kv_hidden_size: usize,
    max_seq_len: usize,
    lora_rank: usize,
    intermediate_size: usize,
    num_heads: usize,
    quantize_nf4: bool,
) -> Result<()> {
    use trueno_gpu::kernels::backward::{
        BatchedRmsNormBackwardKernel, BatchedSoftmaxBackwardKernel, GemmBackwardAKernel,
        GemmBackwardBKernel, SiluBackwardKernel,
    };
    use trueno_gpu::kernels::Kernel;

    if lora_rank == 0 {
        return Ok(());
    }

    let cache = KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire backward kernel cache lock".to_string())
    })?;

    let s = max_seq_len as u32;
    let h = hidden_size as u32;
    let r = lora_rank as u32;
    let qd = q_dim as u32;
    let kv = kv_hidden_size as u32;
    let i = intermediate_size as u32;
    let nh = num_heads as u32;

    let mut count = 0u32;
    let target = cache.sm_target().to_string();

    macro_rules! warm {
        ($key:expr, $kernel:expr) => {{
            let ptx = $kernel.emit_ptx_for_target(&target);
            cache.get_or_compile(&$key, &ptx)?;
            count += 1;
        }};
    }

    // Tile size must match BACKWARD_TILE_SIZE in gemm.rs (C-TILE-BWD-007)
    let tile: u32 = 16;

    // ── LoRA backward shapes (always needed) ──
    // gemm_backward_b: weight gradients
    warm!(
        format!("gemm_backward_b_{s}_{r}_{qd}"),
        GemmBackwardBKernel::tiled_unrolled(s, r, qd, tile)
    );
    if kv != qd {
        warm!(
            format!("gemm_backward_b_{s}_{r}_{kv}"),
            GemmBackwardBKernel::tiled_unrolled(s, r, kv, tile)
        );
    }
    warm!(
        format!("gemm_backward_b_{s}_{h}_{r}"),
        GemmBackwardBKernel::tiled_unrolled(s, h, r, tile)
    );

    // gemm_backward_a: input gradients
    warm!(
        format!("gemm_backward_a_{s}_{qd}_{r}"),
        GemmBackwardAKernel::tiled_unrolled(s, qd, r, tile)
    );
    if kv != qd {
        warm!(
            format!("gemm_backward_a_{s}_{kv}_{r}"),
            GemmBackwardAKernel::tiled_unrolled(s, kv, r, tile)
        );
    }
    warm!(
        format!("gemm_backward_a_{s}_{r}_{h}"),
        GemmBackwardAKernel::tiled_unrolled(s, r, h, tile)
    );

    // ── Full fp32 backward shapes (non-NF4 mode) ──
    if !quantize_nf4 {
        // Attention backward: Q/O (S,H,H), K/V (S,kv,H)
        warm!(
            format!("gemm_backward_a_{s}_{h}_{h}"),
            GemmBackwardAKernel::tiled_unrolled(s, h, h, tile)
        );
        warm!(
            format!("gemm_backward_b_{s}_{h}_{h}"),
            GemmBackwardBKernel::tiled_unrolled(s, h, h, tile)
        );
        if kv != h {
            warm!(
                format!("gemm_backward_a_{s}_{kv}_{h}"),
                GemmBackwardAKernel::tiled_unrolled(s, kv, h, tile)
            );
            warm!(
                format!("gemm_backward_b_{s}_{kv}_{h}"),
                GemmBackwardBKernel::tiled_unrolled(s, kv, h, tile)
            );
        }

        // FFN backward: gate/up (S,H,I), down (S,I,H)
        warm!(
            format!("gemm_backward_a_{s}_{h}_{i}"),
            GemmBackwardAKernel::tiled_unrolled(s, h, i, tile)
        );
        warm!(
            format!("gemm_backward_b_{s}_{h}_{i}"),
            GemmBackwardBKernel::tiled_unrolled(s, h, i, tile)
        );
        warm!(
            format!("gemm_backward_a_{s}_{i}_{h}"),
            GemmBackwardAKernel::tiled_unrolled(s, i, h, tile)
        );
        warm!(
            format!("gemm_backward_b_{s}_{i}_{h}"),
            GemmBackwardBKernel::tiled_unrolled(s, i, h, tile)
        );
    }

    // ── Activation backward: SiLU ──
    let si = s * i;
    warm!("silu_backward".to_string(), SiluBackwardKernel::new(si));

    // ── Structured backward kernels (attention + normalization) ──
    // Batched softmax backward: (num_heads * seq_len, seq_len)
    let softmax_rows = nh * s;
    warm!(
        format!("batched_softmax_backward_{softmax_rows}_{s}"),
        BatchedSoftmaxBackwardKernel::new(softmax_rows, s)
    );

    // RMSNorm backward: (seq_len, hidden_size) — called twice per block
    let eps = 1e-5_f32;
    warm!(
        format!("batched_rms_norm_backward_{s}_{h}"),
        BatchedRmsNormBackwardKernel::new(s, h, eps)
    );

    let _ = count;
    Ok(())
}

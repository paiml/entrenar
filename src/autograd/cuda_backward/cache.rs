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

/// Pre-warm backward GEMM kernels for QLoRA LoRA gradient computation (ENT-153).
///
/// The LoRA backward uses both `gemm_backward_b` (weight gradients) and
/// `gemm_backward_a` (input gradients through LoRA):
///
/// `gemm_backward_b` shapes (weight grads):
/// - grad_B_q: `(S, R, qd)` — lora_inter_q^T @ grad_q
/// - grad_B_v: `(S, R, kv)` — lora_inter_v^T @ grad_v
/// - grad_A_q/v: `(S, H, R)` — norm_out^T @ grad_lora_inter
///
/// `gemm_backward_a` shapes (input grads through LoRA):
/// - grad through B_q: `(S, qd, R)` — grad_q @ B_q^T
/// - grad through B_v: `(S, kvh, R)` — grad_v @ B_v^T
/// - LoRA add to norm1: `(S, R, H)` — lora_inter @ A^T
///
/// All must be JIT-compiled before block upload fills VRAM (C-PREWARM-001).
#[cfg(feature = "cuda")]
pub fn pre_warm_lora_backward_kernels(
    hidden_size: usize,
    q_dim: usize,
    kv_hidden_size: usize,
    max_seq_len: usize,
    lora_rank: usize,
) -> Result<()> {
    use trueno_gpu::kernels::backward::{GemmBackwardAKernel, GemmBackwardBKernel};
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

    // ── gemm_backward_b: weight gradients (tiled_unrolled, trueno#109) ──
    // grad_B_q = lora_inter_q^T[R,S] @ grad_q[S,qd] → [R,qd]
    warm!(format!("gemm_backward_b_{s}_{r}_{qd}"), GemmBackwardBKernel::tiled_unrolled(s, r, qd, tile));

    // grad_B_v = lora_inter_v^T[R,S] @ grad_v[S,kv] → [R,kv]
    if kv != qd {
        warm!(format!("gemm_backward_b_{s}_{r}_{kv}"), GemmBackwardBKernel::tiled_unrolled(s, r, kv, tile));
    }

    // grad_A_q/v = norm_out^T[H,S] @ grad_li[S,R] → [H,R]
    warm!(format!("gemm_backward_b_{s}_{h}_{r}"), GemmBackwardBKernel::tiled_unrolled(s, h, r, tile));

    // ── gemm_backward_a: input gradients through LoRA (tiled_unrolled, trueno#109) ──
    // grad through B_q: grad_q[S,qd] @ B_q[R,qd] → grad_lora_inter[S,R]
    warm!(format!("gemm_backward_a_{s}_{qd}_{r}"), GemmBackwardAKernel::tiled_unrolled(s, qd, r, tile));

    // grad through B_v: grad_v[S,kvh] @ B_v[R,kvh] → grad_lora_inter[S,R]
    if kv != qd {
        warm!(format!("gemm_backward_a_{s}_{kv}_{r}"), GemmBackwardAKernel::tiled_unrolled(s, kv, r, tile));
    }

    // LoRA add to grad_norm1: lora_inter[S,R] @ A[H,R] → lora_temp[S,H]
    warm!(format!("gemm_backward_a_{s}_{r}_{h}"), GemmBackwardAKernel::tiled_unrolled(s, r, h, tile));

    let _ = count;
    Ok(())
}

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
#[cfg(feature = "cuda")]
pub(super) struct KernelCache {
    ctx: Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
}

#[cfg(feature = "cuda")]
impl KernelCache {
    pub(super) fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            modules: HashMap::new(),
        }
    }

    pub(super) fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        use std::collections::hash_map::Entry;
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

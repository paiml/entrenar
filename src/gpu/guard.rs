//! VRAM Guard (GPU-SHARE-002).
//!
//! Pre-allocation check enforcing Contract C-VRAM-001:
//! `CudaTrainer::new()` MUST NOT allocate if budget exceeds available VRAM.
//!
//! # Usage
//!
//! ```ignore
//! let guard = VramGuard::acquire(budget_mb, "qlora-7b")?;
//! // ... create CudaTrainer, allocate GPU memory ...
//! guard.update_actual(actual_mb)?;
//! // guard releases on Drop
//! ```

use crate::trace::{TraceStep, TRACER};

use super::error::GpuError;
use super::ledger::VramLedger;
use super::profiler::GpuProfiler;
use super::wait::{self, WaitConfig};

/// VRAM reservation guard.
///
/// Acquires a VRAM reservation on creation and releases it on drop.
/// Enforces C-VRAM-001: no allocation beyond budget.
pub struct VramGuard {
    ledger: VramLedger,
    budget_mb: usize,
}

impl VramGuard {
    /// Acquire a VRAM reservation.
    ///
    /// Checks the ledger and reserves `budget_mb` of VRAM.
    /// Returns `GpuError::InsufficientMemory` if not enough VRAM available.
    pub fn acquire(budget_mb: usize, task: &str) -> Result<Self, GpuError> {
        TRACER.span(
            TraceStep::VramQuery,
            format!("guard_acquire budget={budget_mb}MB"),
            || {
                let mut ledger = super::ledger::auto_ledger();
                ledger.try_reserve(budget_mb, task)?;
                Ok(Self { ledger, budget_mb })
            },
        )
    }

    /// Acquire with waiting: poll until VRAM is available or timeout.
    pub fn acquire_wait(
        budget_mb: usize,
        task: &str,
        timeout_secs: u64,
    ) -> Result<Self, GpuError> {
        TRACER.span(
            TraceStep::WaitPoll,
            format!("guard_wait budget={budget_mb}MB timeout={timeout_secs}s"),
            || {
                let mut ledger = super::ledger::auto_ledger();
                let config = WaitConfig::with_timeout_secs(timeout_secs);
                let mut profiler = GpuProfiler::disabled();
                wait::wait_for_vram(&mut ledger, budget_mb, task, &config, &mut profiler)?;
                Ok(Self { ledger, budget_mb })
            },
        )
    }

    /// Update the actual measured VRAM after GPU initialization.
    ///
    /// Call this after `CudaTrainer::new()` + weight upload to record
    /// the real VRAM usage (may be less than budgeted).
    pub fn update_actual(&mut self, actual_mb: usize) -> Result<(), GpuError> {
        self.ledger.update_actual(actual_mb)
    }

    /// Budget that was reserved (MB).
    pub fn budget_mb(&self) -> usize {
        self.budget_mb
    }

    /// GPU UUID this guard is for.
    pub fn gpu_uuid(&self) -> &str {
        &self.ledger.gpu_uuid
    }

    /// Read the current GPU status display.
    pub fn status(&self) -> Result<String, GpuError> {
        super::ledger::gpu_status_display(&self.ledger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn test_guard_ledger(total_mb: usize) -> VramLedger {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-guard-test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join(format!("guard-{n}-{}.json", std::process::id()));
        VramLedger::new("GPU-test-guard".into(), total_mb, 0.85).with_path(path)
    }

    #[test]
    fn test_guard_direct_acquire() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(5000, "guard-test").expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 5000);
        // Drop releases
        drop(ledger);
    }

    #[test]
    fn test_guard_update_actual() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(8000, "guard-actual").expect("should succeed");
        ledger.update_actual(7200).expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 7200);
    }

    #[test]
    fn test_guard_rejects_over_budget() {
        let mut ledger = test_guard_ledger(10000);
        let result = ledger.try_reserve(9000, "too-big");
        assert!(result.is_err());
    }
}

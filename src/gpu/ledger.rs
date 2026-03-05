//! VRAM Reservation Ledger (GPU-SHARE-001).
//!
//! Uses flock for mutual exclusion and atomic write (write-to-temp, rename)
//! for crash safety.
//!
//! # Contract C-VRAM-001
//!
//! `CudaTrainer::new()` MUST NOT allocate if
//! `ledger.total_reserved() + budget > total_mb × reserve_factor`.
//!
//! # Protocol
//!
//! 1. Acquire `flock(LOCK_EX)` on ledger file
//! 2. Read reservations, prune dead PIDs + expired leases
//! 3. Check capacity: `sum(active.budget_mb) + my_budget <= total_mb × reserve_factor`
//! 4. Write reservation via atomic rename (write tmp → rename)
//! 5. Release lock (close fd / drop)
//! 6. On exit: best-effort cleanup via `Drop`

use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};

use super::error::GpuError;
use super::profiler::GpuProfiler;
use crate::trace::{TraceStep, TRACER};

/// Default ledger location: `~/.cache/entrenar/gpu-ledger.json`.
fn default_ledger_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("entrenar")
        .join("gpu-ledger.json")
}

/// Reserve factor for discrete GPUs (15% headroom).
pub const RESERVE_FACTOR_DISCRETE: f32 = 0.85;

/// Reserve factor for unified memory (40% headroom for OS).
pub const RESERVE_FACTOR_UNIFIED: f32 = 0.60;

/// Default lease duration (24 hours).
pub const DEFAULT_LEASE_HOURS: i64 = 24;

/// A single VRAM reservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reservation {
    /// Unique reservation ID.
    pub id: u64,
    /// Process ID of the holder.
    pub pid: u32,
    /// Budgeted VRAM in MB.
    pub budget_mb: usize,
    /// Actual measured VRAM in MB (updated post-init).
    pub actual_mb: Option<usize>,
    /// Human-readable task description.
    pub task: String,
    /// GPU UUID this reservation is for.
    pub gpu_uuid: String,
    /// When the reservation was created.
    pub started: DateTime<Utc>,
    /// When the lease automatically expires.
    pub lease_expires: DateTime<Utc>,
}

impl Reservation {
    /// Whether this reservation's lease has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.lease_expires
    }

    /// Whether the holding process is still alive (Linux /proc check).
    pub fn is_alive(&self) -> bool {
        Path::new(&format!("/proc/{}/stat", self.pid)).exists()
    }

    /// Whether this reservation should be pruned.
    pub fn should_prune(&self) -> bool {
        self.is_expired() || !self.is_alive()
    }
}

/// Ledger file contents.
#[derive(Debug, Default, Serialize, Deserialize)]
struct LedgerData {
    reservations: Vec<Reservation>,
}

impl LedgerData {
    /// Remove dead PIDs and expired leases in-place.
    fn prune_dead(&mut self) {
        self.reservations.retain(|r| !r.should_prune());
    }

    /// Sum of reserved VRAM for a specific GPU.
    fn total_reserved_for(&self, gpu_uuid: &str) -> usize {
        self.reservations
            .iter()
            .filter(|r| r.gpu_uuid == gpu_uuid)
            .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
            .sum()
    }
}

/// VRAM reservation ledger with flock-based mutual exclusion.
pub struct VramLedger {
    path: PathBuf,
    /// GPU UUID (from nvidia-smi -L).
    pub gpu_uuid: String,
    /// Total GPU memory in MB.
    pub total_mb: usize,
    /// Fraction of total usable (0.85 discrete, 0.60 unified).
    pub reserve_factor: f32,
    lease_hours: i64,
    /// Profiler for brick-phase timing.
    profiler: GpuProfiler,
    /// Our reservation ID if we hold one.
    pub our_reservation_id: Option<u64>,
}

impl VramLedger {
    /// Create a new ledger for the specified GPU.
    pub fn new(gpu_uuid: String, total_mb: usize, reserve_factor: f32) -> Self {
        Self {
            path: default_ledger_path(),
            gpu_uuid,
            total_mb,
            reserve_factor,
            lease_hours: DEFAULT_LEASE_HOURS,
            profiler: GpuProfiler::disabled(),
            our_reservation_id: None,
        }
    }

    /// Create a ledger at a custom path (for testing).
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    /// Enable profiling.
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiler = GpuProfiler::new(enabled);
        self
    }

    /// Set custom lease duration.
    pub fn with_lease_hours(mut self, hours: i64) -> Self {
        self.lease_hours = hours;
        self
    }

    /// Usable VRAM capacity in MB.
    pub fn capacity_mb(&self) -> usize {
        (self.total_mb as f32 * self.reserve_factor) as usize
    }

    /// Total reserved VRAM across all active reservations for our GPU.
    pub fn total_reserved(&self) -> Result<usize, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        self.with_lock_read(|data| {
            data.reservations
                .iter()
                .filter(|r| r.gpu_uuid == gpu_uuid && !r.should_prune())
                .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
                .sum()
        })
    }

    /// Available VRAM for new reservations (capacity - reserved).
    pub fn available_mb(&self) -> Result<usize, GpuError> {
        let reserved = self.total_reserved()?;
        Ok(self.capacity_mb().saturating_sub(reserved))
    }

    /// Try to reserve VRAM. Returns reservation ID on success.
    ///
    /// # Contract C-VRAM-001
    ///
    /// Fails with `GpuError::InsufficientMemory` if
    /// `total_reserved + budget_mb > capacity_mb`.
    pub fn try_reserve(&mut self, budget_mb: usize, task: &str) -> Result<u64, GpuError> {
        TRACER.span(
            TraceStep::LedgerReserve,
            format!("ledger_reserve budget={budget_mb}MB gpu={}", self.gpu_uuid),
            || self.try_reserve_inner(budget_mb, task),
        )
    }

    fn try_reserve_inner(&mut self, budget_mb: usize, task: &str) -> Result<u64, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        let lease_hours = self.lease_hours;
        let capacity = self.capacity_mb();
        let total_mb = self.total_mb;

        let id = self.with_lock_write(|data| {
            data.prune_dead();

            let reserved = data.total_reserved_for(&gpu_uuid);

            if reserved + budget_mb > capacity {
                return Err(GpuError::InsufficientMemory {
                    budget_mb,
                    available_mb: capacity.saturating_sub(reserved),
                    reserved_mb: reserved,
                    total_mb,
                });
            }

            let now = Utc::now();
            let id = reservation_id(&gpu_uuid, std::process::id(), now);
            let reservation = Reservation {
                id,
                pid: std::process::id(),
                budget_mb,
                actual_mb: None,
                task: task.to_string(),
                gpu_uuid: gpu_uuid.clone(),
                started: now,
                lease_expires: now + chrono::Duration::hours(lease_hours),
            };

            data.reservations.push(reservation);
            Ok(id)
        })?;

        self.our_reservation_id = Some(id);
        self.profiler.finish_op();
        Ok(id)
    }

    /// Update the actual measured VRAM for our reservation.
    pub fn update_actual(&mut self, actual_mb: usize) -> Result<(), GpuError> {
        let Some(our_id) = self.our_reservation_id else {
            return Ok(());
        };

        self.with_lock_write(|data| {
            if let Some(r) = data.reservations.iter_mut().find(|r| r.id == our_id) {
                r.actual_mb = Some(actual_mb);
            }
            Ok(())
        })
    }

    /// Release our reservation.
    pub fn release(&mut self) -> Result<(), GpuError> {
        let Some(our_id) = self.our_reservation_id.take() else {
            return Ok(());
        };

        TRACER.span(
            TraceStep::LedgerRelease,
            format!("ledger_release id={our_id}"),
            || {
                self.with_lock_write(|data| {
                    data.reservations.retain(|r| r.id != our_id);
                    Ok(())
                })
            },
        )
    }

    /// Read all reservations for our GPU (pruned).
    pub fn read_reservations(&self) -> Result<Vec<Reservation>, GpuError> {
        let gpu_uuid = self.gpu_uuid.clone();
        self.with_lock_read(|data| {
            data.reservations
                .iter()
                .filter(|r| r.gpu_uuid == gpu_uuid && !r.should_prune())
                .cloned()
                .collect()
        })
    }

    /// Get profiler report.
    pub fn profiler_report(&self) -> String {
        self.profiler.report()
    }

    // ── flock + atomic read/write ──

    /// Execute a read-only operation under flock.
    fn with_lock_read<F, T>(&self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&LedgerData) -> T,
    {
        ensure_parent_dir(&self.path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        file.lock_exclusive()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("flock: {e}"))))?;

        let data = read_ledger(&file)?;
        let result = f(&data);

        #[allow(clippy::incompatible_msrv)]
        file.unlock()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("funlock: {e}"))))?;

        Ok(result)
    }

    /// Execute a read-modify-write operation under flock with atomic write.
    fn with_lock_write<F, T>(&mut self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&mut LedgerData) -> Result<T, GpuError>,
    {
        ensure_parent_dir(&self.path)?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;

        // Phase: lock_acq
        self.profiler.begin(GpuProfiler::LOCK_ACQ);
        file.lock_exclusive()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("flock: {e}"))))?;
        self.profiler.end(GpuProfiler::LOCK_ACQ);

        // Phase: ledger_rd
        self.profiler.begin(GpuProfiler::LEDGER_RD);
        let mut data = read_ledger(&file)?;
        self.profiler.end(GpuProfiler::LEDGER_RD);

        let result = f(&mut data)?;

        // Phase: ledger_wr (atomic: write temp, fsync, rename)
        self.profiler.begin(GpuProfiler::LEDGER_WR);
        atomic_write_ledger(&self.path, &data)?;
        self.profiler.end(GpuProfiler::LEDGER_WR);

        // Phase: lock_rel
        self.profiler.begin(GpuProfiler::LOCK_REL);
        #[allow(clippy::incompatible_msrv)]
        file.unlock()
            .map_err(|e| GpuError::Io(std::io::Error::other(format!("funlock: {e}"))))?;
        self.profiler.end(GpuProfiler::LOCK_REL);

        Ok(result)
    }
}

impl Drop for VramLedger {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

// ── Helper functions ──

/// Generate a deterministic reservation ID.
fn reservation_id(gpu_uuid: &str, pid: u32, time: DateTime<Utc>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    gpu_uuid.hash(&mut hasher);
    pid.hash(&mut hasher);
    time.timestamp_nanos_opt().unwrap_or(0).hash(&mut hasher);
    hasher.finish()
}

/// Read ledger JSON from an open file. Returns empty data if file is empty.
fn read_ledger(file: &File) -> Result<LedgerData, GpuError> {
    let mut contents = String::new();
    let mut reader = file;
    if reader.read_to_string(&mut contents).is_err() || contents.trim().is_empty() {
        return Ok(LedgerData::default());
    }
    serde_json::from_str(&contents)
        .map_err(|e| GpuError::LedgerCorrupt(format!("JSON parse: {e}")))
}

/// Atomic write: write to temp file, fsync, rename over ledger.
fn atomic_write_ledger(path: &Path, data: &LedgerData) -> Result<(), GpuError> {
    let tmp_path = path.with_extension("tmp");
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| GpuError::LedgerCorrupt(format!("JSON serialize: {e}")))?;

    let mut tmp_file = File::create(&tmp_path)?;
    tmp_file.write_all(json.as_bytes())?;
    tmp_file.sync_all()?;

    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Ensure parent directory exists.
fn ensure_parent_dir(path: &Path) -> Result<(), GpuError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Detect GPU UUID by shelling out to `nvidia-smi -L`.
pub fn detect_gpu_uuid() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["-L"])
        .output()
        .ok()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines().find_map(|line| {
                let start = line.find("UUID: ")?;
                let uuid_start = start + 6;
                let end = line[uuid_start..].find(')')? + uuid_start;
                Some(line[uuid_start..end].to_string())
            })
        })
        .unwrap_or_else(|| "GPU-unknown".to_string())
}

/// Detect total GPU memory in MB via `nvidia-smi`.
pub fn detect_total_memory_mb() -> usize {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.trim().lines().next()?.trim().parse::<usize>().ok()
        })
        .unwrap_or(0)
}

/// Detect whether GPU has unified memory (Jetson) vs discrete.
pub fn detect_memory_type() -> MemoryType {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .map_or(MemoryType::Discrete, |out| {
            let name = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if name.contains("jetson") || name.contains("orin") || name.contains("tegra") {
                MemoryType::Unified
            } else {
                MemoryType::Discrete
            }
        })
}

/// GPU memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Discrete GPU (PCIe, e.g., RTX 4090). Reserve factor: 0.85.
    Discrete,
    /// Unified memory (e.g., Jetson Orin). Reserve factor: 0.60.
    Unified,
}

impl MemoryType {
    /// Reserve factor for this memory type.
    pub fn reserve_factor(self) -> f32 {
        match self {
            Self::Discrete => RESERVE_FACTOR_DISCRETE,
            Self::Unified => RESERVE_FACTOR_UNIFIED,
        }
    }
}

/// Create a ledger auto-detecting GPU properties.
pub fn auto_ledger() -> VramLedger {
    let uuid = detect_gpu_uuid();
    let total_mb = detect_total_memory_mb();
    let mem_type = detect_memory_type();
    VramLedger::new(uuid, total_mb, mem_type.reserve_factor())
}

/// Human-readable GPU status display.
pub fn gpu_status_display(ledger: &VramLedger) -> Result<String, GpuError> {
    let reservations = ledger.read_reservations()?;
    let reserved: usize = reservations
        .iter()
        .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
        .sum();

    let mut out = String::new();
    out.push_str(&format!(
        "{}: {} MB total, {:.0}% reserve factor\n",
        ledger.gpu_uuid,
        ledger.total_mb,
        ledger.reserve_factor * 100.0
    ));
    out.push_str(&format!(
        "  Capacity: {} MB usable ({} MB reserved, {} MB available)\n",
        ledger.capacity_mb(),
        reserved,
        ledger.capacity_mb().saturating_sub(reserved),
    ));

    if reservations.is_empty() {
        out.push_str("  Reservations: none\n");
    } else {
        out.push_str(&format!("  Reservations: {}\n", reservations.len()));
        for r in &reservations {
            let actual = r
                .actual_mb.map_or_else(|| "measuring...".to_string(), |a| format!("{a} MB actual"));
            let elapsed = Utc::now().signed_duration_since(r.started);
            let hours = elapsed.num_hours();
            let mins = elapsed.num_minutes() % 60;
            out.push_str(&format!(
                "    PID {}: {} MB budget / {} ({}) — {}h {}m\n",
                r.pid, r.budget_mb, actual, r.task, hours, mins
            ));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn test_ledger_path() -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-ledger-test");
        fs::create_dir_all(&dir).expect("test dir creation should succeed");
        dir.join(format!("test-ledger-{n}-{}.json", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(path.with_extension("tmp"));
    }

    #[test]
    fn test_empty_ledger_has_full_capacity() {
        let path = test_ledger_path();
        let ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        assert_eq!(ledger.capacity_mb(), 20400);
        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);
        assert_eq!(ledger.available_mb().expect("should succeed"), 20400);

        cleanup(&path);
    }

    #[test]
    fn test_reserve_and_release() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        let id = ledger.try_reserve(8000, "test-job").expect("should succeed");
        assert!(id != 0);
        assert_eq!(ledger.total_reserved().expect("should succeed"), 8000);
        assert_eq!(ledger.available_mb().expect("should succeed"), 12400);

        ledger.release().expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_capacity_invariant_prevents_overallocation() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(15000, "job-1").expect("should succeed");

        let result = ledger.try_reserve(10000, "job-2");
        assert!(result.is_err());
        match result.expect_err("should be InsufficientMemory") {
            GpuError::InsufficientMemory {
                budget_mb,
                available_mb,
                ..
            } => {
                assert_eq!(budget_mb, 10000);
                assert_eq!(available_mb, 5400);
            }
            other => panic!("expected InsufficientMemory, got {other}"),
        }

        cleanup(&path);
    }

    #[test]
    fn test_reserve_factor_limits_total() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());

        let result = ledger.try_reserve(9000, "too-big");
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn test_update_actual() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(8000, "test-job").expect("should succeed");
        ledger.update_actual(7300).expect("should succeed");

        assert_eq!(ledger.total_reserved().expect("should succeed"), 7300);

        cleanup(&path);
    }

    #[test]
    fn test_expired_lease_pruned() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85)
            .with_path(path.clone())
            .with_lease_hours(0);

        ledger
            .try_reserve(8000, "expiring-job")
            .expect("should succeed");

        std::thread::sleep(Duration::from_millis(10));

        assert_eq!(ledger.total_reserved().expect("should succeed"), 0);

        cleanup(&path);
    }

    #[test]
    fn test_atomic_write_produces_valid_json() {
        let path = test_ledger_path();
        let mut ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        ledger.try_reserve(5000, "json-test").expect("should succeed");

        let contents = fs::read_to_string(&path).expect("should read");
        let data: LedgerData =
            serde_json::from_str(&contents).expect("should parse");
        assert_eq!(data.reservations.len(), 1);
        assert_eq!(data.reservations[0].budget_mb, 5000);

        cleanup(&path);
    }

    #[test]
    fn test_gpu_status_display() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test-display".into(), 24000, 0.85)
            .with_path(path.clone());

        ledger
            .try_reserve(7000, "display-test")
            .expect("should succeed");

        let status = gpu_status_display(&ledger).expect("should succeed");
        assert!(status.contains("GPU-test-display"));
        assert!(status.contains("24000 MB total"));
        assert!(status.contains("7000 MB budget"));
        assert!(status.contains("display-test"));

        cleanup(&path);
    }

    #[test]
    fn test_memory_type_reserve_factors() {
        assert!((MemoryType::Discrete.reserve_factor() - 0.85).abs() < f32::EPSILON);
        assert!((MemoryType::Unified.reserve_factor() - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reservation_id_deterministic() {
        let now = Utc::now();
        let id1 = reservation_id("GPU-abc", 1234, now);
        let id2 = reservation_id("GPU-abc", 1234, now);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_reservation_id_varies_with_input() {
        let now = Utc::now();
        let id1 = reservation_id("GPU-abc", 1234, now);
        let id2 = reservation_id("GPU-xyz", 1234, now);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_profiling_disabled_by_default() {
        let path = test_ledger_path();
        let ledger =
            VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());

        assert!(!ledger.profiler.is_enabled());
        let report = ledger.profiler_report();
        assert!(report.contains("No operations recorded"));

        cleanup(&path);
    }

    #[test]
    fn test_profiling_enabled_records_phases() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85)
            .with_path(path.clone())
            .with_profiling(true);

        ledger
            .try_reserve(5000, "profiled-job")
            .expect("should succeed");

        let report = ledger.profiler_report();
        assert!(report.contains("lock_acq"));
        assert!(report.contains("ledger_rd"));
        assert!(report.contains("ledger_wr"));

        cleanup(&path);
    }
}

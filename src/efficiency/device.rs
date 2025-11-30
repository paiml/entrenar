//! Compute Device Abstraction (ENT-008)
//!
//! Provides hardware device detection and abstraction for CPU, GPU, TPU,
//! and Apple Silicon devices.

use serde::{Deserialize, Serialize};

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimdCapability {
    /// No SIMD support
    #[default]
    None,
    /// SSE4.1/4.2 (128-bit)
    Sse4,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
}

impl SimdCapability {
    /// Returns the vector width in bits
    pub fn vector_width_bits(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Sse4 => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon => 128,
        }
    }

    /// Detect SIMD capability of current CPU
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else if is_x86_feature_detected!("avx2") {
            Self::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            Self::Sse4
        } else {
            Self::None
        }
    }

    /// Detect SIMD capability of current CPU (ARM)
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        // NEON is mandatory on aarch64
        Self::Neon
    }

    /// Detect SIMD capability (fallback for other architectures)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn detect() -> Self {
        Self::None
    }
}

impl std::fmt::Display for SimdCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Sse4 => write!(f, "SSE4"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
            Self::Neon => write!(f, "NEON"),
        }
    }
}

/// CPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of physical cores
    pub cores: u32,
    /// Number of logical threads (with hyperthreading)
    pub threads: u32,
    /// SIMD capability
    pub simd: SimdCapability,
    /// CPU model name
    pub model: String,
    /// Cache size in bytes (L3 or total)
    pub cache_bytes: u64,
}

impl CpuInfo {
    /// Create new CPU info
    pub fn new(cores: u32, threads: u32, simd: SimdCapability, model: impl Into<String>) -> Self {
        Self {
            cores,
            threads,
            simd,
            model: model.into(),
            cache_bytes: 0,
        }
    }

    /// Set cache size
    pub fn with_cache(mut self, cache_bytes: u64) -> Self {
        self.cache_bytes = cache_bytes;
        self
    }

    /// Detect current CPU information
    pub fn detect() -> Self {
        // Get logical CPU count using standard library
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        // Estimate physical cores (assumes hyperthreading with 2 threads per core)
        // This is a heuristic - for accurate counts would need platform-specific APIs
        let cores = Self::detect_physical_cores().unwrap_or_else(|| threads.max(1));
        let simd = SimdCapability::detect();

        // Try to get CPU model name
        let model = Self::detect_model();

        Self {
            cores,
            threads,
            simd,
            model,
            cache_bytes: 0, // Would need platform-specific APIs
        }
    }

    /// Detect physical core count (Linux-specific)
    #[cfg(target_os = "linux")]
    fn detect_physical_cores() -> Option<u32> {
        std::fs::read_to_string("/proc/cpuinfo").ok().map(|info| {
            // Count unique core IDs
            let mut core_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut current_physical_id = String::new();

            for line in info.lines() {
                if line.starts_with("physical id") {
                    current_physical_id = line
                        .split(':')
                        .nth(1)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();
                } else if line.starts_with("core id") {
                    let core_id = line
                        .split(':')
                        .nth(1)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();
                    core_ids.insert(format!("{current_physical_id}-{core_id}"));
                }
            }

            if core_ids.is_empty() {
                // Fallback: count processor entries
                info.lines()
                    .filter(|line| line.starts_with("processor"))
                    .count() as u32
            } else {
                core_ids.len() as u32
            }
        })
    }

    /// Detect physical core count (macOS-specific)
    #[cfg(target_os = "macos")]
    fn detect_physical_cores() -> Option<u32> {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.physicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
    }

    /// Detect physical core count (fallback)
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn detect_physical_cores() -> Option<u32> {
        None
    }

    /// Detect CPU model name
    #[cfg(target_os = "linux")]
    fn detect_model() -> String {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(target_os = "macos")]
    fn detect_model() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn detect_model() -> String {
        "Unknown CPU".to_string()
    }

    /// Estimate memory bandwidth based on core count (rough approximation)
    pub fn estimated_memory_bandwidth_gbps(&self) -> f64 {
        // Rough estimate: ~20 GB/s per channel, assume 2 channels for desktop
        40.0 * (f64::from(self.cores) / 8.0).min(2.0)
    }
}

/// GPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// Video RAM in bytes
    pub vram_bytes: u64,
    /// CUDA compute capability (major, minor) if NVIDIA
    pub compute_capability: Option<(u32, u32)>,
    /// GPU index (for multi-GPU systems)
    pub index: u32,
}

impl GpuInfo {
    /// Create new GPU info
    pub fn new(name: impl Into<String>, vram_bytes: u64) -> Self {
        Self {
            name: name.into(),
            vram_bytes,
            compute_capability: None,
            index: 0,
        }
    }

    /// Set CUDA compute capability
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.compute_capability = Some((major, minor));
        self
    }

    /// Set GPU index
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = index;
        self
    }

    /// Check if GPU supports specific CUDA compute capability
    pub fn supports_compute_capability(&self, major: u32, minor: u32) -> bool {
        self.compute_capability
            .is_some_and(|(m, n)| m > major || (m == major && n >= minor))
    }

    /// Get VRAM in GB
    pub fn vram_gb(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// TPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TpuInfo {
    /// TPU version (e.g., "v4", "v5e")
    pub version: String,
    /// Number of TPU cores
    pub cores: u32,
    /// High bandwidth memory in bytes
    pub hbm_bytes: u64,
}

impl TpuInfo {
    /// Create new TPU info
    pub fn new(version: impl Into<String>, cores: u32, hbm_bytes: u64) -> Self {
        Self {
            version: version.into(),
            cores,
            hbm_bytes,
        }
    }

    /// Get HBM in GB
    pub fn hbm_gb(&self) -> f64 {
        self.hbm_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Apple Silicon information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppleSiliconInfo {
    /// Chip model (e.g., "M1", "M2 Pro", "M3 Max")
    pub chip: String,
    /// Performance cores
    pub p_cores: u32,
    /// Efficiency cores
    pub e_cores: u32,
    /// GPU cores
    pub gpu_cores: u32,
    /// Neural Engine cores
    pub neural_cores: u32,
    /// Unified memory in bytes
    pub unified_memory_bytes: u64,
}

impl AppleSiliconInfo {
    /// Create new Apple Silicon info
    pub fn new(chip: impl Into<String>) -> Self {
        Self {
            chip: chip.into(),
            p_cores: 0,
            e_cores: 0,
            gpu_cores: 0,
            neural_cores: 16, // Default for most Apple Silicon
            unified_memory_bytes: 0,
        }
    }

    /// Set core configuration
    pub fn with_cores(mut self, p_cores: u32, e_cores: u32, gpu_cores: u32) -> Self {
        self.p_cores = p_cores;
        self.e_cores = e_cores;
        self.gpu_cores = gpu_cores;
        self
    }

    /// Set unified memory
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.unified_memory_bytes = bytes;
        self
    }

    /// Get total CPU cores
    pub fn total_cpu_cores(&self) -> u32 {
        self.p_cores + self.e_cores
    }

    /// Get unified memory in GB
    pub fn unified_memory_gb(&self) -> f64 {
        self.unified_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Detect Apple Silicon information (macOS only)
    #[cfg(target_os = "macos")]
    pub fn detect() -> Option<Self> {
        // Check if we're on Apple Silicon
        let arch = std::env::consts::ARCH;
        if arch != "aarch64" {
            return None;
        }

        // Get chip name from sysctl
        let chip = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".to_string());

        // Get core counts
        let p_cores = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let e_cores = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel1.logicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        // Get memory
        let memory = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        Some(
            Self::new(chip)
                .with_cores(p_cores, e_cores, 0) // GPU cores harder to detect
                .with_memory(memory),
        )
    }

    #[cfg(not(target_os = "macos"))]
    pub fn detect() -> Option<Self> {
        None
    }
}

/// Compute device abstraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// CPU device
    Cpu(CpuInfo),
    /// GPU device (NVIDIA, AMD, Intel)
    Gpu(GpuInfo),
    /// TPU device (Google)
    Tpu(TpuInfo),
    /// Apple Silicon with unified memory
    AppleSilicon(AppleSiliconInfo),
}

impl ComputeDevice {
    /// Auto-detect available compute devices
    pub fn detect() -> Vec<Self> {
        let mut devices = Vec::new();

        // Always detect CPU
        devices.push(Self::Cpu(CpuInfo::detect()));

        // Check for Apple Silicon
        if let Some(apple) = AppleSiliconInfo::detect() {
            devices.push(Self::AppleSilicon(apple));
        }

        // GPU detection would require platform-specific APIs (CUDA, ROCm, Metal)
        // For now, we only auto-detect CPU and Apple Silicon

        devices
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    /// Check if this is a TPU device
    pub fn is_tpu(&self) -> bool {
        matches!(self, Self::Tpu(_))
    }

    /// Check if this is Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        matches!(self, Self::AppleSilicon(_))
    }

    /// Get available memory in bytes
    pub fn memory_bytes(&self) -> u64 {
        match self {
            Self::Cpu(info) => {
                // Return system memory (approximation)
                // In real implementation, would query actual system RAM
                u64::from(info.cores) * 4 * 1024 * 1024 * 1024 // ~4GB per core estimate
            }
            Self::Gpu(info) => info.vram_bytes,
            Self::Tpu(info) => info.hbm_bytes,
            Self::AppleSilicon(info) => info.unified_memory_bytes,
        }
    }

    /// Get device name
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu(info) => &info.model,
            Self::Gpu(info) => &info.name,
            Self::Tpu(info) => &info.version,
            Self::AppleSilicon(info) => &info.chip,
        }
    }

    /// Get compute cores/units
    pub fn compute_units(&self) -> u32 {
        match self {
            Self::Cpu(info) => info.threads,
            Self::Gpu(_) => 0, // Would need more info
            Self::Tpu(info) => info.cores,
            Self::AppleSilicon(info) => info.total_cpu_cores() + info.gpu_cores,
        }
    }

    /// Estimate relative compute power (normalized, CPU = 1.0)
    pub fn relative_compute_power(&self) -> f64 {
        match self {
            Self::Cpu(info) => f64::from(info.threads) / 8.0, // Normalize to 8-thread CPU
            Self::Gpu(info) => {
                // Rough estimate based on VRAM (proxy for capability)
                10.0 * (info.vram_gb() / 8.0) // 8GB GPU ~ 10x CPU
            }
            Self::Tpu(info) => {
                // TPUs are highly optimized for matrix ops
                50.0 * (f64::from(info.cores) / 8.0)
            }
            Self::AppleSilicon(info) => {
                // P-cores are faster, E-cores are efficient
                (f64::from(info.p_cores) * 1.5 + f64::from(info.e_cores) * 0.5) / 8.0
                    + f64::from(info.gpu_cores) * 0.5
            }
        }
    }
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(info) => write!(
                f,
                "CPU: {} ({} cores, {} threads, {})",
                info.model, info.cores, info.threads, info.simd
            ),
            Self::Gpu(info) => {
                write!(f, "GPU: {} ({:.1} GB VRAM", info.name, info.vram_gb())?;
                if let Some((major, minor)) = info.compute_capability {
                    write!(f, ", SM {major}.{minor}")?;
                }
                write!(f, ")")
            }
            Self::Tpu(info) => write!(
                f,
                "TPU: {} ({} cores, {:.1} GB HBM)",
                info.version,
                info.cores,
                info.hbm_gb()
            ),
            Self::AppleSilicon(info) => write!(
                f,
                "Apple Silicon: {} ({}P+{}E cores, {} GPU cores, {:.1} GB)",
                info.chip,
                info.p_cores,
                info.e_cores,
                info.gpu_cores,
                info.unified_memory_gb()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability_vector_width() {
        assert_eq!(SimdCapability::None.vector_width_bits(), 0);
        assert_eq!(SimdCapability::Sse4.vector_width_bits(), 128);
        assert_eq!(SimdCapability::Avx2.vector_width_bits(), 256);
        assert_eq!(SimdCapability::Avx512.vector_width_bits(), 512);
        assert_eq!(SimdCapability::Neon.vector_width_bits(), 128);
    }

    #[test]
    fn test_simd_capability_display() {
        assert_eq!(format!("{}", SimdCapability::Avx2), "AVX2");
        assert_eq!(format!("{}", SimdCapability::Neon), "NEON");
    }

    #[test]
    fn test_simd_capability_detect() {
        let simd = SimdCapability::detect();
        // Should return something (even None is valid)
        let _ = simd.vector_width_bits();
    }

    #[test]
    fn test_cpu_info_new() {
        let cpu = CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel Core i9-12900K");

        assert_eq!(cpu.cores, 8);
        assert_eq!(cpu.threads, 16);
        assert_eq!(cpu.simd, SimdCapability::Avx2);
        assert_eq!(cpu.model, "Intel Core i9-12900K");
    }

    #[test]
    fn test_cpu_info_with_cache() {
        let cpu =
            CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU").with_cache(30 * 1024 * 1024); // 30 MB

        assert_eq!(cpu.cache_bytes, 30 * 1024 * 1024);
    }

    #[test]
    fn test_cpu_info_detect() {
        let cpu = CpuInfo::detect();

        // Should detect at least 1 core
        assert!(cpu.cores >= 1);
        assert!(cpu.threads >= cpu.cores);
        assert!(!cpu.model.is_empty());
    }

    #[test]
    fn test_gpu_info_new() {
        let gpu = GpuInfo::new("NVIDIA RTX 4090", 24 * 1024 * 1024 * 1024);

        assert_eq!(gpu.name, "NVIDIA RTX 4090");
        assert_eq!(gpu.vram_bytes, 24 * 1024 * 1024 * 1024);
        assert!(gpu.compute_capability.is_none());
    }

    #[test]
    fn test_gpu_info_with_compute_capability() {
        let gpu = GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024).with_compute_capability(8, 9);

        assert_eq!(gpu.compute_capability, Some((8, 9)));
        assert!(gpu.supports_compute_capability(8, 0));
        assert!(gpu.supports_compute_capability(8, 9));
        assert!(!gpu.supports_compute_capability(9, 0));
    }

    #[test]
    fn test_gpu_info_vram_gb() {
        let gpu = GpuInfo::new("Test GPU", 8 * 1024 * 1024 * 1024);
        assert!((gpu.vram_gb() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_tpu_info_new() {
        let tpu = TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024);

        assert_eq!(tpu.version, "v4");
        assert_eq!(tpu.cores, 8);
        assert!((tpu.hbm_gb() - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_apple_silicon_info() {
        let m2 = AppleSiliconInfo::new("Apple M2 Pro")
            .with_cores(8, 4, 19)
            .with_memory(32 * 1024 * 1024 * 1024);

        assert_eq!(m2.chip, "Apple M2 Pro");
        assert_eq!(m2.p_cores, 8);
        assert_eq!(m2.e_cores, 4);
        assert_eq!(m2.gpu_cores, 19);
        assert_eq!(m2.total_cpu_cores(), 12);
        assert!((m2.unified_memory_gb() - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_device_detect_returns_cpu() {
        let devices = ComputeDevice::detect();

        // Should always detect at least one CPU
        assert!(!devices.is_empty());
        assert!(devices.iter().any(|d| d.is_cpu()));
    }

    #[test]
    fn test_compute_device_is_methods() {
        let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test"));
        let gpu = ComputeDevice::Gpu(GpuInfo::new("Test GPU", 8 * 1024 * 1024 * 1024));
        let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
        let apple = ComputeDevice::AppleSilicon(AppleSiliconInfo::new("M2"));

        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());

        assert!(gpu.is_gpu());
        assert!(!gpu.is_cpu());

        assert!(tpu.is_tpu());
        assert!(!tpu.is_cpu());

        assert!(apple.is_apple_silicon());
        assert!(!apple.is_cpu());
    }

    #[test]
    fn test_compute_device_memory_bytes() {
        let gpu = ComputeDevice::Gpu(GpuInfo::new("Test", 16 * 1024 * 1024 * 1024));
        assert_eq!(gpu.memory_bytes(), 16 * 1024 * 1024 * 1024);

        let tpu = ComputeDevice::Tpu(TpuInfo::new("v4", 8, 32 * 1024 * 1024 * 1024));
        assert_eq!(tpu.memory_bytes(), 32 * 1024 * 1024 * 1024);

        let apple = ComputeDevice::AppleSilicon(
            AppleSiliconInfo::new("M2").with_memory(24 * 1024 * 1024 * 1024),
        );
        assert_eq!(apple.memory_bytes(), 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_compute_device_name() {
        let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel i9"));
        assert_eq!(cpu.name(), "Intel i9");

        let gpu = ComputeDevice::Gpu(GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024));
        assert_eq!(gpu.name(), "RTX 4090");
    }

    #[test]
    fn test_compute_device_display() {
        let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Intel i9"));
        let display = format!("{cpu}");
        assert!(display.contains("Intel i9"));
        assert!(display.contains("8 cores"));
        assert!(display.contains("AVX2"));

        let gpu = ComputeDevice::Gpu(
            GpuInfo::new("RTX 4090", 24 * 1024 * 1024 * 1024).with_compute_capability(8, 9),
        );
        let display = format!("{gpu}");
        assert!(display.contains("RTX 4090"));
        assert!(display.contains("24.0 GB"));
        assert!(display.contains("SM 8.9"));
    }

    #[test]
    fn test_compute_device_relative_power() {
        let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test"));
        let gpu = ComputeDevice::Gpu(GpuInfo::new("Test", 16 * 1024 * 1024 * 1024));

        // GPU should have higher relative power
        assert!(gpu.relative_compute_power() > cpu.relative_compute_power());
    }

    #[test]
    fn test_compute_device_serialization() {
        let cpu = ComputeDevice::Cpu(CpuInfo::new(8, 16, SimdCapability::Avx2, "Test CPU"));
        let json = serde_json::to_string(&cpu).unwrap();
        let parsed: ComputeDevice = serde_json::from_str(&json).unwrap();

        assert!(parsed.is_cpu());
        assert_eq!(parsed.name(), "Test CPU");
    }
}

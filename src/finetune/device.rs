//! Compute device detection and management
//!
//! Provides CUDA detection with automatic fallback to CPU.

use std::fmt;

/// Compute device for training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    /// CPU-only execution
    Cpu,
    /// CUDA GPU with device ID
    Cuda { device_id: usize },
}

impl ComputeDevice {
    /// Auto-detect best available device
    ///
    /// Prefers CUDA if available with sufficient memory (≥6GB).
    #[must_use]
    pub fn auto_detect() -> Self {
        if Self::cuda_available() {
            if let Some(info) = DeviceInfo::cuda_info(0) {
                if info.memory_gb >= 6.0 {
                    return Self::Cuda { device_id: 0 };
                }
            }
        }
        Self::Cpu
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn cuda_available() -> bool {
        // Check for CUDA via environment and nvidia-smi
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            return true;
        }

        // Try nvidia-smi
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Check if this device is CUDA
    #[must_use]
    pub const fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda { .. })
    }

    /// Check if this device is CPU
    #[must_use]
    pub const fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Get device ID for CUDA devices
    #[must_use]
    pub const fn device_id(&self) -> Option<usize> {
        match self {
            Self::Cuda { device_id } => Some(*device_id),
            Self::Cpu => None,
        }
    }
}

impl Default for ComputeDevice {
    fn default() -> Self {
        Self::auto_detect()
    }
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda { device_id } => write!(f, "CUDA:{device_id}"),
        }
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Total memory in GB
    pub memory_gb: f64,
    /// CUDA compute capability (major.minor)
    pub compute_capability: Option<(u32, u32)>,
    /// Driver version
    pub driver_version: Option<String>,
}

impl DeviceInfo {
    /// Get CPU info
    #[must_use]
    pub fn cpu_info() -> Self {
        let num_cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);

        Self {
            name: format!("CPU ({num_cores} cores)"),
            memory_gb: Self::system_memory_gb(),
            compute_capability: None,
            driver_version: None,
        }
    }

    /// Get CUDA device info
    #[must_use]
    pub fn cuda_info(device_id: usize) -> Option<Self> {
        // Query nvidia-smi for device info
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
                &format!("--id={device_id}"),
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(", ").collect();

        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let memory_mb: f64 = parts[1].parse().unwrap_or(0.0);
            let driver = parts[2].to_string();

            Some(Self {
                name,
                memory_gb: memory_mb / 1024.0,
                compute_capability: None, // Would need CUDA runtime to get this
                driver_version: Some(driver),
            })
        } else {
            None
        }
    }

    /// Get system RAM in GB
    fn system_memory_gb() -> f64 {
        // Read from /proc/meminfo on Linux
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0 / 1024.0;
                        }
                    }
                }
            }
        }
        16.0 // Default fallback
    }

    /// Check if device has sufficient memory for QLoRA
    #[must_use]
    pub fn sufficient_for_qlora(&self) -> bool {
        self.memory_gb >= 6.0
    }

    /// Check if device has sufficient memory for LoRA (fp16)
    #[must_use]
    pub fn sufficient_for_lora(&self) -> bool {
        self.memory_gb >= 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_device_cpu() {
        let device = ComputeDevice::Cpu;
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
        assert_eq!(device.device_id(), None);
        assert_eq!(device.to_string(), "CPU");
    }

    #[test]
    fn test_compute_device_cuda() {
        let device = ComputeDevice::Cuda { device_id: 0 };
        assert!(device.is_cuda());
        assert!(!device.is_cpu());
        assert_eq!(device.device_id(), Some(0));
        assert_eq!(device.to_string(), "CUDA:0");
    }

    #[test]
    fn test_auto_detect_returns_valid_device() {
        let device = ComputeDevice::auto_detect();
        // Should return either CPU or CUDA
        assert!(device.is_cpu() || device.is_cuda());
    }

    #[test]
    fn test_device_info_cpu() {
        let info = DeviceInfo::cpu_info();
        assert!(info.name.contains("CPU"));
        assert!(info.memory_gb > 0.0);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn test_device_default() {
        let device = ComputeDevice::default();
        // Should be valid
        assert!(device.is_cpu() || device.is_cuda());
    }

    #[test]
    fn test_sufficient_memory_checks() {
        let small = DeviceInfo {
            name: "Small GPU".into(),
            memory_gb: 4.0,
            compute_capability: None,
            driver_version: None,
        };
        assert!(!small.sufficient_for_qlora());
        assert!(!small.sufficient_for_lora());

        let medium = DeviceInfo {
            name: "Medium GPU".into(),
            memory_gb: 8.0,
            compute_capability: None,
            driver_version: None,
        };
        assert!(medium.sufficient_for_qlora());
        assert!(!medium.sufficient_for_lora());

        let large = DeviceInfo {
            name: "Large GPU".into(),
            memory_gb: 16.0,
            compute_capability: None,
            driver_version: None,
        };
        assert!(large.sufficient_for_qlora());
        assert!(large.sufficient_for_lora());
    }
}

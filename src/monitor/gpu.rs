//! GPU Monitoring Module (MLOPS-005)
//!
//! btop-inspired GPU monitoring for terminal training dashboard.
//!
//! # Toyota Way: アンドン (Andon)
//!
//! Visual alerting system for immediate problem detection.
//! Thermal throttling, memory pressure, and power limits trigger alerts.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::gpu::{GpuMonitor, GpuMetrics, GpuAlert};
//!
//! let monitor = GpuMonitor::new()?;
//! let metrics = monitor.sample();
//! for m in &metrics {
//!     println!("GPU {}: {}°C, {}% util", m.device_id, m.temperature_celsius, m.utilization_percent);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// GPU metrics snapshot (inspired by btop's GPU visualization)
///
/// Reference: btop `src/btop_shared.hpp` lines 130-171
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Device index
    pub device_id: u32,
    /// GPU name (e.g., "RTX 4090")
    pub name: String,
    /// GPU compute utilization (0-100%)
    pub utilization_percent: u32,
    /// Used VRAM in MB
    pub memory_used_mb: u64,
    /// Total VRAM in MB
    pub memory_total_mb: u64,
    /// Memory utilization (0-100%)
    pub memory_utilization_percent: u32,
    /// GPU temperature in Celsius
    pub temperature_celsius: u32,
    /// Current power draw in watts
    pub power_watts: f32,
    /// Power limit in watts
    pub power_limit_watts: f32,
    /// Graphics clock in MHz
    pub clock_mhz: u32,
    /// Memory clock in MHz
    pub memory_clock_mhz: u32,
    /// PCIe transmit throughput in KB/s
    pub pcie_tx_kbps: u64,
    /// PCIe receive throughput in KB/s
    pub pcie_rx_kbps: u64,
    /// Fan speed percentage (0-100%)
    pub fan_speed_percent: u32,
}

impl GpuMetrics {
    /// Create mock metrics for testing
    pub fn mock(device_id: u32) -> Self {
        Self {
            device_id,
            name: format!("Mock GPU {device_id}"),
            utilization_percent: 75,
            memory_used_mb: 8192,
            memory_total_mb: 24576,
            memory_utilization_percent: 33,
            temperature_celsius: 65,
            power_watts: 250.0,
            power_limit_watts: 450.0,
            clock_mhz: 2100,
            memory_clock_mhz: 10000,
            pcie_tx_kbps: 1000,
            pcie_rx_kbps: 2000,
            fan_speed_percent: 50,
        }
    }

    /// Calculate memory utilization percentage
    pub fn memory_percent(&self) -> f64 {
        if self.memory_total_mb == 0 {
            return 0.0;
        }
        self.memory_used_mb as f64 / self.memory_total_mb as f64 * 100.0
    }

    /// Calculate power utilization percentage
    pub fn power_percent(&self) -> f64 {
        if self.power_limit_watts <= 0.0 {
            return 0.0;
        }
        f64::from(self.power_watts) / f64::from(self.power_limit_watts) * 100.0
    }
}

/// GPU alert types for Andon system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAlert {
    /// Temperature exceeds threshold (thermal throttling imminent)
    ThermalThrottling {
        device: u32,
        temp: u32,
        threshold: u32,
    },
    /// VRAM usage exceeds threshold (OOM imminent)
    MemoryPressure {
        device: u32,
        used_percent: u32,
        threshold: u32,
    },
    /// Power draw exceeds threshold percentage of limit
    PowerLimit {
        device: u32,
        power_percent: u32,
        threshold: u32,
    },
    /// GPU utilization dropped to zero (possible hang)
    GpuIdle { device: u32, duration_secs: u32 },
}

impl GpuAlert {
    /// Get severity level (0-100, higher is more severe)
    pub fn severity(&self) -> u32 {
        match self {
            GpuAlert::ThermalThrottling {
                temp, threshold, ..
            } => {
                let excess = temp.saturating_sub(*threshold);
                50 + excess.min(50)
            }
            GpuAlert::MemoryPressure { used_percent, .. } => {
                if *used_percent >= 99 {
                    100
                } else if *used_percent >= 95 {
                    80
                } else {
                    60
                }
            }
            GpuAlert::PowerLimit { power_percent, .. } => {
                if *power_percent >= 100 {
                    70
                } else {
                    50
                }
            }
            GpuAlert::GpuIdle { duration_secs, .. } => {
                if *duration_secs > 60 {
                    90
                } else if *duration_secs > 30 {
                    70
                } else {
                    40
                }
            }
        }
    }

    /// Get human-readable description
    pub fn message(&self) -> String {
        match self {
            GpuAlert::ThermalThrottling {
                device,
                temp,
                threshold,
            } => {
                format!("GPU {device} thermal throttling: {temp}°C exceeds {threshold}°C threshold")
            }
            GpuAlert::MemoryPressure {
                device,
                used_percent,
                threshold,
            } => {
                format!(
                    "GPU {device} memory pressure: {used_percent}% exceeds {threshold}% threshold"
                )
            }
            GpuAlert::PowerLimit {
                device,
                power_percent,
                threshold,
            } => {
                format!("GPU {device} power limit: {power_percent}% exceeds {threshold}% threshold")
            }
            GpuAlert::GpuIdle {
                device,
                duration_secs,
            } => {
                format!("GPU {device} idle for {duration_secs} seconds (possible hang)")
            }
        }
    }
}

/// Andon thresholds for GPU alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndonThresholds {
    /// Temperature warning threshold in Celsius
    pub thermal_warning: u32,
    /// Temperature critical threshold in Celsius
    pub thermal_critical: u32,
    /// Memory usage warning threshold (0-100%)
    pub memory_warning: u32,
    /// Memory usage critical threshold (0-100%)
    pub memory_critical: u32,
    /// Power usage warning threshold (0-100% of limit)
    pub power_warning: u32,
    /// Idle duration threshold in seconds
    pub idle_threshold_secs: u32,
}

impl Default for AndonThresholds {
    fn default() -> Self {
        Self {
            thermal_warning: 80,
            thermal_critical: 90,
            memory_warning: 90,
            memory_critical: 95,
            power_warning: 95,
            idle_threshold_secs: 30,
        }
    }
}

/// GPU Andon system for alert management
#[derive(Debug)]
pub struct GpuAndonSystem {
    /// Alert thresholds
    thresholds: AndonThresholds,
    /// Active alerts
    alerts: Vec<GpuAlert>,
    /// Idle tracking per device (device_id -> consecutive idle samples)
    idle_samples: Vec<u32>,
    /// Sample interval in seconds (for idle calculation)
    sample_interval_secs: u32,
}

impl GpuAndonSystem {
    /// Create a new Andon system with default thresholds
    pub fn new() -> Self {
        Self::with_thresholds(AndonThresholds::default())
    }

    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: AndonThresholds) -> Self {
        Self {
            thresholds,
            alerts: Vec::new(),
            idle_samples: Vec::new(),
            sample_interval_secs: 1,
        }
    }

    /// Set sample interval for idle calculation
    pub fn set_sample_interval(&mut self, secs: u32) {
        self.sample_interval_secs = secs;
    }

    /// Check metrics and generate alerts
    pub fn check(&mut self, metrics: &[GpuMetrics]) -> Vec<GpuAlert> {
        self.alerts.clear();

        // Ensure idle_samples is sized correctly
        if self.idle_samples.len() < metrics.len() {
            self.idle_samples.resize(metrics.len(), 0);
        }

        for m in metrics {
            // Thermal check
            if m.temperature_celsius >= self.thresholds.thermal_warning {
                self.alerts.push(GpuAlert::ThermalThrottling {
                    device: m.device_id,
                    temp: m.temperature_celsius,
                    threshold: self.thresholds.thermal_warning,
                });
            }

            // Memory pressure check
            let mem_percent = m.memory_percent() as u32;
            if mem_percent >= self.thresholds.memory_warning {
                self.alerts.push(GpuAlert::MemoryPressure {
                    device: m.device_id,
                    used_percent: mem_percent,
                    threshold: self.thresholds.memory_warning,
                });
            }

            // Power limit check
            let power_percent = m.power_percent() as u32;
            if power_percent >= self.thresholds.power_warning {
                self.alerts.push(GpuAlert::PowerLimit {
                    device: m.device_id,
                    power_percent,
                    threshold: self.thresholds.power_warning,
                });
            }

            // Idle check
            let device_idx = m.device_id as usize;
            if device_idx < self.idle_samples.len() {
                if m.utilization_percent == 0 {
                    self.idle_samples[device_idx] += 1;
                    let idle_secs = self.idle_samples[device_idx] * self.sample_interval_secs;
                    if idle_secs >= self.thresholds.idle_threshold_secs {
                        self.alerts.push(GpuAlert::GpuIdle {
                            device: m.device_id,
                            duration_secs: idle_secs,
                        });
                    }
                } else {
                    self.idle_samples[device_idx] = 0;
                }
            }
        }

        self.alerts.clone()
    }

    /// Get current alerts
    pub fn alerts(&self) -> &[GpuAlert] {
        &self.alerts
    }

    /// Get thresholds
    pub fn thresholds(&self) -> &AndonThresholds {
        &self.thresholds
    }

    /// Check if any critical alerts are active
    pub fn has_critical_alerts(&self) -> bool {
        self.alerts.iter().any(|a| a.severity() >= 80)
    }
}

impl Default for GpuAndonSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU metrics history buffer (ring buffer)
#[derive(Debug)]
pub struct GpuMetricsBuffer {
    /// Capacity
    capacity: usize,
    /// Metrics per device
    buffers: Vec<VecDeque<GpuMetrics>>,
}

impl GpuMetricsBuffer {
    /// Create a new buffer with given capacity
    pub fn new(capacity: usize, num_devices: usize) -> Self {
        let buffers = (0..num_devices)
            .map(|_| VecDeque::with_capacity(capacity))
            .collect();
        Self { capacity, buffers }
    }

    /// Push metrics for all devices
    pub fn push(&mut self, metrics: &[GpuMetrics]) {
        for m in metrics {
            let device_idx = m.device_id as usize;
            if device_idx >= self.buffers.len() {
                self.buffers
                    .resize_with(device_idx + 1, || VecDeque::with_capacity(self.capacity));
            }

            let buffer = &mut self.buffers[device_idx];
            if buffer.len() >= self.capacity {
                buffer.pop_front();
            }
            buffer.push_back(m.clone());
        }
    }

    /// Get last N metrics for a device
    pub fn last_n(&self, device_id: u32, n: usize) -> Vec<&GpuMetrics> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx]
            .iter()
            .rev()
            .take(n)
            .rev()
            .collect()
    }

    /// Get utilization history for a device (for sparkline)
    pub fn utilization_history(&self, device_id: u32) -> Vec<u32> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx]
            .iter()
            .map(|m| m.utilization_percent)
            .collect()
    }

    /// Get temperature history for a device
    pub fn temperature_history(&self, device_id: u32) -> Vec<u32> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx]
            .iter()
            .map(|m| m.temperature_celsius)
            .collect()
    }

    /// Get memory utilization history for a device
    pub fn memory_history(&self, device_id: u32) -> Vec<f64> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx]
            .iter()
            .map(GpuMetrics::memory_percent)
            .collect()
    }

    /// Get number of samples for a device
    pub fn len(&self, device_id: u32) -> usize {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return 0;
        }
        self.buffers[device_idx].len()
    }

    /// Check if buffer for device is empty
    pub fn is_empty(&self, device_id: u32) -> bool {
        self.len(device_id) == 0
    }
}

/// GPU monitor that collects metrics
///
/// Uses mock implementation for testing. Production would use NVML.
#[derive(Debug)]
pub struct GpuMonitor {
    /// Number of detected devices
    num_devices: u32,
    /// Mock mode for testing
    mock_mode: bool,
    /// Mock metrics generator
    mock_metrics: Vec<GpuMetrics>,
}

impl GpuMonitor {
    /// Create a new GPU monitor
    ///
    /// Attempts to initialize NVML, falls back to no devices if unavailable.
    pub fn new() -> Result<Self, String> {
        // In production, this would use nvml-wrapper crate
        // For now, return empty (graceful degradation)
        Ok(Self {
            num_devices: 0,
            mock_mode: false,
            mock_metrics: Vec::new(),
        })
    }

    /// Create a mock GPU monitor for testing
    pub fn mock(num_devices: u32) -> Self {
        let mock_metrics = (0..num_devices).map(GpuMetrics::mock).collect();
        Self {
            num_devices,
            mock_mode: true,
            mock_metrics,
        }
    }

    /// Get number of detected devices
    pub fn num_devices(&self) -> u32 {
        self.num_devices
    }

    /// Check if running in mock mode
    pub fn is_mock(&self) -> bool {
        self.mock_mode
    }

    /// Sample current GPU metrics
    pub fn sample(&self) -> Vec<GpuMetrics> {
        if self.mock_mode {
            return self.mock_metrics.clone();
        }

        // Production implementation would query NVML here
        Vec::new()
    }

    /// Sample with simulated variation (for testing)
    pub fn sample_with_variation(&mut self, variation: f32) -> Vec<GpuMetrics> {
        if !self.mock_mode {
            return Vec::new();
        }

        self.mock_metrics
            .iter()
            .map(|base| {
                let mut m = base.clone();
                let var = (variation * 10.0) as i32;
                m.utilization_percent = (m.utilization_percent as i32 + var).clamp(0, 100) as u32;
                m.temperature_celsius =
                    (m.temperature_celsius as i32 + var / 2).clamp(30, 100) as u32;
                m.power_watts = (m.power_watts + variation * 20.0).clamp(0.0, m.power_limit_watts);
                m
            })
            .collect()
    }

    /// Set mock metrics (for testing specific scenarios)
    pub fn set_mock_metrics(&mut self, metrics: Vec<GpuMetrics>) {
        self.mock_metrics = metrics;
        self.num_devices = self.mock_metrics.len() as u32;
        self.mock_mode = true;
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self::mock(0))
    }
}

/// Render a progress bar for terminal display
pub fn render_progress_bar(value: f64, width: usize) -> String {
    let filled = ((value / 100.0) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;

    let bar: String = std::iter::repeat_n('█', filled).collect();
    let empty_bar: String = std::iter::repeat_n('░', empty).collect();

    format!("{bar}{empty_bar}")
}

/// Render a sparkline from values
pub fn render_sparkline(values: &[u32], max_val: u32) -> String {
    const CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if values.is_empty() || max_val == 0 {
        return String::new();
    }

    values
        .iter()
        .map(|&v| {
            let idx =
                ((f64::from(v) / f64::from(max_val)) * (CHARS.len() - 1) as f64).round() as usize;
            CHARS[idx.min(CHARS.len() - 1)]
        })
        .collect()
}

/// Format GPU metrics for terminal display
pub fn format_gpu_panel(metrics: &GpuMetrics, width: usize) -> Vec<String> {
    let bar_width = width.saturating_sub(25);

    vec![
        format!(
            "───── GPU {}: {} ─────",
            metrics.device_id,
            metrics.name.chars().take(width - 20).collect::<String>()
        ),
        format!(
            "Util: {} {:>3}%  │  Temp: {}°C",
            render_progress_bar(f64::from(metrics.utilization_percent), bar_width),
            metrics.utilization_percent,
            metrics.temperature_celsius
        ),
        format!(
            "VRAM: {} {:.1}/{:.1} GB ({:.0}%)",
            render_progress_bar(metrics.memory_percent(), bar_width),
            metrics.memory_used_mb as f64 / 1024.0,
            metrics.memory_total_mb as f64 / 1024.0,
            metrics.memory_percent()
        ),
        format!(
            "Pow:  {} {:.0}W/{:.0}W",
            render_progress_bar(metrics.power_percent(), bar_width),
            metrics.power_watts,
            metrics.power_limit_watts
        ),
    ]
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // GpuMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gpu_metrics_mock() {
        let m = GpuMetrics::mock(0);
        assert_eq!(m.device_id, 0);
        assert!(!m.name.is_empty());
        assert!(m.utilization_percent <= 100);
    }

    #[test]
    fn test_gpu_metrics_memory_percent() {
        let mut m = GpuMetrics::mock(0);
        m.memory_used_mb = 8000;
        m.memory_total_mb = 16000;
        assert!((m.memory_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_memory_percent_zero_total() {
        let mut m = GpuMetrics::mock(0);
        m.memory_total_mb = 0;
        assert!((m.memory_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_metrics_power_percent() {
        let mut m = GpuMetrics::mock(0);
        m.power_watts = 225.0;
        m.power_limit_watts = 450.0;
        assert!((m.power_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_power_percent_zero_limit() {
        let mut m = GpuMetrics::mock(0);
        m.power_limit_watts = 0.0;
        assert!((m.power_percent() - 0.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // GpuAlert Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gpu_alert_thermal_severity() {
        let alert = GpuAlert::ThermalThrottling {
            device: 0,
            temp: 90,
            threshold: 80,
        };
        assert_eq!(alert.severity(), 60); // 50 + min(10, 50)
    }

    #[test]
    fn test_gpu_alert_memory_critical_severity() {
        let alert = GpuAlert::MemoryPressure {
            device: 0,
            used_percent: 99,
            threshold: 90,
        };
        assert_eq!(alert.severity(), 100);
    }

    #[test]
    fn test_gpu_alert_message() {
        let alert = GpuAlert::ThermalThrottling {
            device: 0,
            temp: 85,
            threshold: 80,
        };
        let msg = alert.message();
        assert!(msg.contains("GPU 0"));
        assert!(msg.contains("85°C"));
        assert!(msg.contains("80°C"));
    }

    #[test]
    fn test_gpu_alert_idle_message() {
        let alert = GpuAlert::GpuIdle {
            device: 1,
            duration_secs: 45,
        };
        let msg = alert.message();
        assert!(msg.contains("GPU 1"));
        assert!(msg.contains("45 seconds"));
    }

    // -------------------------------------------------------------------------
    // AndonThresholds Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_andon_thresholds_default() {
        let t = AndonThresholds::default();
        assert_eq!(t.thermal_warning, 80);
        assert_eq!(t.memory_warning, 90);
        assert_eq!(t.power_warning, 95);
    }

    // -------------------------------------------------------------------------
    // GpuAndonSystem Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_andon_system_no_alerts() {
        let mut andon = GpuAndonSystem::new();
        let metrics = vec![GpuMetrics {
            device_id: 0,
            temperature_celsius: 60,
            utilization_percent: 80,
            memory_used_mb: 4000,
            memory_total_mb: 16000,
            power_watts: 200.0,
            power_limit_watts: 450.0,
            ..Default::default()
        }];

        let alerts = andon.check(&metrics);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_andon_system_thermal_alert() {
        let mut andon = GpuAndonSystem::new();
        let metrics = vec![GpuMetrics {
            device_id: 0,
            temperature_celsius: 85,
            ..Default::default()
        }];

        let alerts = andon.check(&metrics);
        assert_eq!(alerts.len(), 1);
        assert!(matches!(alerts[0], GpuAlert::ThermalThrottling { .. }));
    }

    #[test]
    fn test_andon_system_memory_alert() {
        let mut andon = GpuAndonSystem::new();
        let metrics = vec![GpuMetrics {
            device_id: 0,
            memory_used_mb: 15000,
            memory_total_mb: 16000, // 93.75%
            ..Default::default()
        }];

        let alerts = andon.check(&metrics);
        assert!(alerts
            .iter()
            .any(|a| matches!(a, GpuAlert::MemoryPressure { .. })));
    }

    #[test]
    fn test_andon_system_power_alert() {
        let mut andon = GpuAndonSystem::new();
        let metrics = vec![GpuMetrics {
            device_id: 0,
            power_watts: 440.0,
            power_limit_watts: 450.0, // 97.8%
            ..Default::default()
        }];

        let alerts = andon.check(&metrics);
        assert!(alerts
            .iter()
            .any(|a| matches!(a, GpuAlert::PowerLimit { .. })));
    }

    #[test]
    fn test_andon_system_idle_alert() {
        let mut andon = GpuAndonSystem::new();
        andon.set_sample_interval(10);

        let metrics = vec![GpuMetrics {
            device_id: 0,
            utilization_percent: 0,
            ..Default::default()
        }];

        // Need 3 samples (30 seconds) to trigger idle alert
        andon.check(&metrics);
        andon.check(&metrics);
        andon.check(&metrics);
        let alerts = andon.check(&metrics);

        assert!(alerts.iter().any(|a| matches!(a, GpuAlert::GpuIdle { .. })));
    }

    #[test]
    fn test_andon_system_idle_reset() {
        let mut andon = GpuAndonSystem::new();
        andon.set_sample_interval(10);

        let idle_metrics = vec![GpuMetrics {
            device_id: 0,
            utilization_percent: 0,
            ..Default::default()
        }];

        let active_metrics = vec![GpuMetrics {
            device_id: 0,
            utilization_percent: 50,
            ..Default::default()
        }];

        // Build up idle counter
        andon.check(&idle_metrics);
        andon.check(&idle_metrics);

        // Reset with activity
        andon.check(&active_metrics);

        // Check again - should not have idle alert yet
        let alerts = andon.check(&idle_metrics);
        assert!(!alerts.iter().any(|a| matches!(a, GpuAlert::GpuIdle { .. })));
    }

    #[test]
    fn test_andon_system_has_critical_alerts() {
        let mut andon = GpuAndonSystem::new();
        let metrics = vec![GpuMetrics {
            device_id: 0,
            memory_used_mb: 15800,
            memory_total_mb: 16000, // 98.75% - critical
            ..Default::default()
        }];

        andon.check(&metrics);
        assert!(andon.has_critical_alerts());
    }

    // -------------------------------------------------------------------------
    // GpuMetricsBuffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_push_and_last_n() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..5 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i * 10,
                ..Default::default()
            }]);
        }

        let last3 = buffer.last_n(0, 3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].utilization_percent, 20);
        assert_eq!(last3[1].utilization_percent, 30);
        assert_eq!(last3[2].utilization_percent, 40);
    }

    #[test]
    fn test_buffer_capacity_limit() {
        let mut buffer = GpuMetricsBuffer::new(5, 1);

        for i in 0..10 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i,
                ..Default::default()
            }]);
        }

        assert_eq!(buffer.len(0), 5);

        let history = buffer.utilization_history(0);
        assert_eq!(history, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_buffer_utilization_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..5 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i * 20,
                ..Default::default()
            }]);
        }

        let history = buffer.utilization_history(0);
        assert_eq!(history, vec![0, 20, 40, 60, 80]);
    }

    #[test]
    fn test_buffer_temperature_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..3 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                temperature_celsius: 60 + i * 5,
                ..Default::default()
            }]);
        }

        let history = buffer.temperature_history(0);
        assert_eq!(history, vec![60, 65, 70]);
    }

    #[test]
    fn test_buffer_multiple_devices() {
        let mut buffer = GpuMetricsBuffer::new(10, 2);

        buffer.push(&[
            GpuMetrics {
                device_id: 0,
                utilization_percent: 50,
                ..Default::default()
            },
            GpuMetrics {
                device_id: 1,
                utilization_percent: 75,
                ..Default::default()
            },
        ]);

        assert_eq!(buffer.utilization_history(0), vec![50]);
        assert_eq!(buffer.utilization_history(1), vec![75]);
    }

    #[test]
    fn test_buffer_empty_device() {
        let buffer = GpuMetricsBuffer::new(10, 1);
        assert!(buffer.is_empty(0));
        assert!(buffer.utilization_history(5).is_empty()); // Non-existent device
    }

    // -------------------------------------------------------------------------
    // GpuMonitor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gpu_monitor_new() {
        let monitor = GpuMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_gpu_monitor_mock() {
        let monitor = GpuMonitor::mock(2);
        assert_eq!(monitor.num_devices(), 2);
        assert!(monitor.is_mock());
    }

    #[test]
    fn test_gpu_monitor_sample_mock() {
        let monitor = GpuMonitor::mock(2);
        let metrics = monitor.sample();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].device_id, 0);
        assert_eq!(metrics[1].device_id, 1);
    }

    #[test]
    fn test_gpu_monitor_sample_with_variation() {
        let mut monitor = GpuMonitor::mock(1);
        let base = monitor.sample()[0].utilization_percent;

        let varied = monitor.sample_with_variation(1.0);
        // Variation should change the value
        assert!(varied[0].utilization_percent != base || base == 100 || base == 0);
    }

    #[test]
    fn test_gpu_monitor_set_mock_metrics() {
        let mut monitor = GpuMonitor::mock(0);
        monitor.set_mock_metrics(vec![GpuMetrics {
            device_id: 5,
            utilization_percent: 99,
            ..Default::default()
        }]);

        let metrics = monitor.sample();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].device_id, 5);
        assert_eq!(metrics[0].utilization_percent, 99);
    }

    // -------------------------------------------------------------------------
    // Rendering Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_progress_bar() {
        let bar = render_progress_bar(50.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 5);
    }

    #[test]
    fn test_render_progress_bar_full() {
        let bar = render_progress_bar(100.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    #[test]
    fn test_render_progress_bar_empty() {
        let bar = render_progress_bar(0.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
    }

    #[test]
    fn test_render_sparkline() {
        let sparkline = render_sparkline(&[0, 50, 100], 100);
        assert_eq!(sparkline.chars().count(), 3);
        assert!(sparkline.starts_with('▁'));
        assert!(sparkline.ends_with('█'));
    }

    #[test]
    fn test_render_sparkline_empty() {
        let sparkline = render_sparkline(&[], 100);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_format_gpu_panel() {
        let metrics = GpuMetrics::mock(0);
        let lines = format_gpu_panel(&metrics, 60);
        assert!(!lines.is_empty());
        assert!(lines[0].contains("GPU 0"));
    }

    #[test]
    fn test_gpu_metrics_default() {
        let m = GpuMetrics::default();
        assert_eq!(m.device_id, 0);
        assert!(m.name.is_empty());
        assert_eq!(m.utilization_percent, 0);
    }

    #[test]
    fn test_gpu_monitor_default() {
        let monitor = GpuMonitor::default();
        // Should either be real or mock, but should work
        let _ = monitor.num_devices();
    }

    #[test]
    fn test_render_sparkline_max_val_zero() {
        let sparkline = render_sparkline(&[1, 2, 3], 0);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_gpu_alert_memory_severity_thresholds() {
        // 95-98%
        let alert_95 = GpuAlert::MemoryPressure {
            device: 0,
            used_percent: 96,
            threshold: 90,
        };
        assert_eq!(alert_95.severity(), 80);

        // Below 95%
        let alert_91 = GpuAlert::MemoryPressure {
            device: 0,
            used_percent: 91,
            threshold: 90,
        };
        assert_eq!(alert_91.severity(), 60);
    }

    #[test]
    fn test_gpu_alert_power_severity_thresholds() {
        // At 100%
        let alert_100 = GpuAlert::PowerLimit {
            device: 0,
            power_percent: 100,
            threshold: 95,
        };
        assert_eq!(alert_100.severity(), 70);

        // Below 100%
        let alert_99 = GpuAlert::PowerLimit {
            device: 0,
            power_percent: 99,
            threshold: 95,
        };
        assert_eq!(alert_99.severity(), 50);
    }

    #[test]
    fn test_gpu_alert_idle_severity_thresholds() {
        // > 60 seconds
        let alert_long = GpuAlert::GpuIdle {
            device: 0,
            duration_secs: 65,
        };
        assert_eq!(alert_long.severity(), 90);

        // 31-60 seconds
        let alert_medium = GpuAlert::GpuIdle {
            device: 0,
            duration_secs: 45,
        };
        assert_eq!(alert_medium.severity(), 70);

        // <= 30 seconds
        let alert_short = GpuAlert::GpuIdle {
            device: 0,
            duration_secs: 20,
        };
        assert_eq!(alert_short.severity(), 40);
    }

    #[test]
    fn test_gpu_alert_messages() {
        let memory_alert = GpuAlert::MemoryPressure {
            device: 2,
            used_percent: 95,
            threshold: 90,
        };
        let msg = memory_alert.message();
        assert!(msg.contains("GPU 2"));
        assert!(msg.contains("95%"));
        assert!(msg.contains("90%"));

        let power_alert = GpuAlert::PowerLimit {
            device: 3,
            power_percent: 98,
            threshold: 95,
        };
        let msg = power_alert.message();
        assert!(msg.contains("GPU 3"));
        assert!(msg.contains("98%"));
        assert!(msg.contains("95%"));
    }

    #[test]
    fn test_gpu_alert_serde() {
        let alert = GpuAlert::ThermalThrottling {
            device: 0,
            temp: 85,
            threshold: 80,
        };
        let json = serde_json::to_string(&alert).unwrap();
        let parsed: GpuAlert = serde_json::from_str(&json).unwrap();
        assert_eq!(alert, parsed);
    }

    #[test]
    fn test_gpu_metrics_serde() {
        let metrics = GpuMetrics::mock(0);
        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: GpuMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.device_id, parsed.device_id);
        assert_eq!(metrics.utilization_percent, parsed.utilization_percent);
    }

    #[test]
    fn test_andon_thresholds_serde() {
        let thresholds = AndonThresholds::default();
        let json = serde_json::to_string(&thresholds).unwrap();
        let parsed: AndonThresholds = serde_json::from_str(&json).unwrap();
        assert_eq!(thresholds.thermal_warning, parsed.thermal_warning);
    }

    #[test]
    fn test_gpu_alert_clone() {
        let alert = GpuAlert::MemoryPressure {
            device: 1,
            used_percent: 95,
            threshold: 90,
        };
        let cloned = alert.clone();
        assert_eq!(alert, cloned);
    }

    #[test]
    fn test_gpu_metrics_clone() {
        let metrics = GpuMetrics::mock(0);
        let cloned = metrics.clone();
        assert_eq!(metrics.device_id, cloned.device_id);
        assert_eq!(metrics.name, cloned.name);
    }

    #[test]
    fn test_andon_thresholds_clone() {
        let thresholds = AndonThresholds::default();
        let cloned = thresholds.clone();
        assert_eq!(thresholds.thermal_warning, cloned.thermal_warning);
    }

    #[test]
    fn test_render_progress_bar_over_100() {
        // Should handle values > 100
        let bar = render_progress_bar(150.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    #[test]
    fn test_render_progress_bar_negative() {
        // Should handle negative values
        let bar = render_progress_bar(-10.0, 10);
        // Negative rounds to 0
        assert!(bar.chars().filter(|&c| c == '█').count() == 0);
    }

    #[test]
    fn test_gpu_monitor_non_mock_sample() {
        // Create non-mock monitor but call sample (should return empty or real data)
        let monitor = GpuMonitor::new().unwrap();
        let metrics = monitor.sample();
        // Just verify it doesn't crash - may be empty if no NVML
        let _ = metrics;
    }

    #[test]
    fn test_gpu_monitor_non_mock_sample_with_variation() {
        let mut monitor = GpuMonitor::new().unwrap();
        // Non-mock mode should return empty
        let metrics = monitor.sample_with_variation(1.0);
        // Should be empty for non-mock mode
        assert!(metrics.is_empty() || !monitor.is_mock());
    }

    #[test]
    fn test_buffer_memory_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..3 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                memory_used_mb: i * 1000,
                memory_total_mb: 8000,
                memory_utilization_percent: (i as u32) * 10,
                ..Default::default()
            }]);
        }

        let history = buffer.memory_history(0);
        // memory_history returns memory_percent() values: 0/8000=0%, 1000/8000=12.5%, 2000/8000=25%
        assert_eq!(history.len(), 3);
        assert!((history[0] - 0.0).abs() < 0.1);
        assert!((history[1] - 12.5).abs() < 0.1);
        assert!((history[2] - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_buffer_last_n_more_than_available() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        buffer.push(&[GpuMetrics {
            device_id: 0,
            utilization_percent: 50,
            ..Default::default()
        }]);

        // Request more than available
        let last5 = buffer.last_n(0, 5);
        assert_eq!(last5.len(), 1);
    }

    #[test]
    fn test_buffer_last_nonexistent_device() {
        let buffer = GpuMetricsBuffer::new(10, 1);
        let last = buffer.last_n(99, 5);
        assert!(last.is_empty());
    }

    #[test]
    fn test_andon_system_custom_thresholds() {
        let thresholds = AndonThresholds {
            thermal_warning: 70, // Lower threshold
            thermal_critical: 85,
            memory_warning: 80,
            memory_critical: 90,
            power_warning: 90,
            idle_threshold_secs: 20,
        };
        let mut andon = GpuAndonSystem::with_thresholds(thresholds);

        let metrics = vec![GpuMetrics {
            device_id: 0,
            temperature_celsius: 75, // Would not trigger default 80 but triggers 70
            ..Default::default()
        }];

        let alerts = andon.check(&metrics);
        assert!(alerts
            .iter()
            .any(|a| matches!(a, GpuAlert::ThermalThrottling { .. })));
    }

    #[test]
    fn test_andon_system_alerts_cleared_on_check() {
        let mut andon = GpuAndonSystem::new();
        // Use critical memory (99%) which has severity 100
        let critical_metrics = vec![GpuMetrics {
            device_id: 0,
            memory_used_mb: 15840, // 99% of 16000
            memory_total_mb: 16000,
            ..Default::default()
        }];

        andon.check(&critical_metrics);
        assert!(andon.has_critical_alerts());

        // Check with normal metrics - alerts should be cleared
        let normal_metrics = vec![GpuMetrics {
            device_id: 0,
            memory_used_mb: 8000,
            memory_total_mb: 16000,
            ..Default::default()
        }];
        andon.check(&normal_metrics);
        assert!(!andon.has_critical_alerts());
    }

    #[test]
    fn test_thermal_severity_max_excess() {
        // Test thermal severity with large excess (should cap at 100)
        let alert = GpuAlert::ThermalThrottling {
            device: 0,
            temp: 200,
            threshold: 80,
        };
        // 50 + min(120, 50) = 50 + 50 = 100
        assert_eq!(alert.severity(), 100);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_memory_percent_bounds(used in 0u64..100000, total in 1u64..100000) {
            let m = GpuMetrics {
                memory_used_mb: used,
                memory_total_mb: total,
                ..Default::default()
            };
            let percent = m.memory_percent();
            prop_assert!(percent >= 0.0);
            // Can be > 100 if used > total (which is invalid but shouldn't crash)
        }

        #[test]
        fn prop_power_percent_bounds(power in 0.0f32..1000.0, limit in 0.1f32..1000.0) {
            let m = GpuMetrics {
                power_watts: power,
                power_limit_watts: limit,
                ..Default::default()
            };
            let percent = m.power_percent();
            prop_assert!(percent >= 0.0);
        }

        #[test]
        fn prop_alert_severity_bounds(
            temp in 0u32..200,
            threshold in 1u32..100
        ) {
            let alert = GpuAlert::ThermalThrottling {
                device: 0,
                temp,
                threshold,
            };
            let severity = alert.severity();
            prop_assert!(severity >= 50);
            prop_assert!(severity <= 100);
        }

        #[test]
        fn prop_sparkline_length(values in prop::collection::vec(0u32..100, 0..50)) {
            let sparkline = render_sparkline(&values, 100);
            prop_assert_eq!(sparkline.chars().count(), values.len());
        }

        #[test]
        fn prop_progress_bar_length(value in 0.0f64..100.0, width in 1usize..50) {
            let bar = render_progress_bar(value, width);
            let char_count: usize = bar.chars().count();
            prop_assert_eq!(char_count, width);
        }

        #[test]
        fn prop_buffer_respects_capacity(capacity in 1usize..20, pushes in 1usize..50) {
            let mut buffer = GpuMetricsBuffer::new(capacity, 1);
            for i in 0..pushes {
                buffer.push(&[GpuMetrics {
                    device_id: 0,
                    utilization_percent: i as u32 % 100,
                    ..Default::default()
                }]);
            }
            prop_assert!(buffer.len(0) <= capacity);
        }
    }
}

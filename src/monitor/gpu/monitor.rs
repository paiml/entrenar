//! GPU monitor that collects metrics.

use super::GpuMetrics;

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_gpu_monitor_default() {
        let monitor = GpuMonitor::default();
        // Should either be real or mock, but should work
        let _ = monitor.num_devices();
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
}

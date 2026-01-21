//! GPU alert types and Andon system for monitoring.

use serde::{Deserialize, Serialize};

use super::GpuMetrics;

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_andon_thresholds_default() {
        let t = AndonThresholds::default();
        assert_eq!(t.thermal_warning, 80);
        assert_eq!(t.memory_warning, 90);
        assert_eq!(t.power_warning, 95);
    }

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
    fn test_andon_thresholds_clone() {
        let thresholds = AndonThresholds::default();
        let cloned = thresholds.clone();
        assert_eq!(thresholds.thermal_warning, cloned.thermal_warning);
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

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

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
    }
}

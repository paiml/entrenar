//! YAML Configuration for Monitor (ENT-062)
//!
//! Declarative configuration for terminal monitoring.

use super::callback::TerminalMonitorCallback;
use super::capability::{DashboardLayout, TerminalCapabilities, TerminalMode};

/// Monitor configuration for YAML.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MonitorConfig {
    /// Enable terminal monitoring
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Dashboard layout
    #[serde(default)]
    pub layout: String,
    /// Terminal mode (ascii, unicode, ansi)
    #[serde(default)]
    pub terminal_mode: String,
    /// Refresh interval in milliseconds
    #[serde(default = "default_refresh")]
    pub refresh_ms: u64,
    /// Sparkline width
    #[serde(default = "default_sparkline_width")]
    pub sparkline_width: usize,
    /// Show ETA
    #[serde(default = "default_true")]
    pub show_eta: bool,
    /// Reference curve path (optional)
    #[serde(default)]
    pub reference_curve: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_refresh() -> u64 {
    100
}
fn default_sparkline_width() -> usize {
    20
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            layout: "compact".to_string(),
            terminal_mode: "auto".to_string(),
            refresh_ms: 100,
            sparkline_width: 20,
            show_eta: true,
            reference_curve: None,
        }
    }
}

impl MonitorConfig {
    /// Create TerminalMonitorCallback from config.
    pub fn to_callback(&self) -> TerminalMonitorCallback {
        let layout = match self.layout.as_str() {
            "minimal" => DashboardLayout::Minimal,
            "full" => DashboardLayout::Full,
            _ => DashboardLayout::Compact,
        };

        let mode = match self.terminal_mode.as_str() {
            "ascii" => TerminalMode::Ascii,
            "ansi" => TerminalMode::Ansi,
            "unicode" => TerminalMode::Unicode,
            _ => TerminalCapabilities::detect().recommended_mode(),
        };

        TerminalMonitorCallback::new()
            .layout(layout)
            .mode(mode)
            .sparkline_width(self.sparkline_width)
            .refresh_interval_ms(self.refresh_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.layout, "compact");
        assert_eq!(config.terminal_mode, "auto");
        assert_eq!(config.refresh_ms, 100);
        assert_eq!(config.sparkline_width, 20);
        assert!(config.show_eta);
        assert!(config.reference_curve.is_none());
    }

    #[test]
    fn test_monitor_config_to_callback() {
        let config = MonitorConfig::default();
        let _callback = config.to_callback();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_monitor_config_to_callback_minimal() {
        let config = MonitorConfig {
            layout: "minimal".to_string(),
            ..Default::default()
        };
        let _callback = config.to_callback();
    }

    #[test]
    fn test_monitor_config_to_callback_full() {
        let config = MonitorConfig {
            layout: "full".to_string(),
            ..Default::default()
        };
        let _callback = config.to_callback();
    }

    #[test]
    fn test_monitor_config_to_callback_ascii() {
        let config = MonitorConfig {
            terminal_mode: "ascii".to_string(),
            ..Default::default()
        };
        let _callback = config.to_callback();
    }

    #[test]
    fn test_monitor_config_to_callback_ansi() {
        let config = MonitorConfig {
            terminal_mode: "ansi".to_string(),
            ..Default::default()
        };
        let _callback = config.to_callback();
    }

    #[test]
    fn test_monitor_config_serde_roundtrip() {
        let config = MonitorConfig {
            enabled: true,
            layout: "full".to_string(),
            terminal_mode: "unicode".to_string(),
            refresh_ms: 200,
            sparkline_width: 30,
            show_eta: false,
            reference_curve: Some("golden.json".to_string()),
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: MonitorConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.layout, config.layout);
        assert_eq!(parsed.terminal_mode, config.terminal_mode);
        assert_eq!(parsed.refresh_ms, config.refresh_ms);
        assert_eq!(parsed.sparkline_width, config.sparkline_width);
        assert_eq!(parsed.show_eta, config.show_eta);
        assert_eq!(parsed.reference_curve, config.reference_curve);
    }
}

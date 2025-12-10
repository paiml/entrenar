//! Safety Andon for Inference (ENT-110)
//!
//! Toyota Way 自働化 (Jidoka): Automation with human touch.
//! Inference-specific Andon rules: low confidence, high latency, drift.

use super::path::DecisionPath;
use super::trace::DecisionTrace;
use crate::monitor::andon::{Alert, AlertLevel, AndonSystem};
use serde::{Deserialize, Serialize};

/// Safety Integrity Level (IEC 61508)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyIntegrityLevel {
    /// QM: No safety requirements (games, entertainment)
    /// - Ring buffer traces
    /// - Best-effort logging
    QM,

    /// SIL 1: Low safety requirements
    /// - Persistent traces
    /// - Hash verification
    SIL1,

    /// SIL 2: Medium safety requirements
    /// - Hash chain
    /// - Redundant storage
    SIL2,

    /// SIL 3: High safety requirements (automotive ASIL C)
    /// - Hash chain with signatures
    /// - Triple redundant storage
    /// - Hardware security module
    SIL3,

    /// SIL 4: Highest safety requirements (automotive ASIL D)
    /// - All SIL 3 requirements
    /// - Formal verification of trace system
    /// - Independent safety monitor
    SIL4,
}

impl SafetyIntegrityLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SafetyIntegrityLevel::QM => "QM",
            SafetyIntegrityLevel::SIL1 => "SIL1",
            SafetyIntegrityLevel::SIL2 => "SIL2",
            SafetyIntegrityLevel::SIL3 => "SIL3",
            SafetyIntegrityLevel::SIL4 => "SIL4",
        }
    }

    /// Get minimum confidence threshold for this level
    pub fn min_confidence(&self) -> f32 {
        match self {
            SafetyIntegrityLevel::QM => 0.0, // No requirement
            SafetyIntegrityLevel::SIL1 => 0.5,
            SafetyIntegrityLevel::SIL2 => 0.7,
            SafetyIntegrityLevel::SIL3 => 0.8,
            SafetyIntegrityLevel::SIL4 => 0.9,
        }
    }

    /// Get maximum allowed latency in nanoseconds
    pub fn max_latency_ns(&self) -> u64 {
        match self {
            SafetyIntegrityLevel::QM => u64::MAX,      // No requirement
            SafetyIntegrityLevel::SIL1 => 100_000_000, // 100ms
            SafetyIntegrityLevel::SIL2 => 50_000_000,  // 50ms
            SafetyIntegrityLevel::SIL3 => 10_000_000,  // 10ms
            SafetyIntegrityLevel::SIL4 => 1_000_000,   // 1ms
        }
    }
}

/// Emergency condition that triggers immediate action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyCondition {
    /// Collision imminent (AV)
    CollisionImminent {
        /// Time to collision in milliseconds
        time_to_collision_ms: f32,
    },

    /// Sensor degraded
    SensorDegraded {
        /// Sensor identifier
        sensor: String,
        /// Quality score (0.0 - 1.0)
        quality: f32,
    },

    /// Audit chain integrity failure
    AuditChainBroken,

    /// Decision system timeout
    DecisionTimeout {
        /// Maximum allowed latency in milliseconds
        max_ms: f32,
    },

    /// Repeated low-confidence decisions
    ConsecutiveLowConfidence {
        /// Number of consecutive low-confidence decisions
        count: usize,
        /// Threshold for low confidence
        threshold: f32,
    },

    /// NaN or Inf detected in output
    InvalidOutput,
}

impl EmergencyCondition {
    /// Get alert level for this condition
    pub fn alert_level(&self) -> AlertLevel {
        match self {
            EmergencyCondition::CollisionImminent { .. } => AlertLevel::Critical,
            EmergencyCondition::SensorDegraded { quality, .. } if *quality < 0.3 => {
                AlertLevel::Critical
            }
            EmergencyCondition::SensorDegraded { .. } => AlertLevel::Error,
            EmergencyCondition::AuditChainBroken => AlertLevel::Critical,
            EmergencyCondition::DecisionTimeout { .. } => AlertLevel::Error,
            EmergencyCondition::ConsecutiveLowConfidence { .. } => AlertLevel::Warning,
            EmergencyCondition::InvalidOutput => AlertLevel::Critical,
        }
    }

    /// Generate alert message
    pub fn message(&self) -> String {
        match self {
            EmergencyCondition::CollisionImminent {
                time_to_collision_ms,
            } => {
                format!("Collision imminent in {time_to_collision_ms:.1}ms")
            }
            EmergencyCondition::SensorDegraded { sensor, quality } => {
                let quality_pct = quality * 100.0;
                format!("Sensor {sensor} degraded: quality={quality_pct:.1}%")
            }
            EmergencyCondition::AuditChainBroken => "Audit chain integrity failure".to_string(),
            EmergencyCondition::DecisionTimeout { max_ms } => {
                format!("Decision timeout: exceeded {max_ms:.1}ms limit")
            }
            EmergencyCondition::ConsecutiveLowConfidence { count, threshold } => {
                let threshold_pct = threshold * 100.0;
                format!("{count} consecutive decisions below {threshold_pct:.1}% confidence")
            }
            EmergencyCondition::InvalidOutput => "NaN or Inf detected in model output".to_string(),
        }
    }
}

/// Safety Andon for inference monitoring
///
/// # Features
/// - Confidence monitoring
/// - Latency monitoring
/// - Emergency condition detection
/// - Integration with base Andon system
pub struct SafetyAndon {
    /// Base Andon system
    andon: AndonSystem,
    /// Safety integrity level
    sil: SafetyIntegrityLevel,
    /// Minimum acceptable confidence
    min_confidence: f32,
    /// Maximum acceptable latency in nanoseconds
    max_latency_ns: u64,
    /// Consecutive low-confidence counter
    low_confidence_count: usize,
    /// Threshold for low confidence alert
    low_confidence_threshold: usize,
    /// Alert on unknown classification
    alert_on_unknown: bool,
}

impl SafetyAndon {
    /// Create a new safety Andon system
    pub fn new(sil: SafetyIntegrityLevel) -> Self {
        Self {
            andon: AndonSystem::new(),
            min_confidence: sil.min_confidence(),
            max_latency_ns: sil.max_latency_ns(),
            sil,
            low_confidence_count: 0,
            low_confidence_threshold: 5,
            alert_on_unknown: true,
        }
    }

    /// Set custom confidence threshold
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold;
        self
    }

    /// Set custom latency threshold
    pub fn with_max_latency_ns(mut self, max_ns: u64) -> Self {
        self.max_latency_ns = max_ns;
        self
    }

    /// Set consecutive low-confidence threshold
    pub fn with_low_confidence_threshold(mut self, threshold: usize) -> Self {
        self.low_confidence_threshold = threshold;
        self
    }

    /// Disable unknown classification alerts
    pub fn without_unknown_alerts(mut self) -> Self {
        self.alert_on_unknown = false;
        self
    }

    /// Check a trace against safety rules
    pub fn check_trace<P: DecisionPath>(
        &mut self,
        trace: &DecisionTrace<P>,
        latency_budget_ns: u64,
    ) {
        let confidence = trace.confidence();
        let latency_ns = trace.latency_ns;

        // Check for invalid output
        if trace.output.is_nan() || trace.output.is_infinite() {
            self.trigger_emergency(EmergencyCondition::InvalidOutput);
            return;
        }

        // Check confidence
        if confidence < self.min_confidence {
            self.low_confidence_count += 1;

            if self.low_confidence_count >= self.low_confidence_threshold {
                self.trigger_emergency(EmergencyCondition::ConsecutiveLowConfidence {
                    count: self.low_confidence_count,
                    threshold: self.min_confidence,
                });
            } else {
                self.andon.trigger(
                    Alert::warning(format!(
                        "Low confidence: {:.1}% (threshold: {:.1}%)",
                        confidence * 100.0,
                        self.min_confidence * 100.0
                    ))
                    .with_source("SafetyAndon")
                    .with_value(f64::from(confidence)),
                );
            }
        } else {
            self.low_confidence_count = 0;
        }

        // Check latency
        let effective_budget = latency_budget_ns.min(self.max_latency_ns);
        if latency_ns > effective_budget {
            let latency_ms = latency_ns as f64 / 1_000_000.0;
            let budget_ms = effective_budget as f64 / 1_000_000.0;

            if latency_ns > self.max_latency_ns * 2 {
                // Critical: more than 2x over budget
                self.trigger_emergency(EmergencyCondition::DecisionTimeout {
                    max_ms: budget_ms as f32,
                });
            } else {
                self.andon.trigger(
                    Alert::warning(format!(
                        "Latency exceeded: {latency_ms:.2}ms > {budget_ms:.2}ms budget"
                    ))
                    .with_source("SafetyAndon")
                    .with_value(latency_ms),
                );
            }
        }
    }

    /// Trigger an emergency condition
    pub fn trigger_emergency(&mut self, condition: EmergencyCondition) {
        let alert =
            Alert::new(condition.alert_level(), condition.message()).with_source("SafetyAndon");
        self.andon.trigger(alert);
    }

    /// Check if stop has been requested
    pub fn should_stop(&self) -> bool {
        self.andon.should_stop()
    }

    /// Reset the Andon system
    pub fn reset(&mut self) {
        self.andon.reset();
        self.low_confidence_count = 0;
    }

    /// Get alert history
    pub fn history(&self) -> &[Alert] {
        self.andon.history()
    }

    /// Get the safety integrity level
    pub fn sil(&self) -> SafetyIntegrityLevel {
        self.sil
    }

    /// Get reference to the underlying Andon system
    pub fn andon(&self) -> &AndonSystem {
        &self.andon
    }
}

impl Default for SafetyAndon {
    fn default() -> Self {
        Self::new(SafetyIntegrityLevel::QM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::inference::path::LinearPath;
    use crate::monitor::inference::trace::DecisionTrace;

    fn make_trace(confidence: f32, latency_ns: u64, output: f32) -> DecisionTrace<LinearPath> {
        let path = LinearPath::new(vec![0.5], 0.0, 0.0, output).with_probability(confidence);
        DecisionTrace::new(0, 0, 0, path, output, latency_ns)
    }

    #[test]
    fn test_sil_levels() {
        assert_eq!(SafetyIntegrityLevel::QM.min_confidence(), 0.0);
        assert_eq!(SafetyIntegrityLevel::SIL4.min_confidence(), 0.9);

        assert_eq!(SafetyIntegrityLevel::QM.max_latency_ns(), u64::MAX);
        assert_eq!(SafetyIntegrityLevel::SIL4.max_latency_ns(), 1_000_000);
    }

    #[test]
    fn test_sil_as_str() {
        assert_eq!(SafetyIntegrityLevel::QM.as_str(), "QM");
        assert_eq!(SafetyIntegrityLevel::SIL3.as_str(), "SIL3");
    }

    #[test]
    fn test_safety_andon_new() {
        let andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);
        assert_eq!(andon.sil(), SafetyIntegrityLevel::SIL2);
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_safety_andon_confidence_check() {
        let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);

        // High confidence - no alert
        let trace = make_trace(0.9, 1_000, 0.9);
        andon.check_trace(&trace, 10_000_000);
        assert!(!andon.should_stop());
        assert!(andon.history().is_empty());

        // Low confidence - warning
        let trace = make_trace(0.5, 1_000, 0.5);
        andon.check_trace(&trace, 10_000_000);
        assert!(!andon.should_stop());
        assert_eq!(andon.history().len(), 1);
        assert_eq!(andon.history()[0].level, AlertLevel::Warning);
    }

    #[test]
    fn test_safety_andon_consecutive_low_confidence() {
        let mut andon =
            SafetyAndon::new(SafetyIntegrityLevel::SIL2).with_low_confidence_threshold(3);

        // 3 consecutive low confidence decisions
        for _ in 0..3 {
            let trace = make_trace(0.5, 1_000, 0.5);
            andon.check_trace(&trace, 10_000_000);
        }

        // Should have triggered emergency
        assert!(andon.history().len() >= 3);
    }

    #[test]
    fn test_safety_andon_latency_check() {
        let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL3);

        // Fast response - no alert
        let trace = make_trace(0.9, 1_000_000, 0.9);
        andon.check_trace(&trace, 10_000_000);
        assert!(andon.history().is_empty());

        // Slow response - warning
        let trace = make_trace(0.9, 15_000_000, 0.9);
        andon.check_trace(&trace, 10_000_000);
        assert_eq!(andon.history().len(), 1);
    }

    #[test]
    fn test_safety_andon_invalid_output() {
        let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

        // NaN output - critical
        let trace = make_trace(0.9, 1_000, f32::NAN);
        andon.check_trace(&trace, 10_000_000);
        assert!(andon.should_stop());
    }

    #[test]
    fn test_safety_andon_reset() {
        let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

        let trace = make_trace(0.9, 1_000, f32::NAN);
        andon.check_trace(&trace, 10_000_000);
        assert!(andon.should_stop());

        andon.reset();
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_emergency_condition_message() {
        let cond = EmergencyCondition::CollisionImminent {
            time_to_collision_ms: 50.0,
        };
        assert!(cond.message().contains("50.0ms"));

        let cond = EmergencyCondition::SensorDegraded {
            sensor: "lidar".to_string(),
            quality: 0.5,
        };
        assert!(cond.message().contains("lidar"));
        assert!(cond.message().contains("50.0%"));
    }

    #[test]
    fn test_emergency_condition_alert_level() {
        assert_eq!(
            EmergencyCondition::CollisionImminent {
                time_to_collision_ms: 10.0
            }
            .alert_level(),
            AlertLevel::Critical
        );
        assert_eq!(
            EmergencyCondition::AuditChainBroken.alert_level(),
            AlertLevel::Critical
        );
        assert_eq!(
            EmergencyCondition::DecisionTimeout { max_ms: 10.0 }.alert_level(),
            AlertLevel::Error
        );
    }

    #[test]
    fn test_safety_andon_custom_thresholds() {
        let andon = SafetyAndon::new(SafetyIntegrityLevel::QM)
            .with_min_confidence(0.95)
            .with_max_latency_ns(500_000)
            .with_low_confidence_threshold(10);

        assert_eq!(andon.min_confidence, 0.95);
        assert_eq!(andon.max_latency_ns, 500_000);
        assert_eq!(andon.low_confidence_threshold, 10);
    }
}

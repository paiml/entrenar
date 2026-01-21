//! Auto-Retraining Module (APR-073-5)
//!
//! Implements the Andon Cord pattern for automated retraining when drift is detected.
//! Bridges drift detection to the training loop following Toyota Way principles.

use super::drift::{DriftDetector, DriftResult, DriftSummary, Severity};
use crate::error::Result;

/// Retraining trigger policy
#[derive(Clone, Debug, Default)]
pub enum RetrainPolicy {
    /// Retrain if >= N features show drift
    FeatureCount { count: usize },
    /// Retrain if any feature with these names drifts
    CriticalFeature { names: Vec<String> },
    /// Retrain if drift percentage exceeds threshold
    DriftPercentage { threshold: f64 },
    /// Retrain on any critical severity drift
    #[default]
    AnyCritical,
}

/// Action taken by the AutoRetrainer
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// No action needed
    None,
    /// Warning logged but no retrain triggered
    WarningLogged,
    /// Retraining was triggered with given job ID
    RetrainTriggered(String),
}

/// Configuration for auto-retraining
#[derive(Clone, Debug)]
pub struct RetrainConfig {
    /// Policy for when to trigger retraining
    pub policy: RetrainPolicy,
    /// Cooldown period between retrains (in batches processed)
    pub cooldown_batches: usize,
    /// Maximum retrains per session (0 = unlimited)
    pub max_retrains: usize,
    /// Whether to log warnings for non-critical drift
    pub log_warnings: bool,
}

impl Default for RetrainConfig {
    fn default() -> Self {
        Self {
            policy: RetrainPolicy::default(),
            cooldown_batches: 100,
            max_retrains: 0,
            log_warnings: true,
        }
    }
}

/// Callback type for retrain triggers
pub type RetrainCallback = Box<dyn Fn(&[DriftResult]) -> Result<String> + Send + Sync>;

/// Auto-retrainer that monitors drift and triggers retraining
pub struct AutoRetrainer {
    detector: DriftDetector,
    config: RetrainConfig,
    retrain_callback: Option<RetrainCallback>,
    batches_since_retrain: usize,
    total_retrains: usize,
}

impl AutoRetrainer {
    /// Create a new auto-retrainer with given detector and config
    pub fn new(detector: DriftDetector, config: RetrainConfig) -> Self {
        Self {
            detector,
            config,
            retrain_callback: None,
            batches_since_retrain: 0,
            total_retrains: 0,
        }
    }

    /// Set the callback to invoke when retraining is triggered
    ///
    /// The callback receives the drift results and should return a job ID
    /// or an error if retraining failed to start.
    pub fn on_retrain<F>(&mut self, callback: F)
    where
        F: Fn(&[DriftResult]) -> Result<String> + Send + Sync + 'static,
    {
        self.retrain_callback = Some(Box::new(callback));
    }

    /// Process a batch of data and check for drift
    ///
    /// Returns the action taken based on drift detection and policy.
    pub fn process_batch(&mut self, batch: &[Vec<f64>]) -> Result<Action> {
        self.batches_since_retrain += 1;

        // Check for drift
        let results = self.detector.check(batch);

        if results.is_empty() {
            return Ok(Action::None);
        }

        let summary = DriftDetector::summary(&results);

        // Check if we're in cooldown
        if self.batches_since_retrain < self.config.cooldown_batches {
            if summary.has_drift() && self.config.log_warnings {
                return Ok(Action::WarningLogged);
            }
            return Ok(Action::None);
        }

        // Check max retrains limit
        if self.config.max_retrains > 0 && self.total_retrains >= self.config.max_retrains {
            if summary.has_drift() && self.config.log_warnings {
                return Ok(Action::WarningLogged);
            }
            return Ok(Action::None);
        }

        // Evaluate policy
        let should_retrain = self.evaluate_policy(&results, &summary);

        if should_retrain {
            if let Some(ref callback) = self.retrain_callback {
                let job_id = callback(&results)?;
                self.batches_since_retrain = 0;
                self.total_retrains += 1;
                return Ok(Action::RetrainTriggered(job_id));
            }
            // No callback set but policy says retrain
            return Ok(Action::WarningLogged);
        }

        if summary.warnings > 0 && self.config.log_warnings {
            return Ok(Action::WarningLogged);
        }

        Ok(Action::None)
    }

    /// Evaluate whether retraining should be triggered based on policy
    fn evaluate_policy(&self, results: &[DriftResult], summary: &DriftSummary) -> bool {
        match &self.config.policy {
            RetrainPolicy::FeatureCount { count } => summary.drifted_features >= *count,

            RetrainPolicy::CriticalFeature { names } => results
                .iter()
                .any(|r| r.drifted && names.contains(&r.feature)),

            RetrainPolicy::DriftPercentage { threshold } => {
                summary.drift_percentage() >= *threshold
            }

            RetrainPolicy::AnyCritical => results.iter().any(|r| r.severity == Severity::Critical),
        }
    }

    /// Get the underlying drift detector
    pub fn detector(&self) -> &DriftDetector {
        &self.detector
    }

    /// Get mutable reference to drift detector (for setting baseline)
    pub fn detector_mut(&mut self) -> &mut DriftDetector {
        &mut self.detector
    }

    /// Get statistics about retraining
    pub fn stats(&self) -> RetrainerStats {
        RetrainerStats {
            total_retrains: self.total_retrains,
            batches_since_retrain: self.batches_since_retrain,
        }
    }

    /// Reset the cooldown counter
    pub fn reset_cooldown(&mut self) {
        self.batches_since_retrain = self.config.cooldown_batches;
    }
}

/// Statistics about the auto-retrainer
#[derive(Clone, Debug)]
pub struct RetrainerStats {
    /// Total number of retrains triggered
    pub total_retrains: usize,
    /// Batches processed since last retrain
    pub batches_since_retrain: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::drift::DriftTest;

    fn create_detector() -> DriftDetector {
        DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }])
    }

    #[test]
    fn test_retrain_policy_default() {
        let policy = RetrainPolicy::default();
        assert!(matches!(policy, RetrainPolicy::AnyCritical));
    }

    #[test]
    fn test_retrain_config_default() {
        let config = RetrainConfig::default();
        assert_eq!(config.cooldown_batches, 100);
        assert_eq!(config.max_retrains, 0);
        assert!(config.log_warnings);
    }

    #[test]
    fn test_auto_retrainer_no_baseline() {
        let detector = create_detector();
        let config = RetrainConfig::default();
        let mut retrainer = AutoRetrainer::new(detector, config);

        let batch: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let action = retrainer.process_batch(&batch).unwrap();

        assert_eq!(action, Action::None);
    }

    #[test]
    fn test_auto_retrainer_no_drift() {
        let mut detector = create_detector();
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            cooldown_batches: 0,
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        // Same distribution should not trigger
        let action = retrainer.process_batch(&baseline).unwrap();
        assert_eq!(action, Action::None);
    }

    #[test]
    fn test_auto_retrainer_with_drift() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut detector = create_detector();
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            cooldown_batches: 0,
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        let retrain_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&retrain_count);

        retrainer.on_retrain(move |_results| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            Ok("job-123".to_string())
        });

        // Shifted distribution should trigger
        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![i as f64]).collect();
        let action = retrainer.process_batch(&shifted).unwrap();

        assert!(matches!(action, Action::RetrainTriggered(_)));
        assert_eq!(retrain_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cooldown_prevents_retrain() {
        let mut detector = create_detector();
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            cooldown_batches: 10, // Require 10 batches between retrains
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        retrainer.on_retrain(|_| Ok("job".to_string()));

        // Reset cooldown so first batch can trigger
        retrainer.reset_cooldown();

        // First batch with drift should trigger
        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![i as f64]).collect();
        let action1 = retrainer.process_batch(&shifted).unwrap();
        assert!(matches!(action1, Action::RetrainTriggered(_)));

        // Immediate second batch should be blocked by cooldown
        let action2 = retrainer.process_batch(&shifted).unwrap();
        assert_eq!(action2, Action::WarningLogged);
    }

    #[test]
    fn test_max_retrains_limit() {
        let mut detector = create_detector();
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            cooldown_batches: 0,
            max_retrains: 2,
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        retrainer.on_retrain(|_| Ok("job".to_string()));

        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![i as f64]).collect();

        // First two should trigger
        assert!(matches!(
            retrainer.process_batch(&shifted).unwrap(),
            Action::RetrainTriggered(_)
        ));
        assert!(matches!(
            retrainer.process_batch(&shifted).unwrap(),
            Action::RetrainTriggered(_)
        ));

        // Third should be blocked
        assert_eq!(
            retrainer.process_batch(&shifted).unwrap(),
            Action::WarningLogged
        );
    }

    #[test]
    fn test_feature_count_policy() {
        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            policy: RetrainPolicy::FeatureCount { count: 2 },
            cooldown_batches: 0,
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        retrainer.on_retrain(|_| Ok("job".to_string()));

        // Both features shifted - should trigger
        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let action = retrainer.process_batch(&shifted).unwrap();
        assert!(matches!(action, Action::RetrainTriggered(_)));
    }

    #[test]
    fn test_stats() {
        let detector = create_detector();
        let config = RetrainConfig::default();
        let retrainer = AutoRetrainer::new(detector, config);

        let stats = retrainer.stats();
        assert_eq!(stats.total_retrains, 0);
        assert_eq!(stats.batches_since_retrain, 0);
    }

    #[test]
    fn test_action_eq() {
        assert_eq!(Action::None, Action::None);
        assert_eq!(Action::WarningLogged, Action::WarningLogged);
        assert_ne!(Action::None, Action::WarningLogged);
        assert_eq!(
            Action::RetrainTriggered("a".to_string()),
            Action::RetrainTriggered("a".to_string())
        );
    }

    /// APR-073 Section 10.4: Callback must trigger within 10ms
    #[test]
    fn test_callback_latency() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use std::time::Instant;

        let mut detector = create_detector();
        let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        detector.set_baseline(&baseline);

        let config = RetrainConfig {
            cooldown_batches: 0,
            ..Default::default()
        };
        let mut retrainer = AutoRetrainer::new(detector, config);

        // Store callback latency in nanoseconds
        let latency_ns = Arc::new(AtomicU64::new(0));
        let latency_clone = Arc::clone(&latency_ns);

        let callback_start = Arc::new(std::sync::Mutex::new(None::<Instant>));
        let start_clone = Arc::clone(&callback_start);

        retrainer.on_retrain(move |_results| {
            if let Ok(mut guard) = start_clone.lock() {
                if let Some(start) = *guard {
                    let elapsed = start.elapsed().as_nanos() as u64;
                    latency_clone.store(elapsed, Ordering::SeqCst);
                }
            }
            Ok("job-latency-test".to_string())
        });

        // Record start time and process drifted batch
        *callback_start.lock().unwrap() = Some(Instant::now());
        let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![i as f64]).collect();
        let action = retrainer.process_batch(&shifted).unwrap();

        assert!(matches!(action, Action::RetrainTriggered(_)));

        // Verify callback executed within 10ms (10,000,000 ns)
        let latency = latency_ns.load(Ordering::SeqCst);
        assert!(
            latency < 10_000_000,
            "Callback latency {}ns exceeds 10ms requirement",
            latency
        );
    }
}

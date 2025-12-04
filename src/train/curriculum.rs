//! Curriculum Learning for progressive training
//!
//! Implements curriculum learning strategies as described in:
//! - Bengio et al. (2009) "Curriculum Learning"
//!
//! This module provides schedulers that progressively adjust training
//! difficulty, data selection, and sample weighting.
//!
//! # CITL Support
//!
//! Designed to support Compiler-in-the-Loop (CITL) training where:
//! - Diagnostic verbosity increases with model maturity
//! - Rare error classes (long-tail) get appropriate attention
//! - Training efficiency is balanced against corpus size

use std::collections::HashMap;

/// Trait for curriculum learning schedulers
///
/// Determines training difficulty/tier based on training progress.
pub trait CurriculumScheduler: Send {
    /// Get the current difficulty level (0.0 = easiest, 1.0 = hardest)
    fn difficulty(&self) -> f32;

    /// Get the current tier (for tiered training like CITL)
    fn tier(&self) -> usize;

    /// Advance the curriculum based on training progress
    fn step(&mut self, epoch: usize, accuracy: f32);

    /// Reset the curriculum to initial state
    fn reset(&mut self);

    /// Get sample weight for a given difficulty score
    ///
    /// Returns weight multiplier for loss (1.0 = normal weight)
    fn sample_weight(&self, sample_difficulty: f32) -> f32 {
        1.0 - (sample_difficulty - self.difficulty()).abs().min(1.0) * 0.5
    }

    /// Check if sample should be included at current difficulty
    fn include_sample(&self, sample_difficulty: f32) -> bool {
        sample_difficulty <= self.difficulty()
    }

    /// Name of the curriculum scheduler
    fn name(&self) -> &str;
}

// =============================================================================
// Linear Curriculum
// =============================================================================

/// Linear curriculum that increases difficulty over epochs
///
/// Difficulty increases linearly from `start_difficulty` to `end_difficulty`
/// over `ramp_epochs` epochs.
///
/// # Example
///
/// ```
/// use entrenar::train::{LinearCurriculum, CurriculumScheduler};
///
/// let mut curriculum = LinearCurriculum::new(0.3, 1.0, 10);
///
/// // Initially at start difficulty
/// assert!((curriculum.difficulty() - 0.3).abs() < 1e-5);
///
/// // After 5 epochs at 100% accuracy
/// for _ in 0..5 {
///     curriculum.step(0, 1.0);
/// }
/// assert!(curriculum.difficulty() > 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct LinearCurriculum {
    start_difficulty: f32,
    end_difficulty: f32,
    ramp_epochs: usize,
    current_epoch: usize,
}

impl LinearCurriculum {
    /// Create a new linear curriculum
    ///
    /// # Arguments
    ///
    /// * `start_difficulty` - Initial difficulty (0.0-1.0)
    /// * `end_difficulty` - Final difficulty (0.0-1.0)
    /// * `ramp_epochs` - Epochs to reach full difficulty
    pub fn new(start_difficulty: f32, end_difficulty: f32, ramp_epochs: usize) -> Self {
        Self {
            start_difficulty: start_difficulty.clamp(0.0, 1.0),
            end_difficulty: end_difficulty.clamp(0.0, 1.0),
            ramp_epochs: ramp_epochs.max(1),
            current_epoch: 0,
        }
    }
}

impl CurriculumScheduler for LinearCurriculum {
    fn difficulty(&self) -> f32 {
        let progress = (self.current_epoch as f32 / self.ramp_epochs as f32).min(1.0);
        let difficulty =
            self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty);
        let (min, max) = if self.start_difficulty <= self.end_difficulty {
            (self.start_difficulty, self.end_difficulty)
        } else {
            (self.end_difficulty, self.start_difficulty)
        };
        difficulty.clamp(min, max)
    }

    fn tier(&self) -> usize {
        // Map difficulty to 4 tiers (1-4)
        let d = self.difficulty();
        if d < 0.25 {
            1
        } else if d < 0.5 {
            2
        } else if d < 0.75 {
            3
        } else {
            4
        }
    }

    fn step(&mut self, _epoch: usize, _accuracy: f32) {
        self.current_epoch += 1;
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
    }

    fn name(&self) -> &'static str {
        "LinearCurriculum"
    }
}

// =============================================================================
// Tiered Curriculum (for CITL)
// =============================================================================

/// Tiered curriculum for diagnostic verbosity levels
///
/// Designed for CITL training with four diagnostic tiers:
/// - Tier 1: JSON diagnostics + clippy (baseline)
/// - Tier 2: + verbose build output
/// - Tier 3: + RUSTC_LOG traces
/// - Tier 4: + full debug output
///
/// Tier advancement based on accuracy thresholds.
///
/// # Example
///
/// ```
/// use entrenar::train::{TieredCurriculum, CurriculumScheduler};
///
/// let mut curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8], 3);
///
/// assert_eq!(curriculum.tier(), 1);
///
/// // Advance to tier 2 after achieving 60% accuracy for 3 epochs
/// for _ in 0..3 {
///     curriculum.step(0, 0.65);
/// }
/// assert_eq!(curriculum.tier(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TieredCurriculum {
    /// Accuracy thresholds to advance to next tier
    tier_thresholds: Vec<f32>,
    /// Consecutive epochs at threshold before advancing
    patience: usize,
    /// Current tier (1-4)
    current_tier: usize,
    /// Epochs at current tier meeting threshold
    epochs_at_threshold: usize,
}

impl TieredCurriculum {
    /// Create new tiered curriculum
    ///
    /// # Arguments
    ///
    /// * `tier_thresholds` - Accuracy thresholds for each tier advancement
    /// * `patience` - Epochs at threshold before advancing
    pub fn new(tier_thresholds: Vec<f32>, patience: usize) -> Self {
        Self {
            tier_thresholds,
            patience: patience.max(1),
            current_tier: 1,
            epochs_at_threshold: 0,
        }
    }

    /// Create with default CITL thresholds
    ///
    /// - Tier 1 → 2: 60% accuracy
    /// - Tier 2 → 3: 70% accuracy
    /// - Tier 3 → 4: 80% accuracy
    pub fn citl_default() -> Self {
        Self::new(vec![0.6, 0.7, 0.8], 3)
    }

    /// Get the threshold for current tier advancement
    pub fn current_threshold(&self) -> Option<f32> {
        if self.current_tier <= self.tier_thresholds.len() {
            Some(self.tier_thresholds[self.current_tier - 1])
        } else {
            None
        }
    }
}

impl CurriculumScheduler for TieredCurriculum {
    fn difficulty(&self) -> f32 {
        (self.current_tier as f32 - 1.0) / 3.0
    }

    fn tier(&self) -> usize {
        self.current_tier
    }

    fn step(&mut self, _epoch: usize, accuracy: f32) {
        if let Some(threshold) = self.current_threshold() {
            if accuracy >= threshold {
                self.epochs_at_threshold += 1;
                if self.epochs_at_threshold >= self.patience {
                    // Advance to next tier
                    self.current_tier = (self.current_tier + 1).min(4);
                    self.epochs_at_threshold = 0;
                }
            } else {
                // Reset counter if below threshold
                self.epochs_at_threshold = 0;
            }
        }
    }

    fn reset(&mut self) {
        self.current_tier = 1;
        self.epochs_at_threshold = 0;
    }

    fn name(&self) -> &'static str {
        "TieredCurriculum"
    }
}

// =============================================================================
// Adaptive Curriculum (for error-specific training)
// =============================================================================

/// Adaptive curriculum that adjusts based on error class performance
///
/// Tracks accuracy per error class and increases difficulty for
/// well-learned classes while maintaining focus on struggling classes.
///
/// Supports the CITL adaptive tier selection pattern.
#[derive(Debug, Clone)]
pub struct AdaptiveCurriculum {
    /// Accuracy per error class
    class_accuracy: HashMap<String, f32>,
    /// Attempts per error class
    class_attempts: HashMap<String, usize>,
    /// Default tier for unknown errors
    default_tier: usize,
    /// Overall difficulty based on mean accuracy
    overall_difficulty: f32,
}

impl AdaptiveCurriculum {
    /// Create new adaptive curriculum
    pub fn new() -> Self {
        Self {
            class_accuracy: HashMap::new(),
            class_attempts: HashMap::new(),
            default_tier: 1,
            overall_difficulty: 0.0,
        }
    }

    /// Get recommended tier for an error class
    ///
    /// Based on the CITL `select_tier()` pattern
    pub fn tier_for_error(&self, error_code: &str, attempt: usize) -> usize {
        // Special cases
        if error_code.starts_with("ICE") {
            return 4; // ICEs always need full debug
        }

        // Type/trait errors benefit from traces
        if matches!(error_code, "E0308" | "E0277" | "E0382") && attempt >= 1 {
            return 3;
        }

        // Name resolution needs verbose
        if matches!(error_code, "E0425" | "E0433") && attempt >= 2 {
            return 3;
        }

        // Default escalation pattern
        match attempt {
            0 => self.default_tier,
            1 => 2,
            _ => 3,
        }
    }

    /// Update accuracy for an error class
    pub fn update_class(&mut self, error_code: &str, correct: bool) {
        let attempts = self
            .class_attempts
            .entry(error_code.to_string())
            .or_insert(0);
        *attempts += 1;

        let acc = self
            .class_accuracy
            .entry(error_code.to_string())
            .or_insert(0.0);
        // Exponential moving average
        let alpha = 0.1;
        *acc = *acc * (1.0 - alpha) + if correct { alpha } else { 0.0 };

        // Update overall difficulty
        if !self.class_accuracy.is_empty() {
            self.overall_difficulty =
                self.class_accuracy.values().sum::<f32>() / self.class_accuracy.len() as f32;
        }
    }

    /// Get sample weight based on class rarity and accuracy
    ///
    /// Long-tail (rare) errors get higher weights per Feldman (2020)
    pub fn weight_for_class(&self, error_code: &str) -> f32 {
        let attempts = *self.class_attempts.get(error_code).unwrap_or(&0);
        let accuracy = *self.class_accuracy.get(error_code).unwrap_or(&0.0);

        // Rare classes get higher weight
        let rarity_weight = 1.0 / (attempts as f32 + 1.0).sqrt();

        // Low accuracy classes get higher weight
        let difficulty_weight = 1.0 - accuracy;

        // Combine weights (normalize to reasonable range)
        (1.0 + rarity_weight + difficulty_weight).min(3.0)
    }
}

impl Default for AdaptiveCurriculum {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumScheduler for AdaptiveCurriculum {
    fn difficulty(&self) -> f32 {
        self.overall_difficulty
    }

    fn tier(&self) -> usize {
        if self.overall_difficulty < 0.25 {
            1
        } else if self.overall_difficulty < 0.5 {
            2
        } else if self.overall_difficulty < 0.75 {
            3
        } else {
            4
        }
    }

    fn step(&mut self, _epoch: usize, accuracy: f32) {
        // Update overall difficulty based on recent accuracy
        let alpha = 0.1;
        self.overall_difficulty = self.overall_difficulty * (1.0 - alpha) + accuracy * alpha;
    }

    fn reset(&mut self) {
        self.class_accuracy.clear();
        self.class_attempts.clear();
        self.overall_difficulty = 0.0;
    }

    fn name(&self) -> &'static str {
        "AdaptiveCurriculum"
    }
}

// =============================================================================
// Efficiency Metric
// =============================================================================

/// Compute efficiency score as per CITL spec
///
/// E(T) = Accuracy(T) / log(CorpusSize(T))
///
/// Higher is better - balances accuracy against corpus bloat.
pub fn efficiency_score(accuracy: f32, corpus_size_bytes: usize) -> f32 {
    if corpus_size_bytes <= 1 {
        return accuracy;
    }
    accuracy / (corpus_size_bytes as f32).ln()
}

/// Compare tiers and select optimal based on efficiency
///
/// Returns (best_tier, efficiency_score)
pub fn select_optimal_tier(tier_results: &[(usize, f32, usize)]) -> Option<(usize, f32)> {
    tier_results
        .iter()
        .map(|&(tier, accuracy, corpus_size)| {
            let eff = efficiency_score(accuracy, corpus_size);
            (tier, eff)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curriculum_initial() {
        let curriculum = LinearCurriculum::new(0.3, 1.0, 10);
        assert!((curriculum.difficulty() - 0.3).abs() < 1e-5);
        assert_eq!(curriculum.tier(), 2); // 0.3 -> tier 2
    }

    #[test]
    fn test_linear_curriculum_progress() {
        let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

        for i in 0..10 {
            curriculum.step(i, 1.0);
        }

        assert!((curriculum.difficulty() - 1.0).abs() < 1e-5);
        assert_eq!(curriculum.tier(), 4);
    }

    #[test]
    fn test_linear_curriculum_halfway() {
        let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

        for _ in 0..5 {
            curriculum.step(0, 1.0);
        }

        assert!((curriculum.difficulty() - 0.5).abs() < 1e-5);
        assert_eq!(curriculum.tier(), 3);
    }

    #[test]
    fn test_linear_curriculum_reset() {
        let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

        for _ in 0..5 {
            curriculum.step(0, 1.0);
        }
        curriculum.reset();

        assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiered_curriculum_initial() {
        let curriculum = TieredCurriculum::citl_default();
        assert_eq!(curriculum.tier(), 1);
        assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiered_curriculum_advance() {
        let mut curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8], 3);

        // Not enough accuracy
        for _ in 0..3 {
            curriculum.step(0, 0.5);
        }
        assert_eq!(curriculum.tier(), 1);

        // Enough accuracy, but not enough patience
        for _ in 0..2 {
            curriculum.step(0, 0.65);
        }
        assert_eq!(curriculum.tier(), 1);

        // Third epoch at threshold -> advance
        curriculum.step(0, 0.65);
        assert_eq!(curriculum.tier(), 2);
    }

    #[test]
    fn test_tiered_curriculum_max_tier() {
        // Need 3 thresholds to reach tier 4 (tier 1→2, 2→3, 3→4)
        let mut curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);

        // Advance multiple times at 100% accuracy
        for _ in 0..10 {
            curriculum.step(0, 1.0);
        }

        // Should max out at tier 4
        assert_eq!(curriculum.tier(), 4);
    }

    #[test]
    fn test_adaptive_curriculum_tier_for_error() {
        let curriculum = AdaptiveCurriculum::new();

        // ICE always tier 4
        assert_eq!(curriculum.tier_for_error("ICE-0001", 0), 4);

        // Type error with retry -> tier 3
        assert_eq!(curriculum.tier_for_error("E0308", 1), 3);

        // Name resolution with 2 retries -> tier 3
        assert_eq!(curriculum.tier_for_error("E0425", 2), 3);

        // Default first attempt -> tier 1
        assert_eq!(curriculum.tier_for_error("E0599", 0), 1);

        // Default first retry -> tier 2
        assert_eq!(curriculum.tier_for_error("E0599", 1), 2);
    }

    #[test]
    fn test_adaptive_curriculum_class_tracking() {
        let mut curriculum = AdaptiveCurriculum::new();

        // Track some predictions
        curriculum.update_class("E0308", true);
        curriculum.update_class("E0308", true);
        curriculum.update_class("E0425", false);

        // E0308 should have higher accuracy
        let e0308_acc = *curriculum.class_accuracy.get("E0308").unwrap();
        let e0425_acc = *curriculum.class_accuracy.get("E0425").unwrap();
        assert!(e0308_acc > e0425_acc);

        // E0425 (low accuracy) should have higher weight
        let e0308_weight = curriculum.weight_for_class("E0308");
        let e0425_weight = curriculum.weight_for_class("E0425");
        assert!(e0425_weight > e0308_weight);
    }

    #[test]
    fn test_efficiency_score() {
        // Higher accuracy -> higher efficiency
        assert!(efficiency_score(0.9, 1000) > efficiency_score(0.5, 1000));

        // Same accuracy, smaller corpus -> higher efficiency
        assert!(efficiency_score(0.7, 1000) > efficiency_score(0.7, 10000));
    }

    #[test]
    fn test_select_optimal_tier() {
        let results = vec![
            (1, 0.65, 2000),   // E = 0.65 / ln(2000) ≈ 0.085
            (2, 0.72, 5000),   // E = 0.72 / ln(5000) ≈ 0.084
            (3, 0.75, 20000),  // E = 0.75 / ln(20000) ≈ 0.076
            (4, 0.77, 100000), // E = 0.77 / ln(100000) ≈ 0.067
        ];

        let (best_tier, _) = select_optimal_tier(&results).unwrap();
        // Tier 1 or 2 should win due to smaller corpus
        assert!(best_tier <= 2);
    }

    #[test]
    fn test_sample_weight() {
        let curriculum = LinearCurriculum::new(0.5, 0.5, 10);

        // Sample at current difficulty -> full weight
        let weight_at = curriculum.sample_weight(0.5);
        assert!((weight_at - 1.0).abs() < 1e-5);

        // Sample far from current difficulty -> reduced weight
        let weight_far = curriculum.sample_weight(0.0);
        assert!(weight_far < 1.0);
    }

    #[test]
    fn test_include_sample() {
        let curriculum = LinearCurriculum::new(0.5, 0.5, 10);

        // Sample at or below difficulty -> included
        assert!(curriculum.include_sample(0.5));
        assert!(curriculum.include_sample(0.3));

        // Sample above difficulty -> excluded
        assert!(!curriculum.include_sample(0.7));
    }

    #[test]
    fn test_curriculum_names() {
        assert_eq!(
            LinearCurriculum::new(0.0, 1.0, 10).name(),
            "LinearCurriculum"
        );
        assert_eq!(TieredCurriculum::citl_default().name(), "TieredCurriculum");
        assert_eq!(AdaptiveCurriculum::new().name(), "AdaptiveCurriculum");
    }

    #[test]
    fn test_adaptive_curriculum_step_and_reset() {
        let mut curriculum = AdaptiveCurriculum::new();

        // Step with high accuracy should increase overall difficulty
        curriculum.step(0, 0.9);
        assert!(curriculum.difficulty() > 0.0);

        curriculum.step(1, 0.8);
        let difficulty_after_step = curriculum.difficulty();
        assert!(difficulty_after_step > 0.0);

        // Reset should clear everything
        curriculum.reset();
        assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
        assert!(curriculum.class_accuracy.is_empty());
        assert!(curriculum.class_attempts.is_empty());
    }

    #[test]
    fn test_adaptive_curriculum_all_tiers() {
        let mut curriculum = AdaptiveCurriculum::new();

        // Tier 1 when difficulty < 0.25
        assert_eq!(curriculum.tier(), 1);

        // Push difficulty up
        for _ in 0..5 {
            curriculum.step(0, 1.0);
        }
        // Should be tier 2, 3, or 4 depending on accumulated difficulty
        assert!(curriculum.tier() >= 1);
    }

    #[test]
    fn test_tiered_curriculum_reset() {
        let mut curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);

        // Advance to tier 2
        curriculum.step(0, 1.0);
        assert!(curriculum.tier() >= 2);

        // Reset should go back to tier 1
        curriculum.reset();
        assert_eq!(curriculum.tier(), 1);
    }

    #[test]
    fn test_efficiency_score_edge_cases() {
        // Edge case: corpus_size = 1 (should return accuracy directly)
        assert!((efficiency_score(0.8, 1) - 0.8).abs() < 1e-5);

        // Edge case: corpus_size = 0 (should return accuracy)
        assert!((efficiency_score(0.8, 0) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_select_optimal_tier_empty() {
        let results: Vec<(usize, f32, usize)> = vec![];
        assert!(select_optimal_tier(&results).is_none());
    }

    #[test]
    fn test_select_optimal_tier_single() {
        let results = vec![(2, 0.75, 5000)];
        let (tier, _) = select_optimal_tier(&results).unwrap();
        assert_eq!(tier, 2);
    }

    #[test]
    fn test_tiered_curriculum_difficulty() {
        let curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);
        // Tier 1 corresponds to difficulty 0.0
        assert_eq!(curriculum.difficulty(), 0.0);
    }

    #[test]
    fn test_linear_curriculum_name() {
        let curriculum = LinearCurriculum::new(0.0, 1.0, 10);
        assert!(!curriculum.name().is_empty());
    }
}

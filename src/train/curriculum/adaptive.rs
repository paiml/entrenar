//! Adaptive curriculum scheduler for error-specific training

use std::collections::HashMap;

use super::CurriculumScheduler;

/// Adaptive curriculum that adjusts based on error class performance
///
/// Tracks accuracy per error class and increases difficulty for
/// well-learned classes while maintaining focus on struggling classes.
///
/// Supports the CITL adaptive tier selection pattern.
#[derive(Debug, Clone)]
pub struct AdaptiveCurriculum {
    /// Accuracy per error class
    pub(crate) class_accuracy: HashMap<String, f32>,
    /// Attempts per error class
    pub(crate) class_attempts: HashMap<String, usize>,
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

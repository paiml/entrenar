//! Pruning configuration module
//!
//! Provides configuration types for pruning methods, schedules,
//! and parameters.

use crate::prune::schedule::PruningSchedule;
use serde::{Deserialize, Serialize};

/// Pruning method selection.
///
/// Each method has different trade-offs between accuracy and computational cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PruneMethod {
    /// Magnitude-based pruning (Han et al., 2015)
    /// Simple, fast, no calibration required.
    #[default]
    Magnitude,

    /// Wanda: Weight and Activation pruning (Sun et al., 2023)
    /// Requires calibration data for activation statistics.
    Wanda,

    /// SparseGPT: Hessian-based pruning (Frantar & Alistarh, 2023)
    /// Most accurate but computationally expensive.
    SparseGpt,

    /// Minitron depth pruning - removes entire layers.
    MinitronDepth,

    /// Minitron width pruning - removes channels.
    MinitronWidth,
}

impl PruneMethod {
    /// Check if this method requires calibration data.
    pub fn requires_calibration(&self) -> bool {
        matches!(
            self,
            PruneMethod::Wanda
                | PruneMethod::SparseGpt
                | PruneMethod::MinitronDepth
                | PruneMethod::MinitronWidth
        )
    }

    /// Get the display name for this method.
    pub fn display_name(&self) -> &'static str {
        match self {
            PruneMethod::Magnitude => "Magnitude",
            PruneMethod::Wanda => "Wanda",
            PruneMethod::SparseGpt => "SparseGPT",
            PruneMethod::MinitronDepth => "Minitron (Depth)",
            PruneMethod::MinitronWidth => "Minitron (Width)",
        }
    }
}

/// Sparsity pattern selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SparsityPatternConfig {
    /// Unstructured sparsity - any weight can be pruned.
    #[default]
    Unstructured,

    /// N:M structured sparsity (e.g., 2:4 for NVIDIA Ampere).
    #[serde(rename = "nm")]
    NM {
        /// Number of non-zero elements per group.
        n: usize,
        /// Group size.
        m: usize,
    },

    /// Block sparsity - entire blocks pruned together.
    Block {
        /// Block height.
        height: usize,
        /// Block width.
        width: usize,
    },

    /// Row sparsity - entire output channels pruned.
    Row,

    /// Column sparsity - entire input channels pruned.
    Column,
}

impl SparsityPatternConfig {
    /// Create 2:4 sparsity pattern for NVIDIA Ampere.
    pub fn nm_2_4() -> Self {
        SparsityPatternConfig::NM { n: 2, m: 4 }
    }

    /// Create 4:8 sparsity pattern.
    pub fn nm_4_8() -> Self {
        SparsityPatternConfig::NM { n: 4, m: 8 }
    }

    /// Get the theoretical sparsity for this pattern.
    pub fn theoretical_sparsity(&self) -> f32 {
        match self {
            SparsityPatternConfig::Unstructured => 0.0, // Variable
            SparsityPatternConfig::NM { n, m } => 1.0 - (*n as f32 / *m as f32),
            SparsityPatternConfig::Block { .. } => 0.0, // Variable
            SparsityPatternConfig::Row => 0.0,          // Variable
            SparsityPatternConfig::Column => 0.0,       // Variable
        }
    }
}

/// Configuration for pruning operations.
///
/// # Example
///
/// ```
/// use entrenar::prune::{PruningConfig, PruningSchedule, PruneMethod};
///
/// let config = PruningConfig::default()
///     .with_method(PruneMethod::Wanda)
///     .with_schedule(PruningSchedule::Gradual {
///         start_step: 1000,
///         end_step: 5000,
///         initial_sparsity: 0.0,
///         final_sparsity: 0.5,
///         frequency: 100,
///     })
///     .with_target_sparsity(0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning method to use.
    method: PruneMethod,

    /// Target sparsity (0.0 to 1.0).
    target_sparsity: f32,

    /// Sparsity pattern.
    pattern: SparsityPatternConfig,

    /// Pruning schedule.
    schedule: PruningSchedule,

    /// Whether to fine-tune after pruning.
    fine_tune_after_pruning: bool,

    /// Number of fine-tuning steps.
    fine_tune_steps: usize,

    /// Learning rate for fine-tuning.
    fine_tune_lr: f32,

    /// Whether to skip first and last layers (recommended for LLMs).
    skip_embed_layers: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruneMethod::default(),
            target_sparsity: 0.5,
            pattern: SparsityPatternConfig::default(),
            schedule: PruningSchedule::default(),
            fine_tune_after_pruning: true,
            fine_tune_steps: 1000,
            fine_tune_lr: 1e-5,
            skip_embed_layers: true,
        }
    }
}

impl PruningConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pruning method.
    pub fn with_method(mut self, method: PruneMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the target sparsity.
    pub fn with_target_sparsity(mut self, sparsity: f32) -> Self {
        self.target_sparsity = sparsity.clamp(0.0, 1.0);
        self
    }

    /// Set the sparsity pattern.
    pub fn with_pattern(mut self, pattern: SparsityPatternConfig) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the pruning schedule.
    pub fn with_schedule(mut self, schedule: PruningSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Enable or disable fine-tuning after pruning.
    pub fn with_fine_tune(mut self, enabled: bool) -> Self {
        self.fine_tune_after_pruning = enabled;
        self
    }

    /// Set the number of fine-tuning steps.
    pub fn with_fine_tune_steps(mut self, steps: usize) -> Self {
        self.fine_tune_steps = steps;
        self
    }

    /// Set the fine-tuning learning rate.
    pub fn with_fine_tune_lr(mut self, lr: f32) -> Self {
        self.fine_tune_lr = lr;
        self
    }

    /// Enable or disable skipping embedding layers.
    pub fn with_skip_embed_layers(mut self, skip: bool) -> Self {
        self.skip_embed_layers = skip;
        self
    }

    /// Get the pruning method.
    pub fn method(&self) -> PruneMethod {
        self.method
    }

    /// Get the target sparsity.
    pub fn target_sparsity(&self) -> f32 {
        self.target_sparsity
    }

    /// Get the sparsity pattern.
    pub fn pattern(&self) -> &SparsityPatternConfig {
        &self.pattern
    }

    /// Get the pruning schedule.
    pub fn schedule(&self) -> &PruningSchedule {
        &self.schedule
    }

    /// Check if fine-tuning is enabled.
    pub fn fine_tune_after_pruning(&self) -> bool {
        self.fine_tune_after_pruning
    }

    /// Get fine-tuning steps.
    pub fn fine_tune_steps(&self) -> usize {
        self.fine_tune_steps
    }

    /// Get fine-tuning learning rate.
    pub fn fine_tune_lr(&self) -> f32 {
        self.fine_tune_lr
    }

    /// Check if embedding layers should be skipped.
    pub fn skip_embed_layers(&self) -> bool {
        self.skip_embed_layers
    }

    /// Check if this configuration requires calibration data.
    pub fn requires_calibration(&self) -> bool {
        self.method.requires_calibration()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        // Validate schedule
        self.schedule.validate()?;

        // Validate target sparsity
        if self.target_sparsity < 0.0 || self.target_sparsity > 1.0 {
            return Err(format!(
                "target_sparsity ({}) must be between 0.0 and 1.0",
                self.target_sparsity
            ));
        }

        // Validate N:M pattern
        if let SparsityPatternConfig::NM { n, m } = &self.pattern {
            if *n >= *m {
                return Err(format!("N ({n}) must be less than M ({m})"));
            }
            if *m == 0 {
                return Err("M cannot be 0".to_string());
            }
        }

        // Validate block pattern
        if let SparsityPatternConfig::Block { height, width } = &self.pattern {
            if *height == 0 || *width == 0 {
                return Err("Block dimensions must be non-zero".to_string());
            }
        }

        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PruneMethod Tests
    // =========================================================================

    #[test]
    fn test_prune_method_requires_calibration() {
        // TEST_ID: CFG-001
        assert!(
            !PruneMethod::Magnitude.requires_calibration(),
            "CFG-001 FALSIFIED: Magnitude should not require calibration"
        );
        assert!(
            PruneMethod::Wanda.requires_calibration(),
            "CFG-001 FALSIFIED: Wanda should require calibration"
        );
        assert!(
            PruneMethod::SparseGpt.requires_calibration(),
            "CFG-001 FALSIFIED: SparseGPT should require calibration"
        );
        assert!(
            PruneMethod::MinitronDepth.requires_calibration(),
            "CFG-001 FALSIFIED: MinitronDepth should require calibration"
        );
        assert!(
            PruneMethod::MinitronWidth.requires_calibration(),
            "CFG-001 FALSIFIED: MinitronWidth should require calibration"
        );
    }

    #[test]
    fn test_prune_method_display_names() {
        // TEST_ID: CFG-002
        assert_eq!(PruneMethod::Magnitude.display_name(), "Magnitude");
        assert_eq!(PruneMethod::Wanda.display_name(), "Wanda");
        assert_eq!(PruneMethod::SparseGpt.display_name(), "SparseGPT");
        assert_eq!(
            PruneMethod::MinitronDepth.display_name(),
            "Minitron (Depth)"
        );
        assert_eq!(
            PruneMethod::MinitronWidth.display_name(),
            "Minitron (Width)"
        );
    }

    #[test]
    fn test_prune_method_default() {
        // TEST_ID: CFG-003
        assert_eq!(
            PruneMethod::default(),
            PruneMethod::Magnitude,
            "CFG-003 FALSIFIED: Default method should be Magnitude"
        );
    }

    // =========================================================================
    // SparsityPatternConfig Tests
    // =========================================================================

    #[test]
    fn test_sparsity_pattern_nm_2_4() {
        // TEST_ID: CFG-010
        let pattern = SparsityPatternConfig::nm_2_4();
        match pattern {
            SparsityPatternConfig::NM { n, m } => {
                assert_eq!(n, 2);
                assert_eq!(m, 4);
            }
            _ => panic!("CFG-010 FALSIFIED: Expected NM pattern"),
        }
    }

    #[test]
    fn test_sparsity_pattern_nm_4_8() {
        // TEST_ID: CFG-011
        let pattern = SparsityPatternConfig::nm_4_8();
        match pattern {
            SparsityPatternConfig::NM { n, m } => {
                assert_eq!(n, 4);
                assert_eq!(m, 8);
            }
            _ => panic!("CFG-011 FALSIFIED: Expected NM pattern"),
        }
    }

    #[test]
    fn test_sparsity_pattern_theoretical_sparsity() {
        // TEST_ID: CFG-012
        // 2:4 = 50% sparsity (2 zeros out of 4)
        let nm_2_4 = SparsityPatternConfig::nm_2_4();
        assert!(
            (nm_2_4.theoretical_sparsity() - 0.5).abs() < 1e-6,
            "CFG-012 FALSIFIED: 2:4 should have 50% sparsity"
        );

        // 4:8 = 50% sparsity
        let nm_4_8 = SparsityPatternConfig::nm_4_8();
        assert!(
            (nm_4_8.theoretical_sparsity() - 0.5).abs() < 1e-6,
            "CFG-012 FALSIFIED: 4:8 should have 50% sparsity"
        );

        // Unstructured has variable sparsity (returns 0 as placeholder)
        let unstructured = SparsityPatternConfig::Unstructured;
        assert_eq!(unstructured.theoretical_sparsity(), 0.0);
    }

    #[test]
    fn test_sparsity_pattern_block_theoretical_sparsity() {
        // TEST_ID: CFG-014
        // Block patterns have variable sparsity (returns 0 as placeholder)
        let block = SparsityPatternConfig::Block {
            height: 4,
            width: 4,
        };
        assert_eq!(
            block.theoretical_sparsity(),
            0.0,
            "CFG-014 FALSIFIED: Block should return 0.0 for variable sparsity"
        );
    }

    #[test]
    fn test_sparsity_pattern_row_theoretical_sparsity() {
        // TEST_ID: CFG-015
        // Row patterns have variable sparsity (returns 0 as placeholder)
        let row = SparsityPatternConfig::Row;
        assert_eq!(
            row.theoretical_sparsity(),
            0.0,
            "CFG-015 FALSIFIED: Row should return 0.0 for variable sparsity"
        );
    }

    #[test]
    fn test_sparsity_pattern_column_theoretical_sparsity() {
        // TEST_ID: CFG-016
        // Column patterns have variable sparsity (returns 0 as placeholder)
        let column = SparsityPatternConfig::Column;
        assert_eq!(
            column.theoretical_sparsity(),
            0.0,
            "CFG-016 FALSIFIED: Column should return 0.0 for variable sparsity"
        );
    }

    #[test]
    fn test_sparsity_pattern_default() {
        // TEST_ID: CFG-013
        assert_eq!(
            SparsityPatternConfig::default(),
            SparsityPatternConfig::Unstructured,
            "CFG-013 FALSIFIED: Default pattern should be Unstructured"
        );
    }

    // =========================================================================
    // PruningConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default_values() {
        // TEST_ID: CFG-020
        let config = PruningConfig::default();
        assert_eq!(config.method(), PruneMethod::Magnitude);
        assert!((config.target_sparsity() - 0.5).abs() < 1e-6);
        assert_eq!(*config.pattern(), SparsityPatternConfig::Unstructured);
        assert!(config.fine_tune_after_pruning());
        assert_eq!(config.fine_tune_steps(), 1000);
        assert!((config.fine_tune_lr() - 1e-5).abs() < 1e-10);
        assert!(config.skip_embed_layers());
    }

    #[test]
    fn test_config_builder_pattern() {
        // TEST_ID: CFG-021
        let config = PruningConfig::new()
            .with_method(PruneMethod::Wanda)
            .with_target_sparsity(0.7)
            .with_pattern(SparsityPatternConfig::nm_2_4())
            .with_fine_tune(false)
            .with_fine_tune_steps(500)
            .with_fine_tune_lr(1e-4)
            .with_skip_embed_layers(false);

        assert_eq!(config.method(), PruneMethod::Wanda);
        assert!((config.target_sparsity() - 0.7).abs() < 1e-6);
        match config.pattern() {
            SparsityPatternConfig::NM { n, m } => {
                assert_eq!(*n, 2);
                assert_eq!(*m, 4);
            }
            _ => panic!("CFG-021 FALSIFIED: Expected NM pattern"),
        }
        assert!(!config.fine_tune_after_pruning());
        assert_eq!(config.fine_tune_steps(), 500);
        assert!((config.fine_tune_lr() - 1e-4).abs() < 1e-10);
        assert!(!config.skip_embed_layers());
    }

    #[test]
    fn test_config_target_sparsity_clamped() {
        // TEST_ID: CFG-022
        let config = PruningConfig::new().with_target_sparsity(1.5);
        assert_eq!(
            config.target_sparsity(),
            1.0,
            "CFG-022 FALSIFIED: Sparsity should be clamped to 1.0"
        );

        let config2 = PruningConfig::new().with_target_sparsity(-0.5);
        assert_eq!(
            config2.target_sparsity(),
            0.0,
            "CFG-022 FALSIFIED: Sparsity should be clamped to 0.0"
        );
    }

    #[test]
    fn test_config_requires_calibration() {
        // TEST_ID: CFG-023
        let magnitude_config = PruningConfig::new().with_method(PruneMethod::Magnitude);
        assert!(
            !magnitude_config.requires_calibration(),
            "CFG-023 FALSIFIED: Magnitude config should not require calibration"
        );

        let wanda_config = PruningConfig::new().with_method(PruneMethod::Wanda);
        assert!(
            wanda_config.requires_calibration(),
            "CFG-023 FALSIFIED: Wanda config should require calibration"
        );
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_config_validate_valid() {
        // TEST_ID: CFG-030
        let config = PruningConfig::default();
        assert!(
            config.validate().is_ok(),
            "CFG-030 FALSIFIED: Default config should be valid"
        );
    }

    #[test]
    fn test_config_validate_invalid_nm() {
        // TEST_ID: CFG-031
        let config = PruningConfig::new().with_pattern(SparsityPatternConfig::NM {
            n: 5, // Invalid: n >= m
            m: 4,
        });
        assert!(
            config.validate().is_err(),
            "CFG-031 FALSIFIED: N >= M should be invalid"
        );
    }

    #[test]
    fn test_config_validate_zero_m() {
        // TEST_ID: CFG-032
        let config = PruningConfig::new().with_pattern(SparsityPatternConfig::NM { n: 0, m: 0 });
        assert!(
            config.validate().is_err(),
            "CFG-032 FALSIFIED: M=0 should be invalid"
        );
    }

    #[test]
    fn test_config_validate_zero_block() {
        // TEST_ID: CFG-033
        let config = PruningConfig::new().with_pattern(SparsityPatternConfig::Block {
            height: 0,
            width: 4,
        });
        assert!(
            config.validate().is_err(),
            "CFG-033 FALSIFIED: Zero block dimension should be invalid"
        );
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_config_serialize_json() {
        // TEST_ID: CFG-040
        let config = PruningConfig::new()
            .with_method(PruneMethod::Wanda)
            .with_target_sparsity(0.5);

        let json = serde_json::to_string(&config).unwrap();
        assert!(
            json.contains("wanda"),
            "CFG-040 FALSIFIED: JSON should contain method name"
        );

        let deserialized: PruningConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.method(),
            PruneMethod::Wanda,
            "CFG-040 FALSIFIED: Deserialized method should match"
        );
    }

    #[test]
    fn test_config_serialize_yaml() {
        // TEST_ID: CFG-041
        let config = PruningConfig::new()
            .with_method(PruneMethod::SparseGpt)
            .with_pattern(SparsityPatternConfig::nm_2_4());

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(
            yaml.contains("sparse_gpt"),
            "CFG-041 FALSIFIED: YAML should contain method name"
        );
    }

    #[test]
    fn test_config_deserialize_from_yaml() {
        // TEST_ID: CFG-042
        let yaml = r"
method: wanda
target_sparsity: 0.5
pattern:
  type: nm
  n: 2
  m: 4
schedule:
  type: one_shot
  step: 1000
fine_tune_after_pruning: true
fine_tune_steps: 500
fine_tune_lr: 0.00001
skip_embed_layers: true
";
        let config: PruningConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.method(), PruneMethod::Wanda);
        assert!((config.target_sparsity() - 0.5).abs() < 1e-6);
        match config.pattern() {
            SparsityPatternConfig::NM { n, m } => {
                assert_eq!(*n, 2);
                assert_eq!(*m, 4);
            }
            _ => panic!("CFG-042 FALSIFIED: Expected NM pattern"),
        }
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_config_clone() {
        // TEST_ID: CFG-050
        let config = PruningConfig::new()
            .with_method(PruneMethod::Wanda)
            .with_target_sparsity(0.7);

        let cloned = config.clone();
        assert_eq!(
            config.method(),
            cloned.method(),
            "CFG-050 FALSIFIED: Cloned method should match"
        );
        assert!(
            (config.target_sparsity() - cloned.target_sparsity()).abs() < 1e-6,
            "CFG-050 FALSIFIED: Cloned target_sparsity should match"
        );
    }

    // =========================================================================
    // Debug Tests
    // =========================================================================

    #[test]
    fn test_config_debug() {
        // TEST_ID: CFG-060
        let config = PruningConfig::new().with_method(PruneMethod::Wanda);
        let debug = format!("{config:?}");
        assert!(
            debug.contains("Wanda"),
            "CFG-060 FALSIFIED: Debug should contain method name"
        );
    }

    #[test]
    fn test_pattern_debug() {
        // TEST_ID: CFG-061
        let pattern = SparsityPatternConfig::nm_2_4();
        let debug = format!("{pattern:?}");
        assert!(
            debug.contains("NM"),
            "CFG-061 FALSIFIED: Debug should contain pattern type"
        );
    }
}

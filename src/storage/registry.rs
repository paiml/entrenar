//! Model Registry with Staging Workflows (MLOPS-008)
//!
//! Kanban-style model lifecycle management.
//!
//! # Toyota Way: カンバン (Kanban)
//!
//! Visual workflow stages for model promotion with pull-based progression.
//! Models flow: None → Development → Staging → Production → Archived
//!
//! # Example
//!
//! ```ignore
//! use entrenar::storage::registry::{ModelRegistry, ModelStage, InMemoryRegistry};
//!
//! let mut registry = InMemoryRegistry::new();
//! registry.register_model("llama-7b-finetuned", "path/to/model.safetensors")?;
//! registry.transition_stage("llama-7b-finetuned", 1, ModelStage::Staging, Some("alice"))?;
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Model lifecycle stages (Kanban workflow)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelStage {
    /// Not assigned to any stage
    None,
    /// In active development
    Development,
    /// Being tested/validated
    Staging,
    /// Deployed and serving traffic
    Production,
    /// Retired from active use
    Archived,
}

impl ModelStage {
    /// Check if transition to target stage is valid
    pub fn can_transition_to(&self, target: ModelStage) -> bool {
        match (self, target) {
            // Any stage can go to Archived
            (_, ModelStage::Archived) => true,
            // None can go to Development
            (ModelStage::None, ModelStage::Development) => true,
            // Development can go to Staging
            (ModelStage::Development, ModelStage::Staging) => true,
            // Staging can go to Production
            (ModelStage::Staging, ModelStage::Production) => true,
            // Production can go back to Staging (rollback)
            (ModelStage::Production, ModelStage::Staging) => true,
            // Staging can go back to Development (rejected)
            (ModelStage::Staging, ModelStage::Development) => true,
            // Archived can be restored to Development
            (ModelStage::Archived, ModelStage::Development) => true,
            // Same stage is a no-op
            (a, b) if *a == b => true,
            _ => false,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelStage::None => "None",
            ModelStage::Development => "Development",
            ModelStage::Staging => "Staging",
            ModelStage::Production => "Production",
            ModelStage::Archived => "Archived",
        }
    }
}

impl std::fmt::Display for ModelStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Model version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name
    pub name: String,
    /// Version number (monotonically increasing)
    pub version: u32,
    /// Current stage
    pub stage: ModelStage,
    /// URI to model artifacts
    pub artifact_uri: String,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Tags for organization
    pub tags: HashMap<String, String>,
    /// Description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last promotion timestamp
    pub promoted_at: Option<DateTime<Utc>>,
    /// User who last promoted
    pub promoted_by: Option<String>,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(name: &str, version: u32, artifact_uri: &str) -> Self {
        Self {
            name: name.to_string(),
            version,
            stage: ModelStage::None,
            artifact_uri: artifact_uri.to_string(),
            metrics: HashMap::new(),
            tags: HashMap::new(),
            description: None,
            created_at: Utc::now(),
            promoted_at: None,
            promoted_by: None,
        }
    }

    /// Add a metric
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Comparison between two model versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// First version
    pub v1: u32,
    /// Second version
    pub v2: u32,
    /// Metric differences (positive = v2 is better for maximizing metrics)
    pub metric_diffs: HashMap<String, f64>,
    /// Whether v2 is better overall
    pub v2_is_better: bool,
    /// Summary of changes
    pub summary: String,
}

/// Stage transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTransition {
    /// Model name
    pub model_name: String,
    /// Version
    pub version: u32,
    /// Previous stage
    pub from_stage: ModelStage,
    /// New stage
    pub to_stage: ModelStage,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User who made the transition
    pub user: Option<String>,
    /// Reason for transition
    pub reason: Option<String>,
}

/// Promotion policy for stage transitions (Poka-yoke)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPolicy {
    /// Required metrics with thresholds
    pub required_metrics: Vec<MetricRequirement>,
    /// Minimum test coverage
    pub min_test_coverage: Option<f64>,
    /// Required number of approvals
    pub required_approvals: u32,
    /// Auto-promote if all requirements pass
    pub auto_promote_on_pass: bool,
    /// Target stage this policy applies to
    pub target_stage: ModelStage,
}

impl PromotionPolicy {
    /// Create a new promotion policy for a target stage
    pub fn new(target_stage: ModelStage) -> Self {
        Self {
            required_metrics: Vec::new(),
            min_test_coverage: None,
            required_approvals: 0,
            auto_promote_on_pass: false,
            target_stage,
        }
    }

    /// Add a metric requirement
    pub fn require_metric(mut self, name: &str, comparison: Comparison, threshold: f64) -> Self {
        self.required_metrics.push(MetricRequirement {
            name: name.to_string(),
            comparison,
            threshold,
        });
        self
    }

    /// Set minimum test coverage
    pub fn require_coverage(mut self, coverage: f64) -> Self {
        self.min_test_coverage = Some(coverage);
        self
    }

    /// Set required approvals
    pub fn require_approvals(mut self, count: u32) -> Self {
        self.required_approvals = count;
        self
    }

    /// Enable auto-promotion
    pub fn auto_promote(mut self) -> Self {
        self.auto_promote_on_pass = true;
        self
    }

    /// Check if a model version meets the policy requirements
    pub fn check(&self, model: &ModelVersion, approvals: u32) -> PolicyCheckResult {
        let mut failed_requirements = Vec::new();

        // Check metrics
        for req in &self.required_metrics {
            if let Some(&value) = model.metrics.get(&req.name) {
                if !req.comparison.check(value, req.threshold) {
                    failed_requirements.push(format!(
                        "Metric '{}' = {} does not satisfy {} {}",
                        req.name,
                        value,
                        req.comparison.as_str(),
                        req.threshold
                    ));
                }
            } else {
                failed_requirements.push(format!("Missing required metric '{}'", req.name));
            }
        }

        // Check test coverage
        if let Some(min_coverage) = self.min_test_coverage {
            if let Some(&coverage) = model.metrics.get("test_coverage") {
                if coverage < min_coverage {
                    failed_requirements.push(format!(
                        "Test coverage {coverage} < required {min_coverage}"
                    ));
                }
            } else {
                failed_requirements.push("Missing test_coverage metric".to_string());
            }
        }

        // Check approvals
        if approvals < self.required_approvals {
            failed_requirements.push(format!(
                "Approvals {} < required {}",
                approvals, self.required_approvals
            ));
        }

        PolicyCheckResult {
            passed: failed_requirements.is_empty(),
            failed_requirements,
        }
    }
}

/// Metric requirement for promotion policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRequirement {
    /// Metric name
    pub name: String,
    /// Comparison operator
    pub comparison: Comparison,
    /// Threshold value
    pub threshold: f64,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparison {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
}

impl Comparison {
    /// Check if value satisfies comparison with threshold
    pub fn check(&self, value: f64, threshold: f64) -> bool {
        match self {
            Comparison::Gt => value > threshold,
            Comparison::Gte => value >= threshold,
            Comparison::Lt => value < threshold,
            Comparison::Lte => value <= threshold,
            Comparison::Eq => (value - threshold).abs() < f64::EPSILON,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Comparison::Gt => ">",
            Comparison::Gte => ">=",
            Comparison::Lt => "<",
            Comparison::Lte => "<=",
            Comparison::Eq => "==",
        }
    }
}

/// Result of policy check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCheckResult {
    /// Whether all requirements passed
    pub passed: bool,
    /// List of failed requirements
    pub failed_requirements: Vec<String>,
}

/// Registry errors
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Version not found: {0} v{1}")]
    VersionNotFound(String, u32),

    #[error("Invalid stage transition from {0} to {1}")]
    InvalidTransition(ModelStage, ModelStage),

    #[error("Policy check failed: {0}")]
    PolicyFailed(String),

    #[error("Registry error: {0}")]
    Internal(String),
}

/// Result type for registry operations
pub type Result<T> = std::result::Result<T, RegistryError>;

/// Model registry trait
pub trait ModelRegistry: Send + Sync {
    /// Register a new model version
    fn register_model(&mut self, name: &str, artifact_uri: &str) -> Result<ModelVersion>;

    /// Get a model version
    fn get_model(&self, name: &str, version: u32) -> Result<ModelVersion>;

    /// Get latest version of a model
    fn get_latest(&self, name: &str) -> Result<ModelVersion>;

    /// Get latest version at a specific stage
    fn get_latest_by_stage(&self, name: &str, stage: ModelStage) -> Option<ModelVersion>;

    /// List all versions of a model
    fn list_versions(&self, name: &str) -> Result<Vec<ModelVersion>>;

    /// Transition model to new stage
    fn transition_stage(
        &mut self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        user: Option<&str>,
    ) -> Result<()>;

    /// Compare two versions
    fn compare_versions(&self, name: &str, v1: u32, v2: u32) -> Result<VersionComparison>;

    /// Log metrics for a model version
    fn log_metrics(
        &mut self,
        name: &str,
        version: u32,
        metrics: HashMap<String, f64>,
    ) -> Result<()>;

    /// Get transition history for a model
    fn get_transition_history(&self, name: &str) -> Result<Vec<StageTransition>>;

    /// Set promotion policy for a stage
    fn set_policy(&mut self, policy: PromotionPolicy);

    /// Get promotion policy for a stage
    fn get_policy(&self, stage: ModelStage) -> Option<&PromotionPolicy>;

    /// Check if model can be promoted (with policy check)
    fn can_promote(
        &self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        approvals: u32,
    ) -> Result<PolicyCheckResult>;
}

/// In-memory model registry for testing
#[derive(Debug, Default)]
pub struct InMemoryRegistry {
    /// Models by name -> version -> ModelVersion
    models: HashMap<String, HashMap<u32, ModelVersion>>,
    /// Stage transition history
    transitions: Vec<StageTransition>,
    /// Promotion policies by stage
    policies: HashMap<ModelStage, PromotionPolicy>,
    /// Auto-rollback configuration
    rollback_enabled: HashMap<String, (String, f64)>, // model -> (metric, threshold)
}

impl InMemoryRegistry {
    /// Create a new in-memory registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable auto-rollback for a model
    pub fn enable_auto_rollback(&mut self, model: &str, metric: &str, threshold: f64) {
        self.rollback_enabled
            .insert(model.to_string(), (metric.to_string(), threshold));
    }

    /// Check if rollback is needed based on metrics
    pub fn check_rollback(&self, model: &str, current_metric: f64) -> bool {
        if let Some((_, threshold)) = self.rollback_enabled.get(model) {
            current_metric < *threshold
        } else {
            false
        }
    }

    /// Get next version number for a model
    fn next_version(&self, name: &str) -> u32 {
        self.models.get(name).map_or(1, |versions| {
            versions.keys().max().copied().unwrap_or(0) + 1
        })
    }
}

impl ModelRegistry for InMemoryRegistry {
    fn register_model(&mut self, name: &str, artifact_uri: &str) -> Result<ModelVersion> {
        let version = self.next_version(name);
        let model = ModelVersion::new(name, version, artifact_uri);

        self.models
            .entry(name.to_string())
            .or_default()
            .insert(version, model.clone());

        Ok(model)
    }

    fn get_model(&self, name: &str, version: u32) -> Result<ModelVersion> {
        self.models
            .get(name)
            .and_then(|versions| versions.get(&version))
            .cloned()
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))
    }

    fn get_latest(&self, name: &str) -> Result<ModelVersion> {
        self.models
            .get(name)
            .and_then(|versions| {
                let max_version = versions.keys().max()?;
                versions.get(max_version)
            })
            .cloned()
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))
    }

    fn get_latest_by_stage(&self, name: &str, stage: ModelStage) -> Option<ModelVersion> {
        self.models.get(name).and_then(|versions| {
            versions
                .values()
                .filter(|m| m.stage == stage)
                .max_by_key(|m| m.version)
                .cloned()
        })
    }

    fn list_versions(&self, name: &str) -> Result<Vec<ModelVersion>> {
        self.models
            .get(name)
            .map(|versions| {
                let mut v: Vec<_> = versions.values().cloned().collect();
                v.sort_by_key(|m| m.version);
                v
            })
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))
    }

    fn transition_stage(
        &mut self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        user: Option<&str>,
    ) -> Result<()> {
        let model = self
            .models
            .get_mut(name)
            .and_then(|versions| versions.get_mut(&version))
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))?;

        if !model.stage.can_transition_to(target_stage) {
            return Err(RegistryError::InvalidTransition(model.stage, target_stage));
        }

        let from_stage = model.stage;
        model.stage = target_stage;
        model.promoted_at = Some(Utc::now());
        model.promoted_by = user.map(ToString::to_string);

        // Record transition
        self.transitions.push(StageTransition {
            model_name: name.to_string(),
            version,
            from_stage,
            to_stage: target_stage,
            timestamp: Utc::now(),
            user: user.map(ToString::to_string),
            reason: None,
        });

        Ok(())
    }

    fn compare_versions(&self, name: &str, v1: u32, v2: u32) -> Result<VersionComparison> {
        let m1 = self.get_model(name, v1)?;
        let m2 = self.get_model(name, v2)?;

        let mut metric_diffs = HashMap::new();
        let mut v2_better_count = 0;
        let mut total_comparisons = 0;

        // Compare all metrics from both versions
        let all_metrics: std::collections::HashSet<_> =
            m1.metrics.keys().chain(m2.metrics.keys()).collect();

        for metric in all_metrics {
            let val1 = m1.metrics.get(metric).copied().unwrap_or(0.0);
            let val2 = m2.metrics.get(metric).copied().unwrap_or(0.0);
            let diff = val2 - val1;
            metric_diffs.insert(metric.clone(), diff);

            // Assume higher is better for most metrics
            if diff > 0.0 {
                v2_better_count += 1;
            }
            total_comparisons += 1;
        }

        let v2_is_better = total_comparisons > 0 && v2_better_count > total_comparisons / 2;

        let summary = if v2_is_better {
            format!(
                "Version {v2} is better than {v1} on {v2_better_count}/{total_comparisons} metrics"
            )
        } else {
            format!("Version {v2} is not definitively better than {v1}")
        };

        Ok(VersionComparison {
            v1,
            v2,
            metric_diffs,
            v2_is_better,
            summary,
        })
    }

    fn log_metrics(
        &mut self,
        name: &str,
        version: u32,
        metrics: HashMap<String, f64>,
    ) -> Result<()> {
        let model = self
            .models
            .get_mut(name)
            .and_then(|versions| versions.get_mut(&version))
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))?;

        model.metrics.extend(metrics);
        Ok(())
    }

    fn get_transition_history(&self, name: &str) -> Result<Vec<StageTransition>> {
        let history: Vec<_> = self
            .transitions
            .iter()
            .filter(|t| t.model_name == name)
            .cloned()
            .collect();

        if history.is_empty() && !self.models.contains_key(name) {
            return Err(RegistryError::ModelNotFound(name.to_string()));
        }

        Ok(history)
    }

    fn set_policy(&mut self, policy: PromotionPolicy) {
        self.policies.insert(policy.target_stage, policy);
    }

    fn get_policy(&self, stage: ModelStage) -> Option<&PromotionPolicy> {
        self.policies.get(&stage)
    }

    fn can_promote(
        &self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        approvals: u32,
    ) -> Result<PolicyCheckResult> {
        let model = self.get_model(name, version)?;

        // Check stage transition validity
        if !model.stage.can_transition_to(target_stage) {
            return Ok(PolicyCheckResult {
                passed: false,
                failed_requirements: vec![format!(
                    "Cannot transition from {} to {}",
                    model.stage, target_stage
                )],
            });
        }

        // Check policy if exists
        if let Some(policy) = self.policies.get(&target_stage) {
            Ok(policy.check(&model, approvals))
        } else {
            // No policy = always allowed
            Ok(PolicyCheckResult {
                passed: true,
                failed_requirements: Vec::new(),
            })
        }
    }
}

// =============================================================================
// Tests (TDD - Written First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ModelStage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stage_none_to_development() {
        assert!(ModelStage::None.can_transition_to(ModelStage::Development));
    }

    #[test]
    fn test_stage_development_to_staging() {
        assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_staging_to_production() {
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_stage_production_rollback_to_staging() {
        assert!(ModelStage::Production.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_any_to_archived() {
        assert!(ModelStage::None.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Development.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_stage_invalid_transitions() {
        assert!(!ModelStage::None.can_transition_to(ModelStage::Production));
        assert!(!ModelStage::Development.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(ModelStage::Production.to_string(), "Production");
        assert_eq!(ModelStage::Development.as_str(), "Development");
    }

    // -------------------------------------------------------------------------
    // ModelVersion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_version_new() {
        let model = ModelVersion::new("test-model", 1, "/path/to/model");
        assert_eq!(model.name, "test-model");
        assert_eq!(model.version, 1);
        assert_eq!(model.stage, ModelStage::None);
    }

    #[test]
    fn test_model_version_with_metric() {
        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.95);
        assert_eq!(model.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_model_version_with_tag() {
        let model = ModelVersion::new("test", 1, "/path").with_tag("framework", "pytorch");
        assert_eq!(model.tags.get("framework"), Some(&"pytorch".to_string()));
    }

    #[test]
    fn test_model_version_with_description() {
        let model = ModelVersion::new("test", 1, "/path").with_description("A test model");
        assert_eq!(model.description, Some("A test model".to_string()));
    }

    // -------------------------------------------------------------------------
    // Comparison Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_comparison_gt() {
        assert!(Comparison::Gt.check(0.96, 0.95));
        assert!(!Comparison::Gt.check(0.95, 0.95));
    }

    #[test]
    fn test_comparison_gte() {
        assert!(Comparison::Gte.check(0.95, 0.95));
        assert!(Comparison::Gte.check(0.96, 0.95));
    }

    #[test]
    fn test_comparison_lt() {
        assert!(Comparison::Lt.check(0.5, 1.0));
        assert!(!Comparison::Lt.check(1.0, 1.0));
    }

    #[test]
    fn test_comparison_eq() {
        assert!(Comparison::Eq.check(0.95, 0.95));
        assert!(!Comparison::Eq.check(0.95, 0.96));
    }

    // -------------------------------------------------------------------------
    // PromotionPolicy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_promotion_policy_new() {
        let policy = PromotionPolicy::new(ModelStage::Production);
        assert_eq!(policy.target_stage, ModelStage::Production);
        assert!(policy.required_metrics.is_empty());
    }

    #[test]
    fn test_promotion_policy_require_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        assert_eq!(policy.required_metrics.len(), 1);
        assert_eq!(policy.required_metrics[0].name, "accuracy");
    }

    #[test]
    fn test_promotion_policy_check_pass() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.96);

        let result = policy.check(&model, 0);
        assert!(result.passed);
    }

    #[test]
    fn test_promotion_policy_check_fail_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.90);

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(!result.failed_requirements.is_empty());
    }

    #[test]
    fn test_promotion_policy_check_fail_missing_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path");

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(result.failed_requirements[0].contains("Missing"));
    }

    #[test]
    fn test_promotion_policy_check_approvals() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_approvals(2);

        let model = ModelVersion::new("test", 1, "/path");

        // Not enough approvals
        let result = policy.check(&model, 1);
        assert!(!result.passed);

        // Enough approvals
        let result = policy.check(&model, 2);
        assert!(result.passed);
    }

    #[test]
    fn test_promotion_policy_check_coverage() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_coverage(0.90);

        let model = ModelVersion::new("test", 1, "/path").with_metric("test_coverage", 0.85);

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(result.failed_requirements[0].contains("coverage"));
    }

    // -------------------------------------------------------------------------
    // InMemoryRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_register_model() {
        let mut registry = InMemoryRegistry::new();
        let model = registry.register_model("test-model", "/path/v1").unwrap();

        assert_eq!(model.name, "test-model");
        assert_eq!(model.version, 1);
    }

    #[test]
    fn test_registry_register_multiple_versions() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();
        let model2 = registry.register_model("test-model", "/path/v2").unwrap();

        assert_eq!(model2.version, 2);
    }

    #[test]
    fn test_registry_get_model() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        let model = registry.get_model("test-model", 1).unwrap();
        assert_eq!(model.artifact_uri, "/path/v1");
    }

    #[test]
    fn test_registry_get_model_not_found() {
        let registry = InMemoryRegistry::new();
        let result = registry.get_model("nonexistent", 1);
        assert!(matches!(result, Err(RegistryError::VersionNotFound(_, _))));
    }

    #[test]
    fn test_registry_get_latest() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();
        registry.register_model("test-model", "/path/v2").unwrap();

        let latest = registry.get_latest("test-model").unwrap();
        assert_eq!(latest.version, 2);
    }

    #[test]
    fn test_registry_get_latest_by_stage() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();
        registry.register_model("test-model", "/path/v2").unwrap();

        registry
            .transition_stage("test-model", 1, ModelStage::Development, None)
            .unwrap();
        registry
            .transition_stage("test-model", 2, ModelStage::Development, None)
            .unwrap();
        registry
            .transition_stage("test-model", 2, ModelStage::Staging, None)
            .unwrap();

        let latest_dev = registry.get_latest_by_stage("test-model", ModelStage::Development);
        let latest_staging = registry.get_latest_by_stage("test-model", ModelStage::Staging);

        assert_eq!(latest_dev.map(|m| m.version), Some(1));
        assert_eq!(latest_staging.map(|m| m.version), Some(2));
    }

    #[test]
    fn test_registry_list_versions() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();
        registry.register_model("test-model", "/path/v2").unwrap();

        let versions = registry.list_versions("test-model").unwrap();
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);
    }

    #[test]
    fn test_registry_transition_stage() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        registry
            .transition_stage("test-model", 1, ModelStage::Development, Some("alice"))
            .unwrap();

        let model = registry.get_model("test-model", 1).unwrap();
        assert_eq!(model.stage, ModelStage::Development);
        assert_eq!(model.promoted_by, Some("alice".to_string()));
    }

    #[test]
    fn test_registry_transition_invalid() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        // Try to go directly to Production from None
        let result = registry.transition_stage("test-model", 1, ModelStage::Production, None);
        assert!(matches!(
            result,
            Err(RegistryError::InvalidTransition(_, _))
        ));
    }

    #[test]
    fn test_registry_compare_versions() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();
        registry.register_model("test-model", "/path/v2").unwrap();

        let mut metrics1 = HashMap::new();
        metrics1.insert("accuracy".to_string(), 0.90);
        registry.log_metrics("test-model", 1, metrics1).unwrap();

        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), 0.95);
        registry.log_metrics("test-model", 2, metrics2).unwrap();

        let comparison = registry.compare_versions("test-model", 1, 2).unwrap();
        assert!(comparison.v2_is_better);
        assert!(comparison.metric_diffs.get("accuracy").unwrap() > &0.0);
    }

    #[test]
    fn test_registry_log_metrics() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("f1".to_string(), 0.92);
        registry.log_metrics("test-model", 1, metrics).unwrap();

        let model = registry.get_model("test-model", 1).unwrap();
        assert_eq!(model.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(model.metrics.get("f1"), Some(&0.92));
    }

    #[test]
    fn test_registry_get_transition_history() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        registry
            .transition_stage("test-model", 1, ModelStage::Development, None)
            .unwrap();
        registry
            .transition_stage("test-model", 1, ModelStage::Staging, None)
            .unwrap();

        let history = registry.get_transition_history("test-model").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].to_stage, ModelStage::Development);
        assert_eq!(history[1].to_stage, ModelStage::Staging);
    }

    #[test]
    fn test_registry_set_and_get_policy() {
        let mut registry = InMemoryRegistry::new();

        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        registry.set_policy(policy);

        let retrieved = registry.get_policy(ModelStage::Production);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().required_metrics.len(), 1);
    }

    #[test]
    fn test_registry_can_promote_with_policy() {
        let mut registry = InMemoryRegistry::new();
        registry.register_model("test-model", "/path/v1").unwrap();

        registry
            .transition_stage("test-model", 1, ModelStage::Development, None)
            .unwrap();
        registry
            .transition_stage("test-model", 1, ModelStage::Staging, None)
            .unwrap();

        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );
        registry.set_policy(policy);

        // Without required metrics
        let result = registry
            .can_promote("test-model", 1, ModelStage::Production, 0)
            .unwrap();
        assert!(!result.passed);

        // Add metrics
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.96);
        registry.log_metrics("test-model", 1, metrics).unwrap();

        // With required metrics
        let result = registry
            .can_promote("test-model", 1, ModelStage::Production, 0)
            .unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_registry_auto_rollback() {
        let mut registry = InMemoryRegistry::new();
        registry.enable_auto_rollback("test-model", "accuracy", 0.90);

        // Should rollback
        assert!(registry.check_rollback("test-model", 0.85));
        // Should not rollback
        assert!(!registry.check_rollback("test-model", 0.95));
        // No rollback config
        assert!(!registry.check_rollback("other-model", 0.50));
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
        fn prop_stage_self_transition(stage in any::<u8>().prop_map(|n| match n % 5 {
            0 => ModelStage::None,
            1 => ModelStage::Development,
            2 => ModelStage::Staging,
            3 => ModelStage::Production,
            _ => ModelStage::Archived,
        })) {
            // Self-transition is always valid
            prop_assert!(stage.can_transition_to(stage));
        }

        #[test]
        fn prop_all_stages_can_archive(stage in any::<u8>().prop_map(|n| match n % 5 {
            0 => ModelStage::None,
            1 => ModelStage::Development,
            2 => ModelStage::Staging,
            3 => ModelStage::Production,
            _ => ModelStage::Archived,
        })) {
            // All stages can transition to Archived
            prop_assert!(stage.can_transition_to(ModelStage::Archived));
        }

        #[test]
        fn prop_version_numbers_increase(count in 1usize..20) {
            let mut registry = InMemoryRegistry::new();
            let mut last_version = 0u32;

            for _ in 0..count {
                let model = registry.register_model("test", "/path").unwrap();
                prop_assert!(model.version > last_version);
                last_version = model.version;
            }
        }

        #[test]
        fn prop_comparison_consistent(value in -1000.0f64..1000.0, threshold in -1000.0f64..1000.0) {
            // Gt and Lte are complementary
            let gt = Comparison::Gt.check(value, threshold);
            let lte = Comparison::Lte.check(value, threshold);
            prop_assert!(gt != lte || value == threshold);
        }

        #[test]
        fn prop_policy_check_deterministic(
            accuracy in 0.0f64..1.0,
            threshold in 0.0f64..1.0,
            approvals in 0u32..10,
            required_approvals in 0u32..10
        ) {
            let policy = PromotionPolicy::new(ModelStage::Production)
                .require_metric("accuracy", Comparison::Gte, threshold)
                .require_approvals(required_approvals);

            let model = ModelVersion::new("test", 1, "/path")
                .with_metric("accuracy", accuracy);

            let result1 = policy.check(&model, approvals);
            let result2 = policy.check(&model, approvals);

            prop_assert_eq!(result1.passed, result2.passed);
        }

        #[test]
        fn prop_metrics_preserved(
            metrics in prop::collection::hash_map(
                "[a-z]{1,10}",
                0.0f64..1.0,
                1..10
            )
        ) {
            let mut registry = InMemoryRegistry::new();
            registry.register_model("test", "/path").unwrap();
            registry.log_metrics("test", 1, metrics.clone()).unwrap();

            let model = registry.get_model("test", 1).unwrap();
            for (key, value) in &metrics {
                prop_assert_eq!(model.metrics.get(key), Some(value));
            }
        }
    }
}

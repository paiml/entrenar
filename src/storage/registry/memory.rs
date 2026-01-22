//! In-memory model registry implementation

use chrono::Utc;
use std::collections::HashMap;

use super::comparison::VersionComparison;
use super::error::{RegistryError, Result};
use super::policy::{PolicyCheckResult, PromotionPolicy};
use super::stage::ModelStage;
use super::traits::ModelRegistry;
use super::transition::StageTransition;
use super::version::ModelVersion;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::registry::comparison::Comparison;

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

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

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

//! Ruchy Session Bridge (ENT-033)
//!
//! Provides session bridge for preserving training history from Ruchy
//! interactive sessions into Entrenar artifacts.
//!
//! This module is feature-gated behind `ruchy-sessions`.

use crate::research::{ArtifactType, Author, ContributorRole, License, ResearchArtifact};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors that can occur during session bridging.
#[derive(Debug, thiserror::Error)]
pub enum RuchyBridgeError {
    /// Session data is invalid or corrupted
    #[error("Invalid session data: {0}")]
    InvalidSession(String),

    /// Required session field is missing
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Session has no training history
    #[error("Session has no training history")]
    NoTrainingHistory,
}

/// Training metrics from a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Loss values over time
    pub loss_history: Vec<f64>,
    /// Accuracy values over time (optional)
    pub accuracy_history: Vec<f64>,
    /// Learning rate schedule
    pub lr_history: Vec<f64>,
    /// Gradient norms (for debugging)
    pub grad_norm_history: Vec<f64>,
    /// Custom metrics
    pub custom: HashMap<String, Vec<f64>>,
}

impl SessionMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a loss value.
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);
    }

    /// Add an accuracy value.
    pub fn add_accuracy(&mut self, accuracy: f64) {
        self.accuracy_history.push(accuracy);
    }

    /// Add a learning rate value.
    pub fn add_lr(&mut self, lr: f64) {
        self.lr_history.push(lr);
    }

    /// Add a gradient norm value.
    pub fn add_grad_norm(&mut self, norm: f64) {
        self.grad_norm_history.push(norm);
    }

    /// Add a custom metric value.
    pub fn add_custom(&mut self, name: impl Into<String>, value: f64) {
        self.custom.entry(name.into()).or_default().push(value);
    }

    /// Get final loss (last value).
    pub fn final_loss(&self) -> Option<f64> {
        self.loss_history.last().copied()
    }

    /// Get final accuracy (last value).
    pub fn final_accuracy(&self) -> Option<f64> {
        self.accuracy_history.last().copied()
    }

    /// Get best loss (minimum).
    pub fn best_loss(&self) -> Option<f64> {
        self.loss_history.iter().copied().reduce(f64::min)
    }

    /// Get best accuracy (maximum).
    pub fn best_accuracy(&self) -> Option<f64> {
        self.accuracy_history.iter().copied().reduce(f64::max)
    }

    /// Get total training steps.
    pub fn total_steps(&self) -> usize {
        self.loss_history.len()
    }

    /// Check if metrics are empty.
    pub fn is_empty(&self) -> bool {
        self.loss_history.is_empty() && self.accuracy_history.is_empty() && self.custom.is_empty()
    }
}

/// Entrenar session representation (converted from Ruchy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrenarSession {
    /// Session identifier
    pub id: String,
    /// Session name/title
    pub name: String,
    /// User who created the session
    pub user: Option<String>,
    /// Session creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Session end timestamp (None if still active)
    pub ended_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Model architecture used
    pub model_architecture: Option<String>,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, String>,
    /// Training metrics
    pub metrics: SessionMetrics,
    /// Code cells/history from notebook
    pub code_history: Vec<CodeCell>,
    /// Session tags
    pub tags: Vec<String>,
    /// Notes/annotations
    pub notes: Option<String>,
}

/// A code cell from the session history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCell {
    /// Cell execution order
    pub execution_order: u32,
    /// Source code
    pub source: String,
    /// Output (if captured)
    pub output: Option<String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
}

impl EntrenarSession {
    /// Create a new session.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            user: None,
            created_at: chrono::Utc::now(),
            ended_at: None,
            model_architecture: None,
            dataset_id: None,
            config: HashMap::new(),
            metrics: SessionMetrics::new(),
            code_history: Vec::new(),
            tags: Vec::new(),
            notes: None,
        }
    }

    /// Set user.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set model architecture.
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.model_architecture = Some(arch.into());
        self
    }

    /// Set dataset.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset_id = Some(dataset.into());
        self
    }

    /// Add configuration parameter.
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Add a code cell.
    pub fn add_code_cell(&mut self, cell: CodeCell) {
        self.code_history.push(cell);
    }

    /// Mark session as ended.
    pub fn end(&mut self) {
        self.ended_at = Some(chrono::Utc::now());
    }

    /// Calculate session duration.
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.ended_at.map(|end| end - self.created_at)
    }

    /// Check if session has training data.
    pub fn has_training_data(&self) -> bool {
        !self.metrics.is_empty()
    }

    /// Export session to JSON for external tools (CLI, notebooks).
    /// (Issue #75: Session Export API for ruchy integration)
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn export_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        let export = SessionExport {
            session_id: self.id.clone(),
            name: self.name.clone(),
            user: self.user.clone(),
            created_at: self.created_at.to_rfc3339(),
            ended_at: self.ended_at.map(|t| t.to_rfc3339()),
            duration_seconds: self.duration().map(|d| d.num_seconds()),
            model_architecture: self.model_architecture.clone(),
            dataset_id: self.dataset_id.clone(),
            config: self.config.clone(),
            metrics: MetricsExportSummary {
                total_steps: self.metrics.total_steps(),
                final_loss: self.metrics.final_loss(),
                best_loss: self.metrics.best_loss(),
                final_accuracy: self.metrics.final_accuracy(),
                best_accuracy: self.metrics.best_accuracy(),
                loss_history: self.metrics.loss_history.clone(),
                accuracy_history: self.metrics.accuracy_history.clone(),
                custom_metrics: self.metrics.custom.clone(),
            },
            code_cells_count: self.code_history.len(),
            tags: self.tags.clone(),
            notes: self.notes.clone(),
        };
        serde_json::to_value(export)
    }

    /// Export session to pretty-printed JSON string.
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn export_json_string(&self) -> Result<String, serde_json::Error> {
        let value = self.export_json()?;
        serde_json::to_string_pretty(&value)
    }
}

/// Session export structure for JSON serialization (Issue #75).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExport {
    /// Session identifier
    pub session_id: String,
    /// Session name
    pub name: String,
    /// User who created the session
    pub user: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: String,
    /// End timestamp (RFC 3339)
    pub ended_at: Option<String>,
    /// Duration in seconds
    pub duration_seconds: Option<i64>,
    /// Model architecture
    pub model_architecture: Option<String>,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, String>,
    /// Training metrics summary
    pub metrics: MetricsExportSummary,
    /// Number of code cells
    pub code_cells_count: usize,
    /// Session tags
    pub tags: Vec<String>,
    /// Notes
    pub notes: Option<String>,
}

/// Metrics export summary for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportSummary {
    /// Total training steps
    pub total_steps: usize,
    /// Final loss value
    pub final_loss: Option<f64>,
    /// Best (minimum) loss value
    pub best_loss: Option<f64>,
    /// Final accuracy value
    pub final_accuracy: Option<f64>,
    /// Best (maximum) accuracy value
    pub best_accuracy: Option<f64>,
    /// Full loss history
    pub loss_history: Vec<f64>,
    /// Full accuracy history
    pub accuracy_history: Vec<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Vec<f64>>,
}

/// Simulated Ruchy session type for conversion.
///
/// In a real implementation, this would be `ruchy::Session`.
/// Here we define a compatible structure for testing.
#[derive(Debug, Clone)]
pub struct RuchySession {
    /// Session ID
    pub session_id: String,
    /// Session title
    pub title: String,
    /// Username
    pub username: Option<String>,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Kernel info
    pub kernel: Option<String>,
    /// Cells
    pub cells: Vec<RuchyCell>,
    /// Variables (serialized)
    pub variables: HashMap<String, String>,
    /// Training runs
    pub training_runs: Vec<TrainingRun>,
}

/// A cell in a Ruchy session.
#[derive(Debug, Clone)]
pub struct RuchyCell {
    /// Cell ID
    pub id: String,
    /// Cell type (code, markdown)
    pub cell_type: String,
    /// Source content
    pub source: String,
    /// Outputs
    pub outputs: Vec<String>,
    /// Execution count
    pub execution_count: Option<u32>,
    /// Timestamp
    pub executed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// A training run within a session.
#[derive(Debug, Clone)]
pub struct TrainingRun {
    /// Run ID
    pub run_id: String,
    /// Model name
    pub model: String,
    /// Dataset
    pub dataset: Option<String>,
    /// Epochs
    pub epochs: u32,
    /// Loss values
    pub losses: Vec<f64>,
    /// Metrics
    pub metrics: HashMap<String, Vec<f64>>,
}

impl From<RuchySession> for EntrenarSession {
    fn from(ruchy: RuchySession) -> Self {
        let mut session = EntrenarSession::new(&ruchy.session_id, &ruchy.title);

        session.user = ruchy.username;
        session.created_at = ruchy.start_time;
        session.ended_at = ruchy.end_time;
        session.model_architecture = ruchy.kernel;

        // Convert cells
        for cell in ruchy.cells {
            if cell.cell_type == "code" {
                let code_cell = CodeCell {
                    execution_order: cell.execution_count.unwrap_or(0),
                    source: cell.source,
                    output: cell.outputs.first().cloned(),
                    timestamp: cell.executed_at.unwrap_or(ruchy.start_time),
                    duration_ms: None,
                };
                session.code_history.push(code_cell);
            }
        }

        // Convert training runs to metrics
        for run in ruchy.training_runs {
            for loss in run.losses {
                session.metrics.add_loss(loss);
            }
            for (name, values) in run.metrics {
                for value in values {
                    session.metrics.add_custom(&name, value);
                }
            }
            if session.dataset_id.is_none() {
                session.dataset_id = run.dataset;
            }
        }

        // Copy variables as config
        session.config = ruchy.variables;

        session
    }
}

/// Convert a Ruchy session to a research artifact.
///
/// Preserves training history and metadata in the artifact.
pub fn session_to_artifact(
    session: &EntrenarSession,
) -> Result<ResearchArtifact, RuchyBridgeError> {
    if !session.has_training_data() && session.code_history.is_empty() {
        return Err(RuchyBridgeError::NoTrainingHistory);
    }

    let mut artifact = ResearchArtifact::new(
        &session.id,
        &session.name,
        ArtifactType::Notebook,
        License::Mit,
    );

    // Add user as author if available
    if let Some(ref user) = session.user {
        let author = Author::new(user)
            .with_role(ContributorRole::Software)
            .with_role(ContributorRole::Investigation);
        artifact = artifact.with_author(author);
    }

    // Add description with metrics summary
    let description = build_session_description(session);
    artifact = artifact.with_description(description);

    // Add keywords from tags
    if session.tags.is_empty() {
        artifact = artifact.with_keywords(["training", "experiment", "entrenar"]);
    } else {
        artifact = artifact.with_keywords(session.tags.iter().map(String::as_str));
    }

    // Set version based on training steps
    let steps = session.metrics.total_steps();
    artifact = artifact.with_version(format!("1.0.0+steps{steps}"));

    Ok(artifact)
}

/// Build a description from session data.
fn build_session_description(session: &EntrenarSession) -> String {
    let mut parts = Vec::new();

    if let Some(ref arch) = session.model_architecture {
        parts.push(format!("Model: {arch}"));
    }

    if let Some(ref dataset) = session.dataset_id {
        parts.push(format!("Dataset: {dataset}"));
    }

    let steps = session.metrics.total_steps();
    if steps > 0 {
        parts.push(format!("Training steps: {steps}"));
    }

    if let Some(loss) = session.metrics.final_loss() {
        parts.push(format!("Final loss: {loss:.4}"));
    }

    if let Some(acc) = session.metrics.final_accuracy() {
        parts.push(format!("Final accuracy: {acc:.2}%"));
    }

    if let Some(duration) = session.duration() {
        let hours = duration.num_hours();
        let minutes = duration.num_minutes() % 60;
        parts.push(format!("Duration: {hours}h {minutes}m"));
    }

    if parts.is_empty() {
        format!("Training session from Ruchy ({})", session.id)
    } else {
        parts.join(". ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_metrics_creation() {
        let mut metrics = SessionMetrics::new();
        assert!(metrics.is_empty());

        metrics.add_loss(0.5);
        metrics.add_loss(0.3);
        metrics.add_loss(0.2);

        assert!(!metrics.is_empty());
        assert_eq!(metrics.total_steps(), 3);
        assert_eq!(metrics.final_loss(), Some(0.2));
        assert_eq!(metrics.best_loss(), Some(0.2));
    }

    #[test]
    fn test_session_metrics_accuracy() {
        let mut metrics = SessionMetrics::new();

        metrics.add_accuracy(0.7);
        metrics.add_accuracy(0.8);
        metrics.add_accuracy(0.85);

        assert_eq!(metrics.final_accuracy(), Some(0.85));
        assert_eq!(metrics.best_accuracy(), Some(0.85));
    }

    #[test]
    fn test_session_metrics_custom() {
        let mut metrics = SessionMetrics::new();

        metrics.add_custom("f1_score", 0.75);
        metrics.add_custom("f1_score", 0.82);

        assert_eq!(metrics.custom.get("f1_score"), Some(&vec![0.75, 0.82]));
    }

    #[test]
    fn test_entrenar_session_creation() {
        let session = EntrenarSession::new("sess-001", "My Training Session")
            .with_user("alice")
            .with_architecture("llama-7b")
            .with_dataset("custom-dataset")
            .with_config("batch_size", "32")
            .with_config("learning_rate", "1e-4")
            .with_tag("fine-tuning")
            .with_notes("Initial experiment");

        assert_eq!(session.id, "sess-001");
        assert_eq!(session.name, "My Training Session");
        assert_eq!(session.user, Some("alice".to_string()));
        assert_eq!(session.model_architecture, Some("llama-7b".to_string()));
        assert_eq!(session.dataset_id, Some("custom-dataset".to_string()));
        assert_eq!(session.config.get("batch_size"), Some(&"32".to_string()));
        assert_eq!(session.tags, vec!["fine-tuning"]);
        assert!(!session.has_training_data());
    }

    #[test]
    fn test_entrenar_session_with_metrics() {
        let mut session = EntrenarSession::new("sess-001", "Training");
        session.metrics.add_loss(0.5);
        session.metrics.add_loss(0.3);

        assert!(session.has_training_data());
        assert_eq!(session.metrics.total_steps(), 2);
    }

    #[test]
    fn test_entrenar_session_duration() {
        let mut session = EntrenarSession::new("sess-001", "Training");
        let start = session.created_at;

        // Simulate session end
        session.ended_at = Some(start + chrono::Duration::hours(2));

        let duration = session.duration().unwrap();
        assert_eq!(duration.num_hours(), 2);
    }

    #[test]
    fn test_code_cell_creation() {
        let cell = CodeCell {
            execution_order: 1,
            source: "model.train()".to_string(),
            output: Some("Training started...".to_string()),
            timestamp: chrono::Utc::now(),
            duration_ms: Some(1500),
        };

        assert_eq!(cell.execution_order, 1);
        assert_eq!(cell.source, "model.train()");
        assert!(cell.output.is_some());
    }

    #[test]
    fn test_ruchy_session_conversion() {
        let ruchy = RuchySession {
            session_id: "ruchy-123".to_string(),
            title: "LLaMA Fine-tuning".to_string(),
            username: Some("bob".to_string()),
            start_time: chrono::Utc::now(),
            end_time: None,
            kernel: Some("python3".to_string()),
            cells: vec![RuchyCell {
                id: "cell-1".to_string(),
                cell_type: "code".to_string(),
                source: "import entrenar".to_string(),
                outputs: vec!["OK".to_string()],
                execution_count: Some(1),
                executed_at: Some(chrono::Utc::now()),
            }],
            variables: HashMap::from([("lr".to_string(), "0.001".to_string())]),
            training_runs: vec![TrainingRun {
                run_id: "run-1".to_string(),
                model: "llama".to_string(),
                dataset: Some("alpaca".to_string()),
                epochs: 3,
                losses: vec![0.5, 0.3, 0.2],
                metrics: HashMap::new(),
            }],
        };

        let session: EntrenarSession = ruchy.into();

        assert_eq!(session.id, "ruchy-123");
        assert_eq!(session.name, "LLaMA Fine-tuning");
        assert_eq!(session.user, Some("bob".to_string()));
        assert_eq!(session.model_architecture, Some("python3".to_string()));
        assert_eq!(session.dataset_id, Some("alpaca".to_string()));
        assert_eq!(session.code_history.len(), 1);
        assert_eq!(session.metrics.total_steps(), 3);
        assert_eq!(session.config.get("lr"), Some(&"0.001".to_string()));
    }

    #[test]
    fn test_session_to_artifact_success() {
        let mut session = EntrenarSession::new("sess-001", "My Experiment")
            .with_user("alice")
            .with_architecture("llama-7b")
            .with_dataset("custom-data")
            .with_tag("lora")
            .with_tag("fine-tuning");

        session.metrics.add_loss(0.5);
        session.metrics.add_loss(0.3);
        session.metrics.add_loss(0.2);

        let artifact = session_to_artifact(&session).unwrap();

        assert_eq!(artifact.id, "sess-001");
        assert_eq!(artifact.title, "My Experiment");
        assert_eq!(artifact.artifact_type, ArtifactType::Notebook);
        assert_eq!(artifact.authors.len(), 1);
        assert_eq!(artifact.authors[0].name, "alice");
        assert!(artifact.keywords.contains(&"lora".to_string()));
        assert!(artifact.version.contains("steps3"));
    }

    #[test]
    fn test_session_to_artifact_no_training() {
        let session = EntrenarSession::new("sess-001", "Empty Session");
        let result = session_to_artifact(&session);
        assert!(matches!(result, Err(RuchyBridgeError::NoTrainingHistory)));
    }

    #[test]
    fn test_session_to_artifact_with_code_only() {
        let mut session = EntrenarSession::new("sess-001", "Code Session");
        session.add_code_cell(CodeCell {
            execution_order: 1,
            source: "print('hello')".to_string(),
            output: None,
            timestamp: chrono::Utc::now(),
            duration_ms: None,
        });

        let artifact = session_to_artifact(&session).unwrap();
        assert_eq!(artifact.id, "sess-001");
    }

    #[test]
    fn test_build_session_description() {
        let mut session = EntrenarSession::new("sess-001", "Test")
            .with_architecture("gpt2")
            .with_dataset("wiki");

        session.metrics.add_loss(0.5);
        session.metrics.add_loss(0.2);
        session.metrics.add_accuracy(85.5);

        let desc = build_session_description(&session);

        assert!(desc.contains("gpt2"));
        assert!(desc.contains("wiki"));
        assert!(desc.contains("2")); // steps
        assert!(desc.contains("0.2")); // loss
    }

    // Issue #75: Session Export API tests
    #[test]
    fn test_export_json_basic() {
        let session = EntrenarSession::new("export-001", "Export Test")
            .with_user("tester")
            .with_tag("export-test");

        let json = session.export_json().unwrap();

        assert_eq!(json["session_id"], "export-001");
        assert_eq!(json["name"], "Export Test");
        assert_eq!(json["user"], "tester");
        assert_eq!(json["metrics"]["total_steps"], 0);
        assert_eq!(json["code_cells_count"], 0);
        assert!(json["tags"]
            .as_array()
            .unwrap()
            .contains(&"export-test".into()));
    }

    #[test]
    fn test_export_json_with_metrics() {
        let mut session = EntrenarSession::new("export-002", "Metrics Export")
            .with_architecture("llama-7b")
            .with_dataset("alpaca");

        session.metrics.add_loss(0.5);
        session.metrics.add_loss(0.3);
        session.metrics.add_loss(0.2);
        session.metrics.add_accuracy(0.7);
        session.metrics.add_accuracy(0.85);
        session.metrics.add_custom("f1", 0.78);

        let json = session.export_json().unwrap();

        assert_eq!(json["metrics"]["total_steps"], 3);
        assert_eq!(json["metrics"]["final_loss"], 0.2);
        assert_eq!(json["metrics"]["best_loss"], 0.2);
        assert_eq!(json["metrics"]["final_accuracy"], 0.85);
        assert_eq!(json["metrics"]["best_accuracy"], 0.85);
        assert_eq!(json["metrics"]["loss_history"].as_array().unwrap().len(), 3);
        assert_eq!(
            json["metrics"]["accuracy_history"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(json["metrics"]["custom_metrics"]["f1"].as_array().is_some());
    }

    #[test]
    fn test_export_json_string() {
        let session =
            EntrenarSession::new("export-003", "String Export").with_config("batch_size", "32");

        let json_str = session.export_json_string().unwrap();

        assert!(json_str.contains("\"session_id\": \"export-003\""));
        assert!(json_str.contains("\"batch_size\": \"32\""));
        // Pretty print should have newlines
        assert!(json_str.contains('\n'));
    }

    #[test]
    fn test_export_json_with_duration() {
        let mut session = EntrenarSession::new("export-004", "Duration Export");
        let start = session.created_at;
        session.ended_at = Some(start + chrono::Duration::hours(1) + chrono::Duration::minutes(30));

        let json = session.export_json().unwrap();

        assert_eq!(json["duration_seconds"], 5400); // 1.5 hours = 5400 seconds
        assert!(json["ended_at"].as_str().is_some());
    }

    #[test]
    fn test_export_json_roundtrip() {
        let mut session = EntrenarSession::new("export-005", "Roundtrip Test")
            .with_user("alice")
            .with_architecture("transformer")
            .with_dataset("custom-data")
            .with_config("epochs", "10")
            .with_tag("test");

        session.metrics.add_loss(0.4);
        session.metrics.add_accuracy(0.9);

        let json = session.export_json().unwrap();
        let export: SessionExport = serde_json::from_value(json).unwrap();

        assert_eq!(export.session_id, "export-005");
        assert_eq!(export.name, "Roundtrip Test");
        assert_eq!(export.user, Some("alice".to_string()));
        assert_eq!(export.model_architecture, Some("transformer".to_string()));
        assert_eq!(export.dataset_id, Some("custom-data".to_string()));
        assert_eq!(export.metrics.total_steps, 1);
        assert_eq!(export.metrics.final_loss, Some(0.4));
        assert_eq!(export.metrics.final_accuracy, Some(0.9));
    }
}

//! Lineage event types and structures

use super::timestamp::LamportTimestamp;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Type of lineage event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LineageEventType {
    /// Training run started
    RunStarted,
    /// Metric was logged
    MetricLogged,
    /// Artifact was saved
    ArtifactSaved,
    /// Training run completed
    RunCompleted,
    /// Model was promoted to production
    ModelPromoted,
    /// Model was rolled back
    ModelRolledBack,
}

impl LineageEventType {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::RunStarted => "Run started",
            Self::MetricLogged => "Metric logged",
            Self::ArtifactSaved => "Artifact saved",
            Self::RunCompleted => "Run completed",
            Self::ModelPromoted => "Model promoted",
            Self::ModelRolledBack => "Model rolled back",
        }
    }
}

impl std::fmt::Display for LineageEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A single event in the causal lineage
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEvent {
    /// Lamport timestamp for causal ordering
    pub timestamp: LamportTimestamp,
    /// Type of event
    pub event_type: LineageEventType,
    /// Associated run ID
    pub run_id: String,
    /// Optional additional context
    pub context: Option<String>,
}

impl LineageEvent {
    /// Create a new lineage event
    pub fn new(timestamp: LamportTimestamp, event_type: LineageEventType, run_id: &str) -> Self {
        Self {
            timestamp,
            event_type,
            run_id: run_id.to_string(),
            context: None,
        }
    }

    /// Add context to the event
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl PartialOrd for LineageEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LineageEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

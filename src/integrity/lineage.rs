//! Lamport Timestamps and Causal Lineage (ENT-014)
//!
//! Provides logical clocks for tracking causal ordering of events
//! across distributed training runs without requiring synchronized wall clocks.
//!
//! # Example
//!
//! ```
//! use entrenar::integrity::{LamportTimestamp, LineageEventType, CausalLineage};
//!
//! let mut ts1 = LamportTimestamp::new("node-1");
//! let mut ts2 = LamportTimestamp::new("node-2");
//!
//! // Local events increment counter
//! ts1.increment();
//! ts1.increment();
//!
//! // Receiving a message merges timestamps
//! ts2.merge(&ts1);
//!
//! // Check causal ordering
//! assert!(ts1.happens_before(&ts2));
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Lamport logical timestamp for causal ordering
///
/// Implements Lamport's logical clock algorithm for establishing
/// happens-before relationships without synchronized wall clocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LamportTimestamp {
    /// Logical counter value
    pub counter: u64,
    /// Node identifier for tie-breaking
    pub node_id: String,
}

impl LamportTimestamp {
    /// Create a new timestamp for a node
    pub fn new(node_id: &str) -> Self {
        Self {
            counter: 0,
            node_id: node_id.to_string(),
        }
    }

    /// Create a timestamp with specific counter value
    pub fn with_counter(node_id: &str, counter: u64) -> Self {
        Self {
            counter,
            node_id: node_id.to_string(),
        }
    }

    /// Increment the timestamp for a local event
    ///
    /// Returns a copy of the new timestamp value.
    pub fn increment(&mut self) -> Self {
        self.counter += 1;
        self.clone()
    }

    /// Merge with another timestamp (on message receive)
    ///
    /// Sets counter to max(self.counter, other.counter) + 1
    /// Returns a copy of the new timestamp value.
    pub fn merge(&mut self, other: &Self) -> Self {
        self.counter = self.counter.max(other.counter) + 1;
        self.clone()
    }

    /// Check if this timestamp happens-before another
    ///
    /// Returns true if:
    /// - self.counter < other.counter, OR
    /// - self.counter == other.counter AND self.node_id < other.node_id
    ///
    /// Note: If neither happens_before the other, events are concurrent.
    pub fn happens_before(&self, other: &Self) -> bool {
        match self.counter.cmp(&other.counter) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.node_id < other.node_id,
        }
    }

    /// Check if events are concurrent (neither happens-before the other)
    pub fn is_concurrent_with(&self, other: &Self) -> bool {
        self.counter == other.counter && self.node_id != other.node_id
    }
}

impl PartialOrd for LamportTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LamportTimestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.counter.cmp(&other.counter) {
            Ordering::Equal => self.node_id.cmp(&other.node_id),
            other => other,
        }
    }
}

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

/// Causal lineage tracking for a set of events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalLineage {
    /// Events in causal order
    pub events: Vec<LineageEvent>,
}

impl CausalLineage {
    /// Create a new empty lineage
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the lineage
    pub fn add_event(&mut self, event: LineageEvent) {
        self.events.push(event);
        self.events.sort();
    }

    /// Get events in causal order
    pub fn events_in_order(&self) -> &[LineageEvent] {
        &self.events
    }

    /// Get events for a specific run
    pub fn events_for_run(&self, run_id: &str) -> Vec<&LineageEvent> {
        self.events.iter().filter(|e| e.run_id == run_id).collect()
    }

    /// Get events of a specific type
    pub fn events_of_type(&self, event_type: LineageEventType) -> Vec<&LineageEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Find the latest event for a run
    pub fn latest_event_for_run(&self, run_id: &str) -> Option<&LineageEvent> {
        self.events.iter().rev().find(|e| e.run_id == run_id)
    }

    /// Check if run A causally precedes run B
    pub fn run_precedes(&self, run_a: &str, run_b: &str) -> bool {
        let a_events = self.events_for_run(run_a);
        let b_events = self.events_for_run(run_b);

        if a_events.is_empty() || b_events.is_empty() {
            return false;
        }

        // A precedes B if all of A's events happen before B's first event
        let b_first = b_events
            .first()
            .expect("b_events is non-empty (checked above)");
        a_events
            .iter()
            .all(|a| a.timestamp.happens_before(&b_first.timestamp))
    }

    /// Get the total number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if lineage is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // LamportTimestamp Tests
    // ==========================================================================

    #[test]
    fn test_lamport_timestamp_new() {
        let ts = LamportTimestamp::new("node-1");
        assert_eq!(ts.counter, 0);
        assert_eq!(ts.node_id, "node-1");
    }

    #[test]
    fn test_lamport_timestamp_with_counter() {
        let ts = LamportTimestamp::with_counter("node-1", 42);
        assert_eq!(ts.counter, 42);
    }

    #[test]
    fn test_lamport_timestamp_increment() {
        let mut ts = LamportTimestamp::new("node-1");
        assert_eq!(ts.counter, 0);

        ts.increment();
        assert_eq!(ts.counter, 1);

        ts.increment();
        assert_eq!(ts.counter, 2);

        let returned = ts.increment();
        assert_eq!(ts.counter, 3);
        assert_eq!(returned.counter, 3);
    }

    #[test]
    fn test_lamport_timestamp_merge() {
        let mut ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-2", 10);

        // Merge should set counter to max(5, 10) + 1 = 11
        ts1.merge(&ts2);
        assert_eq!(ts1.counter, 11);
    }

    #[test]
    fn test_lamport_timestamp_merge_smaller() {
        let mut ts1 = LamportTimestamp::with_counter("node-1", 10);
        let ts2 = LamportTimestamp::with_counter("node-2", 5);

        // Merge should set counter to max(10, 5) + 1 = 11
        ts1.merge(&ts2);
        assert_eq!(ts1.counter, 11);
    }

    #[test]
    fn test_lamport_timestamp_happens_before_counter() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-2", 10);

        assert!(ts1.happens_before(&ts2));
        assert!(!ts2.happens_before(&ts1));
    }

    #[test]
    fn test_lamport_timestamp_happens_before_same_counter() {
        let ts1 = LamportTimestamp::with_counter("node-a", 5);
        let ts2 = LamportTimestamp::with_counter("node-b", 5);

        // Same counter, use node_id for ordering
        assert!(ts1.happens_before(&ts2)); // "node-a" < "node-b"
        assert!(!ts2.happens_before(&ts1));
    }

    #[test]
    fn test_lamport_timestamp_happens_before_same_node() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-1", 5);

        // Same counter AND same node - neither happens before
        assert!(!ts1.happens_before(&ts2));
        assert!(!ts2.happens_before(&ts1));
    }

    #[test]
    fn test_lamport_timestamp_is_concurrent() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-2", 5);

        assert!(ts1.is_concurrent_with(&ts2));
        assert!(ts2.is_concurrent_with(&ts1));
    }

    #[test]
    fn test_lamport_timestamp_not_concurrent_different_counter() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-2", 6);

        assert!(!ts1.is_concurrent_with(&ts2));
    }

    #[test]
    fn test_lamport_timestamp_ordering() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-2", 10);
        let ts3 = LamportTimestamp::with_counter("node-1", 10);

        assert!(ts1 < ts2);
        assert!(ts3 < ts2); // Same counter, but "node-1" < "node-2"
    }

    #[test]
    fn test_lamport_timestamp_scenario() {
        // Simulate distributed scenario
        let mut node1 = LamportTimestamp::new("node-1");
        let mut node2 = LamportTimestamp::new("node-2");

        // Node 1 does local work
        node1.increment(); // counter = 1
        node1.increment(); // counter = 2

        // Node 2 does local work
        node2.increment(); // counter = 1

        // Node 2 receives message from Node 1
        node2.merge(&node1); // counter = max(1, 2) + 1 = 3

        assert!(node1.happens_before(&node2));
        assert_eq!(node2.counter, 3);
    }

    // ==========================================================================
    // LineageEventType Tests
    // ==========================================================================

    #[test]
    fn test_lineage_event_type_description() {
        assert_eq!(LineageEventType::RunStarted.description(), "Run started");
        assert_eq!(
            LineageEventType::ModelPromoted.description(),
            "Model promoted"
        );
    }

    #[test]
    fn test_lineage_event_type_display() {
        assert_eq!(
            format!("{}", LineageEventType::RunCompleted),
            "Run completed"
        );
    }

    // ==========================================================================
    // LineageEvent Tests
    // ==========================================================================

    #[test]
    fn test_lineage_event_new() {
        let ts = LamportTimestamp::new("node-1");
        let event = LineageEvent::new(ts, LineageEventType::RunStarted, "run-001");

        assert_eq!(event.run_id, "run-001");
        assert_eq!(event.event_type, LineageEventType::RunStarted);
        assert!(event.context.is_none());
    }

    #[test]
    fn test_lineage_event_with_context() {
        let ts = LamportTimestamp::new("node-1");
        let event = LineageEvent::new(ts, LineageEventType::MetricLogged, "run-001")
            .with_context("loss=0.5");

        assert_eq!(event.context, Some("loss=0.5".to_string()));
    }

    #[test]
    fn test_lineage_event_ordering() {
        let ts1 = LamportTimestamp::with_counter("node-1", 5);
        let ts2 = LamportTimestamp::with_counter("node-1", 10);

        let event1 = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001");
        let event2 = LineageEvent::new(ts2, LineageEventType::RunCompleted, "run-001");

        assert!(event1 < event2);
    }

    // ==========================================================================
    // CausalLineage Tests
    // ==========================================================================

    #[test]
    fn test_causal_lineage_new() {
        let lineage = CausalLineage::new();
        assert!(lineage.is_empty());
        assert_eq!(lineage.len(), 0);
    }

    #[test]
    fn test_causal_lineage_add_event() {
        let mut lineage = CausalLineage::new();

        let ts1 = LamportTimestamp::with_counter("node-1", 1);
        let event1 = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001");

        lineage.add_event(event1);
        assert_eq!(lineage.len(), 1);
    }

    #[test]
    fn test_causal_lineage_events_in_order() {
        let mut lineage = CausalLineage::new();

        // Add out of order
        let ts2 = LamportTimestamp::with_counter("node-1", 10);
        let ts1 = LamportTimestamp::with_counter("node-1", 5);

        lineage.add_event(LineageEvent::new(
            ts2,
            LineageEventType::RunCompleted,
            "run-001",
        ));
        lineage.add_event(LineageEvent::new(
            ts1,
            LineageEventType::RunStarted,
            "run-001",
        ));

        let events = lineage.events_in_order();
        assert_eq!(events[0].timestamp.counter, 5);
        assert_eq!(events[1].timestamp.counter, 10);
    }

    #[test]
    fn test_causal_lineage_events_for_run() {
        let mut lineage = CausalLineage::new();

        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 1),
            LineageEventType::RunStarted,
            "run-001",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 2),
            LineageEventType::RunStarted,
            "run-002",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 3),
            LineageEventType::RunCompleted,
            "run-001",
        ));

        let run1_events = lineage.events_for_run("run-001");
        assert_eq!(run1_events.len(), 2);
    }

    #[test]
    fn test_causal_lineage_events_of_type() {
        let mut lineage = CausalLineage::new();

        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 1),
            LineageEventType::RunStarted,
            "run-001",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 2),
            LineageEventType::MetricLogged,
            "run-001",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 3),
            LineageEventType::MetricLogged,
            "run-001",
        ));

        let metric_events = lineage.events_of_type(LineageEventType::MetricLogged);
        assert_eq!(metric_events.len(), 2);
    }

    #[test]
    fn test_causal_lineage_latest_event_for_run() {
        let mut lineage = CausalLineage::new();

        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 1),
            LineageEventType::RunStarted,
            "run-001",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 5),
            LineageEventType::RunCompleted,
            "run-001",
        ));

        let latest = lineage.latest_event_for_run("run-001").unwrap();
        assert_eq!(latest.event_type, LineageEventType::RunCompleted);
        assert_eq!(latest.timestamp.counter, 5);
    }

    #[test]
    fn test_causal_lineage_run_precedes() {
        let mut lineage = CausalLineage::new();

        // Run A: events at counter 1, 2
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 1),
            LineageEventType::RunStarted,
            "run-A",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 2),
            LineageEventType::RunCompleted,
            "run-A",
        ));

        // Run B: events at counter 5, 6
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 5),
            LineageEventType::RunStarted,
            "run-B",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 6),
            LineageEventType::RunCompleted,
            "run-B",
        ));

        assert!(lineage.run_precedes("run-A", "run-B"));
        assert!(!lineage.run_precedes("run-B", "run-A"));
    }

    #[test]
    fn test_causal_lineage_run_precedes_overlapping() {
        let mut lineage = CausalLineage::new();

        // Run A: events at counter 1, 5
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 1),
            LineageEventType::RunStarted,
            "run-A",
        ));
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 5),
            LineageEventType::RunCompleted,
            "run-A",
        ));

        // Run B: events at counter 3, 6 (overlaps with A)
        lineage.add_event(LineageEvent::new(
            LamportTimestamp::with_counter("node-1", 3),
            LineageEventType::RunStarted,
            "run-B",
        ));

        // A does not precede B because A's event at 5 > B's first event at 3
        assert!(!lineage.run_precedes("run-A", "run-B"));
    }

    #[test]
    fn test_lamport_timestamp_serialization() {
        let ts = LamportTimestamp::with_counter("node-1", 42);
        let json = serde_json::to_string(&ts).unwrap();
        let parsed: LamportTimestamp = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.counter, 42);
        assert_eq!(parsed.node_id, "node-1");
    }

    #[test]
    fn test_lineage_event_serialization() {
        let ts = LamportTimestamp::with_counter("node-1", 10);
        let event = LineageEvent::new(ts, LineageEventType::ModelPromoted, "run-001")
            .with_context("version=1.0");

        let json = serde_json::to_string(&event).unwrap();
        let parsed: LineageEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.event_type, LineageEventType::ModelPromoted);
        assert_eq!(parsed.context, Some("version=1.0".to_string()));
    }
}

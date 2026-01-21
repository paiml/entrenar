//! Provenance Graph for Incident Reconstruction (ENT-111)
//!
//! Captures causal relationships between system entities for forensic analysis.

use super::path::DecisionPath;
use super::trace::DecisionTrace;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a node in the provenance graph
pub type NodeId = u64;

/// Provenance graph node types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProvenanceNode {
    /// Raw sensor/input data
    Input {
        /// Source identifier
        source: String,
        /// Timestamp in nanoseconds
        timestamp_ns: u64,
        /// Hash of the input data
        hash: u64,
    },

    /// Preprocessing transformation
    Transform {
        /// Operation name
        operation: String,
        /// References to input nodes
        input_refs: Vec<NodeId>,
    },

    /// Model inference
    Inference {
        /// Model identifier
        model_id: String,
        /// Model version
        model_version: String,
        /// Confidence of the inference
        confidence: f32,
        /// Output value
        output: f32,
    },

    /// Post-processing or sensor fusion
    Fusion {
        /// Fusion strategy
        strategy: String,
        /// References to input nodes
        input_refs: Vec<NodeId>,
    },

    /// Final action/output
    Action {
        /// Action type
        action_type: String,
        /// Confidence
        confidence: f32,
        /// Alternatives considered
        alternatives: Vec<(String, f32)>,
    },
}

impl ProvenanceNode {
    /// Get the timestamp if available
    pub fn timestamp_ns(&self) -> Option<u64> {
        match self {
            ProvenanceNode::Input { timestamp_ns, .. } => Some(*timestamp_ns),
            _ => None,
        }
    }

    /// Get node type as string
    pub fn type_name(&self) -> &'static str {
        match self {
            ProvenanceNode::Input { .. } => "Input",
            ProvenanceNode::Transform { .. } => "Transform",
            ProvenanceNode::Inference { .. } => "Inference",
            ProvenanceNode::Fusion { .. } => "Fusion",
            ProvenanceNode::Action { .. } => "Action",
        }
    }
}

/// Causal relation between nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRelation {
    /// Data flowed from source to sink
    DataFlow,
    /// Inference triggered by input
    Triggered,
    /// Decision influenced by
    Influenced,
    /// Action caused by decision
    Caused,
}

/// Edge in provenance graph (directed, acyclic)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    /// Source node
    pub from: NodeId,
    /// Destination node
    pub to: NodeId,
    /// Causal relation
    pub relation: CausalRelation,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

/// Provenance graph for causal analysis
///
/// # Features
/// - DAG structure for causal relationships
/// - Backward traversal for root cause analysis
/// - Anomaly detection in decision chains
pub struct ProvenanceGraph {
    /// Nodes indexed by ID
    nodes: HashMap<NodeId, ProvenanceNode>,
    /// Edges
    edges: Vec<ProvenanceEdge>,
    /// Adjacency list (forward): node -> outgoing edges
    forward: HashMap<NodeId, Vec<usize>>,
    /// Adjacency list (backward): node -> incoming edges
    backward: HashMap<NodeId, Vec<usize>>,
    /// Next node ID
    next_id: NodeId,
}

impl ProvenanceGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            forward: HashMap::new(),
            backward: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: ProvenanceNode) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: ProvenanceEdge) {
        let edge_idx = self.edges.len();
        self.forward.entry(edge.from).or_default().push(edge_idx);
        self.backward.entry(edge.to).or_default().push(edge_idx);
        self.edges.push(edge);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&ProvenanceNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, ProvenanceNode> {
        &self.nodes
    }

    /// Get all edges
    pub fn edges(&self) -> &[ProvenanceEdge] {
        &self.edges
    }

    /// Get incoming edges for a node
    pub fn incoming_edges(&self, id: NodeId) -> Vec<&ProvenanceEdge> {
        self.backward
            .get(&id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, id: NodeId) -> Vec<&ProvenanceEdge> {
        self.forward
            .get(&id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get predecessor nodes
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.incoming_edges(id)
            .into_iter()
            .map(|e| e.from)
            .collect()
    }

    /// Get successor nodes
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        self.outgoing_edges(id).into_iter().map(|e| e.to).collect()
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add an inference node from a decision trace
    pub fn add_inference<P: DecisionPath>(
        &mut self,
        trace: &DecisionTrace<P>,
        model_id: &str,
        model_version: &str,
    ) -> NodeId {
        self.add_node(ProvenanceNode::Inference {
            model_id: model_id.to_string(),
            model_version: model_version.to_string(),
            confidence: trace.confidence(),
            output: trace.output,
        })
    }
}

impl Default for ProvenanceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Reconstructed attack/incident path
#[derive(Debug, Clone)]
pub struct AttackPath {
    /// Nodes in causal order (root cause → incident)
    pub nodes: Vec<(NodeId, ProvenanceNode)>,
    /// Edges connecting the path
    pub edges: Vec<ProvenanceEdge>,
    /// Time span of the incident in nanoseconds
    pub duration_ns: u64,
    /// Identified anomaly indices (in nodes vector)
    pub anomaly_indices: Vec<usize>,
}

impl AttackPath {
    /// Number of nodes in the path
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if path contains anomalies
    pub fn has_anomalies(&self) -> bool {
        !self.anomaly_indices.is_empty()
    }
}

/// Anomaly detected in provenance path
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Node ID where anomaly was detected
    pub node_id: NodeId,
    /// Description of the anomaly
    pub description: String,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
}

/// Incident reconstructor using provenance graph
pub struct IncidentReconstructor<'a> {
    graph: &'a ProvenanceGraph,
}

impl<'a> IncidentReconstructor<'a> {
    /// Create a new reconstructor
    pub fn new(graph: &'a ProvenanceGraph) -> Self {
        Self { graph }
    }

    /// Trace backwards from incident node to root causes
    pub fn reconstruct_path(&self, incident_node: NodeId, max_depth: usize) -> AttackPath {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back((incident_node, 0usize));
        visited.insert(incident_node);

        while let Some((node_id, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            if let Some(node) = self.graph.get_node(node_id) {
                nodes.push((node_id, node.clone()));
            }

            for edge in self.graph.incoming_edges(node_id) {
                edges.push(edge.clone());

                if !visited.contains(&edge.from) {
                    visited.insert(edge.from);
                    queue.push_back((edge.from, depth + 1));
                }
            }
        }

        // Reverse to get causal order (root → incident)
        nodes.reverse();

        // Calculate duration
        let duration_ns = self.calculate_duration(&nodes);

        AttackPath {
            nodes,
            edges,
            duration_ns,
            anomaly_indices: Vec::new(),
        }
    }

    /// Calculate duration from timestamps
    fn calculate_duration(&self, nodes: &[(NodeId, ProvenanceNode)]) -> u64 {
        let timestamps: Vec<u64> = nodes.iter().filter_map(|(_, n)| n.timestamp_ns()).collect();

        if timestamps.len() < 2 {
            return 0;
        }

        let min = *timestamps.iter().min().unwrap_or(&0);
        let max = *timestamps.iter().max().unwrap_or(&0);
        max - min
    }

    /// Identify anomalies in a path
    pub fn identify_anomalies(&self, path: &AttackPath, confidence_threshold: f32) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        for (idx, (node_id, node)) in path.nodes.iter().enumerate() {
            // Check for low confidence inferences
            if let ProvenanceNode::Inference { confidence, .. } = node {
                if *confidence < confidence_threshold {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!(
                            "Low confidence inference: {:.1}% (threshold: {:.1}%)",
                            confidence * 100.0,
                            confidence_threshold * 100.0
                        ),
                        severity: 1.0 - *confidence,
                    });
                }
            }

            // Check for suspicious fusion with many inputs
            if let ProvenanceNode::Fusion { input_refs, .. } = node {
                if input_refs.len() > 10 {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!(
                            "Unusually many fusion inputs: {} (expected <10)",
                            input_refs.len()
                        ),
                        severity: 0.3,
                    });
                }
            }

            // Flag nodes with no predecessors (except inputs)
            if !matches!(node, ProvenanceNode::Input { .. }) {
                let preds = self.graph.predecessors(*node_id);
                if preds.is_empty() {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!("{} node has no predecessors", node.type_name()),
                        severity: 0.5,
                    });
                }
            }

            let _ = idx; // Used in future enhancements
        }

        anomalies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_graph_new() {
        let graph = ProvenanceGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = ProvenanceGraph::new();

        let id = graph.add_node(ProvenanceNode::Input {
            source: "camera".to_string(),
            timestamp_ns: 1000,
            hash: 0xdeadbeef,
        });

        assert_eq!(id, 0);
        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_node(0).is_some());
    }

    #[test]
    fn test_add_edge() {
        let mut graph = ProvenanceGraph::new();

        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "camera".to_string(),
            timestamp_ns: 1000,
            hash: 0xdeadbeef,
        });

        let id2 = graph.add_node(ProvenanceNode::Inference {
            model_id: "detector".to_string(),
            model_version: "1.0".to_string(),
            confidence: 0.9,
            output: 1.0,
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::Triggered,
            timestamp_ns: 1100,
        });

        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.predecessors(id2), vec![id1]);
        assert_eq!(graph.successors(id1), vec![id2]);
    }

    #[test]
    fn test_incoming_outgoing_edges() {
        let mut graph = ProvenanceGraph::new();

        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Input {
            source: "b".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        let id3 = graph.add_node(ProvenanceNode::Fusion {
            strategy: "merge".to_string(),
            input_refs: vec![id1, id2],
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id3,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 100,
        });
        graph.add_edge(ProvenanceEdge {
            from: id2,
            to: id3,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 100,
        });

        assert_eq!(graph.incoming_edges(id3).len(), 2);
        assert_eq!(graph.outgoing_edges(id1).len(), 1);
    }

    #[test]
    fn test_node_type_name() {
        let node = ProvenanceNode::Input {
            source: "test".to_string(),
            timestamp_ns: 0,
            hash: 0,
        };
        assert_eq!(node.type_name(), "Input");

        let node = ProvenanceNode::Inference {
            model_id: "m".to_string(),
            model_version: "v".to_string(),
            confidence: 0.9,
            output: 1.0,
        };
        assert_eq!(node.type_name(), "Inference");
    }

    #[test]
    fn test_reconstruct_path() {
        let mut graph = ProvenanceGraph::new();

        // Build a chain: Input -> Transform -> Inference -> Action
        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "sensor".to_string(),
            timestamp_ns: 1000,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Transform {
            operation: "normalize".to_string(),
            input_refs: vec![id1],
        });
        let id3 = graph.add_node(ProvenanceNode::Inference {
            model_id: "detector".to_string(),
            model_version: "1.0".to_string(),
            confidence: 0.9,
            output: 1.0,
        });
        let id4 = graph.add_node(ProvenanceNode::Action {
            action_type: "brake".to_string(),
            confidence: 0.85,
            alternatives: vec![("accelerate".to_string(), 0.1)],
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 1100,
        });
        graph.add_edge(ProvenanceEdge {
            from: id2,
            to: id3,
            relation: CausalRelation::Triggered,
            timestamp_ns: 1200,
        });
        graph.add_edge(ProvenanceEdge {
            from: id3,
            to: id4,
            relation: CausalRelation::Caused,
            timestamp_ns: 1300,
        });

        let reconstructor = IncidentReconstructor::new(&graph);
        let path = reconstructor.reconstruct_path(id4, 10);

        assert_eq!(path.len(), 4);
        assert_eq!(path.edges.len(), 3);
    }

    #[test]
    fn test_identify_anomalies() {
        let mut graph = ProvenanceGraph::new();

        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "sensor".to_string(),
            timestamp_ns: 1000,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Inference {
            model_id: "detector".to_string(),
            model_version: "1.0".to_string(),
            confidence: 0.3, // Low confidence
            output: 1.0,
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::Triggered,
            timestamp_ns: 1100,
        });

        let reconstructor = IncidentReconstructor::new(&graph);
        let path = reconstructor.reconstruct_path(id2, 10);
        let anomalies = reconstructor.identify_anomalies(&path, 0.7);

        assert!(!anomalies.is_empty());
        assert!(anomalies[0].description.contains("Low confidence"));
    }

    #[test]
    fn test_attack_path() {
        let path = AttackPath {
            nodes: vec![],
            edges: vec![],
            duration_ns: 0,
            anomaly_indices: vec![],
        };

        assert!(path.is_empty());
        assert!(!path.has_anomalies());

        let path_with_anomalies = AttackPath {
            nodes: vec![],
            edges: vec![],
            duration_ns: 0,
            anomaly_indices: vec![0, 1],
        };

        assert!(path_with_anomalies.has_anomalies());
    }

    #[test]
    fn test_causal_relation_equality() {
        assert_eq!(CausalRelation::DataFlow, CausalRelation::DataFlow);
        assert_ne!(CausalRelation::DataFlow, CausalRelation::Triggered);
    }

    // Additional coverage tests

    #[test]
    fn test_provenance_node_timestamp_non_input() {
        let node = ProvenanceNode::Transform {
            operation: "op".to_string(),
            input_refs: vec![],
        };
        assert!(node.timestamp_ns().is_none());

        let node = ProvenanceNode::Inference {
            model_id: "m".to_string(),
            model_version: "v".to_string(),
            confidence: 0.9,
            output: 1.0,
        };
        assert!(node.timestamp_ns().is_none());

        let node = ProvenanceNode::Fusion {
            strategy: "avg".to_string(),
            input_refs: vec![],
        };
        assert!(node.timestamp_ns().is_none());

        let node = ProvenanceNode::Action {
            action_type: "stop".to_string(),
            confidence: 0.8,
            alternatives: vec![],
        };
        assert!(node.timestamp_ns().is_none());
    }

    #[test]
    fn test_provenance_node_all_type_names() {
        assert_eq!(
            ProvenanceNode::Transform {
                operation: "x".to_string(),
                input_refs: vec![]
            }
            .type_name(),
            "Transform"
        );
        assert_eq!(
            ProvenanceNode::Fusion {
                strategy: "x".to_string(),
                input_refs: vec![]
            }
            .type_name(),
            "Fusion"
        );
        assert_eq!(
            ProvenanceNode::Action {
                action_type: "x".to_string(),
                confidence: 0.5,
                alternatives: vec![]
            }
            .type_name(),
            "Action"
        );
    }

    #[test]
    fn test_provenance_graph_default() {
        let graph = ProvenanceGraph::default();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_edges_accessor() {
        let mut graph = ProvenanceGraph::new();
        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Input {
            source: "b".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 100,
        });

        assert_eq!(graph.edges().len(), 1);
    }

    #[test]
    fn test_nodes_accessor() {
        let mut graph = ProvenanceGraph::new();
        graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        graph.add_node(ProvenanceNode::Input {
            source: "b".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });

        assert_eq!(graph.nodes().len(), 2);
    }

    #[test]
    fn test_incoming_edges_no_edges() {
        let mut graph = ProvenanceGraph::new();
        let id = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        assert!(graph.incoming_edges(id).is_empty());
    }

    #[test]
    fn test_outgoing_edges_no_edges() {
        let mut graph = ProvenanceGraph::new();
        let id = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        assert!(graph.outgoing_edges(id).is_empty());
    }

    #[test]
    fn test_predecessors_empty() {
        let mut graph = ProvenanceGraph::new();
        let id = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        assert!(graph.predecessors(id).is_empty());
    }

    #[test]
    fn test_successors_empty() {
        let mut graph = ProvenanceGraph::new();
        let id = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        assert!(graph.successors(id).is_empty());
    }

    #[test]
    fn test_get_node_nonexistent() {
        let graph = ProvenanceGraph::new();
        assert!(graph.get_node(999).is_none());
    }

    #[test]
    fn test_calculate_duration_with_timestamps() {
        let mut graph = ProvenanceGraph::new();

        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "sensor".to_string(),
            timestamp_ns: 1000,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Input {
            source: "sensor2".to_string(),
            timestamp_ns: 5000,
            hash: 0,
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 2000,
        });

        let reconstructor = IncidentReconstructor::new(&graph);
        let path = reconstructor.reconstruct_path(id2, 10);
        assert!(path.duration_ns >= 4000);
    }

    #[test]
    fn test_reconstruct_path_max_depth_exceeded() {
        let mut graph = ProvenanceGraph::new();

        let id1 = graph.add_node(ProvenanceNode::Input {
            source: "a".to_string(),
            timestamp_ns: 0,
            hash: 0,
        });
        let id2 = graph.add_node(ProvenanceNode::Transform {
            operation: "op1".to_string(),
            input_refs: vec![id1],
        });
        let id3 = graph.add_node(ProvenanceNode::Transform {
            operation: "op2".to_string(),
            input_refs: vec![id2],
        });

        graph.add_edge(ProvenanceEdge {
            from: id1,
            to: id2,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 100,
        });
        graph.add_edge(ProvenanceEdge {
            from: id2,
            to: id3,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 200,
        });

        let reconstructor = IncidentReconstructor::new(&graph);
        // Max depth 1 should not include id1
        let path = reconstructor.reconstruct_path(id3, 1);
        assert!(path.len() <= 3);
    }

    #[test]
    fn test_identify_anomalies_fusion_many_inputs() {
        let mut graph = ProvenanceGraph::new();

        // Create many input nodes
        let input_ids: Vec<NodeId> = (0..15)
            .map(|i| {
                graph.add_node(ProvenanceNode::Input {
                    source: format!("sensor_{i}"),
                    timestamp_ns: 0,
                    hash: 0,
                })
            })
            .collect();

        // Create fusion with many inputs
        let fusion_id = graph.add_node(ProvenanceNode::Fusion {
            strategy: "merge".to_string(),
            input_refs: input_ids.clone(),
        });

        // Add edges
        for id in &input_ids {
            graph.add_edge(ProvenanceEdge {
                from: *id,
                to: fusion_id,
                relation: CausalRelation::DataFlow,
                timestamp_ns: 100,
            });
        }

        let reconstructor = IncidentReconstructor::new(&graph);
        let path = reconstructor.reconstruct_path(fusion_id, 10);
        let anomalies = reconstructor.identify_anomalies(&path, 0.7);

        assert!(anomalies
            .iter()
            .any(|a| a.description.contains("many fusion inputs")));
    }

    #[test]
    fn test_identify_anomalies_no_predecessors() {
        let mut graph = ProvenanceGraph::new();

        // Create a transform node with no predecessors (orphan)
        let id = graph.add_node(ProvenanceNode::Transform {
            operation: "orphan".to_string(),
            input_refs: vec![],
        });

        let reconstructor = IncidentReconstructor::new(&graph);
        let path = reconstructor.reconstruct_path(id, 10);
        let anomalies = reconstructor.identify_anomalies(&path, 0.7);

        assert!(anomalies
            .iter()
            .any(|a| a.description.contains("no predecessors")));
    }

    #[test]
    fn test_attack_path_len() {
        let path = AttackPath {
            nodes: vec![(
                0,
                ProvenanceNode::Input {
                    source: "a".to_string(),
                    timestamp_ns: 0,
                    hash: 0,
                },
            )],
            edges: vec![],
            duration_ns: 0,
            anomaly_indices: vec![],
        };
        assert_eq!(path.len(), 1);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_causal_relation_all_variants() {
        assert_eq!(CausalRelation::DataFlow, CausalRelation::DataFlow);
        assert_eq!(CausalRelation::Triggered, CausalRelation::Triggered);
        assert_eq!(CausalRelation::Influenced, CausalRelation::Influenced);
        assert_eq!(CausalRelation::Caused, CausalRelation::Caused);
        assert_ne!(CausalRelation::Influenced, CausalRelation::Caused);
    }

    #[test]
    fn test_provenance_edge_clone() {
        let edge = ProvenanceEdge {
            from: 1,
            to: 2,
            relation: CausalRelation::DataFlow,
            timestamp_ns: 1000,
        };
        let cloned = edge.clone();
        assert_eq!(edge.from, cloned.from);
        assert_eq!(edge.to, cloned.to);
        assert_eq!(edge.relation, cloned.relation);
        assert_eq!(edge.timestamp_ns, cloned.timestamp_ns);
    }

    #[test]
    fn test_anomaly_struct() {
        let anomaly = Anomaly {
            node_id: 42,
            description: "test anomaly".to_string(),
            severity: 0.75,
        };
        assert_eq!(anomaly.node_id, 42);
        assert_eq!(anomaly.description, "test anomaly");
        assert!((anomaly.severity - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_serde_provenance_node() {
        let node = ProvenanceNode::Input {
            source: "test".to_string(),
            timestamp_ns: 12345,
            hash: 0xdeadbeef,
        };
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: ProvenanceNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.type_name(), "Input");
    }

    #[test]
    fn test_serde_provenance_edge() {
        let edge = ProvenanceEdge {
            from: 1,
            to: 2,
            relation: CausalRelation::Triggered,
            timestamp_ns: 5000,
        };
        let json = serde_json::to_string(&edge).unwrap();
        let deserialized: ProvenanceEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.from, 1);
        assert_eq!(deserialized.to, 2);
        assert_eq!(deserialized.relation, CausalRelation::Triggered);
    }

    #[test]
    fn test_serde_causal_relation() {
        let relation = CausalRelation::Influenced;
        let json = serde_json::to_string(&relation).unwrap();
        let deserialized: CausalRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, CausalRelation::Influenced);
    }

    #[test]
    fn test_add_inference_from_trace() {
        use super::super::path::LinearPath;
        use super::super::trace::DecisionTrace;

        // LinearPath::new(contributions, intercept, logit, prediction)
        let path = LinearPath::new(vec![0.5, 0.3, 0.2], 0.1, 0.7, 0.65);
        // DecisionTrace::new(timestamp_ns, sequence, input_hash, path, output, latency_ns)
        let trace = DecisionTrace::new(1000, 1, 0xdeadbeef, path, 0.7, 100);

        let mut graph = ProvenanceGraph::new();
        let id = graph.add_inference(&trace, "model-1", "v1.0");

        let node = graph.get_node(id).unwrap();
        assert_eq!(node.type_name(), "Inference");
    }

    #[test]
    fn test_attack_path_clone() {
        let path = AttackPath {
            nodes: vec![(
                0,
                ProvenanceNode::Input {
                    source: "a".to_string(),
                    timestamp_ns: 0,
                    hash: 0,
                },
            )],
            edges: vec![],
            duration_ns: 1000,
            anomaly_indices: vec![0],
        };
        let cloned = path.clone();
        assert_eq!(path.len(), cloned.len());
        assert_eq!(path.duration_ns, cloned.duration_ns);
        assert_eq!(path.anomaly_indices.len(), cloned.anomaly_indices.len());
    }

    #[test]
    fn test_anomaly_clone() {
        let anomaly = Anomaly {
            node_id: 1,
            description: "test".to_string(),
            severity: 0.5,
        };
        let cloned = anomaly.clone();
        assert_eq!(anomaly.node_id, cloned.node_id);
        assert_eq!(anomaly.description, cloned.description);
    }
}

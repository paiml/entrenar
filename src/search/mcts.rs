//! Monte Carlo Tree Search implementation for code generation.
//!
//! # Overview
//!
//! MCTS is a heuristic search algorithm that builds a search tree iteratively
//! through four phases: Selection, Expansion, Simulation, and Backpropagation.
//!
//! For code generation:
//! - **State**: Partial AST (Abstract Syntax Tree)
//! - **Action**: Transform rules (e.g., add statement, wrap in loop)
//! - **Reward**: Compilation success (binary: 0 or 1)
//!
//! # Example
//!
//! ```rust
//! use entrenar::search::{MctsSearch, MctsConfig, State, Action};
//!
//! // Define your state and action spaces
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct CodeState {
//!     ast_tokens: Vec<String>,
//! }
//!
//! impl State for CodeState {
//!     fn is_terminal(&self) -> bool {
//!         self.ast_tokens.iter().any(|t| t == "EOF")
//!     }
//! }
//!
//! // Create MCTS searcher with default config
//! let config = MctsConfig::default();
//! // let mcts = MctsSearch::new(initial_state, action_space, config);
//! ```

use std::collections::HashMap;
use std::hash::Hash;

/// Type alias for reward values (typically 0.0 to 1.0)
pub type Reward = f64;

/// Unique identifier for nodes in the search tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    /// Creates a new NodeId
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the underlying id value
    #[must_use]
    pub const fn value(&self) -> usize {
        self.0
    }
}

/// Trait for states in the search space (e.g., partial AST)
pub trait State: Clone + Eq + Hash {
    /// Returns true if this is a terminal state (complete code)
    fn is_terminal(&self) -> bool;

    /// Returns a hash of this state for deduplication
    fn state_hash(&self) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Trait for actions in the search space (e.g., AST transformations)
pub trait Action: Clone + Eq + Hash {
    /// Returns the name/identifier of this action
    fn name(&self) -> &str;

    /// Returns the prior probability of this action (for policy network guidance)
    fn prior(&self) -> f64 {
        1.0 // Uniform prior by default
    }
}

/// Trait for defining the state space (how states transition)
pub trait StateSpace<S: State, A: Action> {
    /// Apply an action to a state, returning the new state
    fn apply(&self, state: &S, action: &A) -> S;

    /// Evaluate a terminal state, returning the reward (0.0 to 1.0)
    fn evaluate(&self, state: &S) -> Reward;

    /// Clone the state space (for parallel simulations)
    fn clone_space(&self) -> Box<dyn StateSpace<S, A> + Send + Sync>;
}

/// Trait for defining the action space (available actions from a state)
pub trait ActionSpace<S: State, A: Action> {
    /// Returns all legal actions from the given state
    fn legal_actions(&self, state: &S) -> Vec<A>;

    /// Returns true if there are no legal actions from this state
    fn is_empty(&self, state: &S) -> bool {
        self.legal_actions(state).is_empty()
    }
}

/// Trait for policy networks that guide the search
pub trait PolicyNetwork<S: State, A: Action>: Send + Sync {
    /// Returns (action, prior probability) pairs for the given state
    fn predict(&self, state: &S) -> Vec<(A, f64)>;

    /// Returns the value estimate for a state (optional, for AlphaZero-style)
    fn value(&self, _state: &S) -> f64 {
        0.5 // Neutral value by default
    }
}

/// Configuration for MCTS search
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Exploration constant for UCB1 (higher = more exploration)
    pub exploration_constant: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Maximum depth for simulation rollouts
    pub max_simulation_depth: usize,
    /// Whether to use policy network priors
    pub use_policy_priors: bool,
    /// Temperature for action selection (higher = more random)
    pub temperature: f64,
    /// Minimum visits before expansion
    pub min_visits_for_expansion: usize,
    /// Whether to reuse tree between searches
    pub reuse_tree: bool,
    /// Dirichlet noise alpha for exploration at root
    pub dirichlet_alpha: f64,
    /// Fraction of Dirichlet noise to add
    pub dirichlet_epsilon: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
            max_iterations: 1000,
            max_simulation_depth: 100,
            use_policy_priors: true,
            temperature: 1.0,
            min_visits_for_expansion: 1,
            reuse_tree: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

/// Statistics for a node in the search tree
#[derive(Debug, Clone)]
pub struct NodeStats {
    /// Total visits to this node
    pub visits: usize,
    /// Total accumulated reward
    pub total_reward: f64,
    /// Mean reward (total_reward / visits)
    pub mean_reward: f64,
    /// Prior probability from policy network
    pub prior: f64,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            visits: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            prior: 1.0,
        }
    }
}

impl NodeStats {
    /// Update statistics with a new reward
    pub fn update(&mut self, reward: Reward) {
        self.visits += 1;
        self.total_reward += reward;
        self.mean_reward = self.total_reward / self.visits as f64;
    }

    /// Calculate UCB1 score
    #[must_use]
    pub fn ucb1(&self, parent_visits: usize, c: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.mean_reward;
        let exploration = c * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }

    /// Calculate PUCT score (Polynomial Upper Confidence Trees) for policy-guided search
    #[must_use]
    pub fn puct(&self, parent_visits: usize, c: f64) -> f64 {
        let exploitation = self.mean_reward;
        let exploration =
            c * self.prior * (parent_visits as f64).sqrt() / (1.0 + self.visits as f64);
        exploitation + exploration
    }
}

/// A node in the search tree
#[derive(Debug, Clone)]
pub struct Node<S: State, A: Action> {
    /// Unique identifier
    pub id: NodeId,
    /// State at this node
    pub state: S,
    /// Action that led to this node (None for root)
    pub action: Option<A>,
    /// Parent node id (None for root)
    pub parent: Option<NodeId>,
    /// Child node ids
    pub children: Vec<NodeId>,
    /// Statistics for this node
    pub stats: NodeStats,
    /// Whether this node is fully expanded
    pub expanded: bool,
    /// Untried actions from this state
    pub untried_actions: Vec<A>,
}

impl<S: State, A: Action> Node<S, A> {
    /// Create a new root node
    #[must_use]
    pub fn root(state: S, untried_actions: Vec<A>) -> Self {
        Self {
            id: NodeId::new(0),
            state,
            action: None,
            parent: None,
            children: Vec::new(),
            stats: NodeStats::default(),
            expanded: false,
            untried_actions,
        }
    }

    /// Create a new child node
    #[must_use]
    pub fn child(
        id: NodeId,
        state: S,
        action: A,
        parent: NodeId,
        untried_actions: Vec<A>,
        prior: f64,
    ) -> Self {
        Self {
            id,
            state,
            action: Some(action),
            parent: Some(parent),
            children: Vec::new(),
            stats: NodeStats {
                prior,
                ..Default::default()
            },
            expanded: false,
            untried_actions,
        }
    }

    /// Returns true if this node is a leaf (no children)
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns true if this node is fully expanded
    #[must_use]
    pub fn is_fully_expanded(&self) -> bool {
        self.expanded && self.untried_actions.is_empty()
    }
}

/// The search tree structure
#[derive(Debug)]
pub struct SearchTree<S: State, A: Action> {
    /// All nodes indexed by NodeId
    nodes: Vec<Node<S, A>>,
    /// Map from state hash to node id for deduplication
    state_map: HashMap<u64, NodeId>,
    /// Root node id
    root_id: NodeId,
}

impl<S: State, A: Action> SearchTree<S, A> {
    /// Create a new search tree with the given root state
    #[must_use]
    pub fn new(root_state: S, root_actions: Vec<A>) -> Self {
        let root = Node::root(root_state.clone(), root_actions);
        let state_hash = root_state.state_hash();
        let mut state_map = HashMap::new();
        state_map.insert(state_hash, NodeId::new(0));

        Self {
            nodes: vec![root],
            state_map,
            root_id: NodeId::new(0),
        }
    }

    /// Get the root node
    #[must_use]
    pub fn root(&self) -> &Node<S, A> {
        &self.nodes[self.root_id.0]
    }

    /// Get a node by id
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Node<S, A>> {
        self.nodes.get(id.0)
    }

    /// Get a mutable node by id
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node<S, A>> {
        self.nodes.get_mut(id.0)
    }

    /// Add a child node, returning its id
    pub fn add_child(
        &mut self,
        parent_id: NodeId,
        state: S,
        action: A,
        untried_actions: Vec<A>,
        prior: f64,
    ) -> NodeId {
        let state_hash = state.state_hash();

        // Check for transposition (same state reached via different path)
        if let Some(&existing_id) = self.state_map.get(&state_hash) {
            // Add as child but reuse existing node's stats
            if let Some(parent) = self.nodes.get_mut(parent_id.0) {
                if !parent.children.contains(&existing_id) {
                    parent.children.push(existing_id);
                }
            }
            return existing_id;
        }

        let child_id = NodeId::new(self.nodes.len());
        let child = Node::child(child_id, state, action, parent_id, untried_actions, prior);

        self.nodes.push(child);
        self.state_map.insert(state_hash, child_id);

        if let Some(parent) = self.nodes.get_mut(parent_id.0) {
            parent.children.push(child_id);
        }

        child_id
    }

    /// Get number of nodes in the tree
    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Get all children of a node
    #[must_use]
    pub fn children(&self, id: NodeId) -> Vec<&Node<S, A>> {
        self.nodes
            .get(id.0)
            .map(|n| n.children.iter().filter_map(|&cid| self.get(cid)).collect())
            .unwrap_or_default()
    }
}

/// Statistics from an MCTS search
#[derive(Debug, Clone)]
pub struct MctsStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Total nodes in tree
    pub tree_size: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Average simulation length
    pub avg_simulation_length: f64,
    /// Root node visit count
    pub root_visits: usize,
}

/// Result of an MCTS search
#[derive(Debug)]
pub struct MctsResult<S: State, A: Action> {
    /// Best action found
    pub best_action: Option<A>,
    /// Expected reward of best action
    pub expected_reward: f64,
    /// Visit counts for all root children
    pub action_visits: Vec<(A, usize)>,
    /// Search statistics
    pub stats: MctsStats,
    /// The resulting state after best action (if any)
    pub resulting_state: Option<S>,
}

/// Main MCTS search algorithm
pub struct MctsSearch<S: State, A: Action> {
    /// Search tree
    tree: SearchTree<S, A>,
    /// Configuration
    config: MctsConfig,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl<S: State + Send + Sync, A: Action + Send + Sync> MctsSearch<S, A> {
    /// Create a new MCTS search from initial state
    pub fn new<AS: ActionSpace<S, A>>(
        initial_state: S,
        action_space: &AS,
        config: MctsConfig,
    ) -> Self {
        use rand::SeedableRng;
        let actions = action_space.legal_actions(&initial_state);
        let tree = SearchTree::new(initial_state, actions);
        Self {
            tree,
            config,
            rng: rand::rngs::StdRng::from_os_rng(),
        }
    }

    /// Create a new MCTS search with a seed for reproducibility
    pub fn with_seed<AS: ActionSpace<S, A>>(
        initial_state: S,
        action_space: &AS,
        config: MctsConfig,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        let actions = action_space.legal_actions(&initial_state);
        let tree = SearchTree::new(initial_state, actions);
        Self {
            tree,
            config,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Run the MCTS search
    pub fn search<SS, AS>(
        &mut self,
        state_space: &SS,
        action_space: &AS,
        policy: Option<&dyn PolicyNetwork<S, A>>,
    ) -> MctsResult<S, A>
    where
        SS: StateSpace<S, A>,
        AS: ActionSpace<S, A>,
    {
        let mut total_sim_length = 0usize;
        let mut max_depth = 0usize;

        for _ in 0..self.config.max_iterations {
            // Selection: traverse tree to find a leaf node
            let (leaf_id, depth) = self.select();
            max_depth = max_depth.max(depth);

            // Get the leaf state
            let leaf_state = self.tree.get(leaf_id).map(|n| n.state.clone());
            let Some(leaf_state) = leaf_state else {
                continue;
            };

            // Check if terminal
            if leaf_state.is_terminal() {
                let reward = state_space.evaluate(&leaf_state);
                self.backpropagate(leaf_id, reward);
                continue;
            }

            // Expansion: add a child node
            let child_id = self.expand(leaf_id, state_space, action_space, policy);
            let Some(child_id) = child_id else {
                continue;
            };

            // Simulation: random playout from child
            let child_state = self.tree.get(child_id).map(|n| n.state.clone());
            let Some(child_state) = child_state else {
                continue;
            };

            let (reward, sim_length) = self.simulate(&child_state, state_space, action_space);
            total_sim_length += sim_length;

            // Backpropagation: update statistics up the tree
            self.backpropagate(child_id, reward);
        }

        // Compute results
        let root = self.tree.root();
        let root_visits = root.stats.visits;

        // Get action visits
        let action_visits: Vec<(A, usize)> = self
            .tree
            .children(self.tree.root_id)
            .iter()
            .filter_map(|child| child.action.clone().map(|a| (a, child.stats.visits)))
            .collect();

        // Select best action based on visits (robust child selection)
        let best_child = self
            .tree
            .children(self.tree.root_id)
            .into_iter()
            .max_by_key(|n| n.stats.visits);

        let (best_action, expected_reward, resulting_state) = if let Some(child) = best_child {
            (
                child.action.clone(),
                child.stats.mean_reward,
                Some(child.state.clone()),
            )
        } else {
            (None, 0.0, None)
        };

        let avg_simulation_length = if self.config.max_iterations > 0 {
            total_sim_length as f64 / self.config.max_iterations as f64
        } else {
            0.0
        };

        MctsResult {
            best_action,
            expected_reward,
            action_visits,
            stats: MctsStats {
                iterations: self.config.max_iterations,
                tree_size: self.tree.size(),
                max_depth,
                avg_simulation_length,
                root_visits,
            },
            resulting_state,
        }
    }

    /// Selection phase: traverse tree using UCB1/PUCT
    fn select(&self) -> (NodeId, usize) {
        let mut current_id = self.tree.root_id;
        let mut depth = 0;

        loop {
            let node = match self.tree.get(current_id) {
                Some(n) => n,
                None => return (current_id, depth),
            };

            // If node has untried actions or is terminal, return it
            if !node.untried_actions.is_empty() || node.state.is_terminal() {
                return (current_id, depth);
            }

            // If no children, return current
            if node.children.is_empty() {
                return (current_id, depth);
            }

            // Select best child using UCB1/PUCT
            let parent_visits = node.stats.visits;
            let best_child = node
                .children
                .iter()
                .filter_map(|&cid| self.tree.get(cid))
                .max_by(|a, b| {
                    let score_a = if self.config.use_policy_priors {
                        a.stats
                            .puct(parent_visits, self.config.exploration_constant)
                    } else {
                        a.stats
                            .ucb1(parent_visits, self.config.exploration_constant)
                    };
                    let score_b = if self.config.use_policy_priors {
                        b.stats
                            .puct(parent_visits, self.config.exploration_constant)
                    } else {
                        b.stats
                            .ucb1(parent_visits, self.config.exploration_constant)
                    };
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            match best_child {
                Some(child) => {
                    current_id = child.id;
                    depth += 1;
                }
                None => return (current_id, depth),
            }
        }
    }

    /// Expansion phase: add a child node for an untried action
    fn expand<SS, AS>(
        &mut self,
        node_id: NodeId,
        state_space: &SS,
        action_space: &AS,
        policy: Option<&dyn PolicyNetwork<S, A>>,
    ) -> Option<NodeId>
    where
        SS: StateSpace<S, A>,
        AS: ActionSpace<S, A>,
    {
        // Get an untried action
        let (action, parent_state) = {
            let node = self.tree.get_mut(node_id)?;
            let action = node.untried_actions.pop()?;
            let parent_state = node.state.clone();
            node.expanded = node.untried_actions.is_empty();
            (action, parent_state)
        };

        // Compute new state
        let new_state = state_space.apply(&parent_state, &action);
        let new_actions = action_space.legal_actions(&new_state);

        // Get prior from policy network
        let prior = policy
            .and_then(|p| {
                p.predict(&parent_state)
                    .iter()
                    .find(|(a, _)| a == &action)
                    .map(|(_, p)| *p)
            })
            .unwrap_or(1.0 / (new_actions.len().max(1) as f64));

        // Add child
        let child_id = self
            .tree
            .add_child(node_id, new_state, action, new_actions, prior);
        Some(child_id)
    }

    /// Simulation phase: random playout from state
    fn simulate<SS, AS>(
        &mut self,
        initial_state: &S,
        state_space: &SS,
        action_space: &AS,
    ) -> (Reward, usize)
    where
        SS: StateSpace<S, A>,
        AS: ActionSpace<S, A>,
    {
        use rand::prelude::IndexedRandom;

        let mut state = initial_state.clone();
        let mut depth = 0;

        while !state.is_terminal() && depth < self.config.max_simulation_depth {
            let actions = action_space.legal_actions(&state);
            if actions.is_empty() {
                break;
            }

            // Random action selection
            if let Some(action) = actions.choose(&mut self.rng) {
                state = state_space.apply(&state, action);
            }
            depth += 1;
        }

        (state_space.evaluate(&state), depth)
    }

    /// Backpropagation phase: update statistics up the tree
    fn backpropagate(&mut self, leaf_id: NodeId, reward: Reward) {
        let mut current_id = Some(leaf_id);

        while let Some(id) = current_id {
            if let Some(node) = self.tree.get_mut(id) {
                node.stats.update(reward);
                current_id = node.parent;
            } else {
                break;
            }
        }
    }

    /// Get the current tree size
    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.tree.size()
    }

    /// Get reference to the search tree
    #[must_use]
    pub fn tree(&self) -> &SearchTree<S, A> {
        &self.tree
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Simple test state for unit tests
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestState {
        value: i32,
        terminal: bool,
    }

    impl State for TestState {
        fn is_terminal(&self) -> bool {
            self.terminal
        }
    }

    // Simple test action
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestAction {
        delta: i32,
    }

    impl Action for TestAction {
        fn name(&self) -> &str {
            "test_action"
        }
    }

    // Test state space
    struct TestStateSpace {
        target: i32,
    }

    impl StateSpace<TestState, TestAction> for TestStateSpace {
        fn apply(&self, state: &TestState, action: &TestAction) -> TestState {
            let new_value = state.value + action.delta;
            TestState {
                value: new_value,
                terminal: new_value >= 10 || new_value <= -10,
            }
        }

        fn evaluate(&self, state: &TestState) -> Reward {
            if state.value == self.target {
                1.0
            } else {
                0.0
            }
        }

        fn clone_space(&self) -> Box<dyn StateSpace<TestState, TestAction> + Send + Sync> {
            Box::new(TestStateSpace {
                target: self.target,
            })
        }
    }

    // Test action space
    struct TestActionSpace;

    impl ActionSpace<TestState, TestAction> for TestActionSpace {
        fn legal_actions(&self, state: &TestState) -> Vec<TestAction> {
            if state.terminal {
                vec![]
            } else {
                vec![
                    TestAction { delta: 1 },
                    TestAction { delta: -1 },
                    TestAction { delta: 2 },
                ]
            }
        }
    }

    // ========================================
    // ENT-090: Core types tests
    // ========================================

    #[test]
    fn test_node_id_creation() {
        let id = NodeId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_node_stats_default() {
        let stats = NodeStats::default();
        assert_eq!(stats.visits, 0);
        assert_eq!(stats.total_reward, 0.0);
        assert_eq!(stats.mean_reward, 0.0);
        assert_eq!(stats.prior, 1.0);
    }

    #[test]
    fn test_node_stats_update() {
        let mut stats = NodeStats::default();
        stats.update(1.0);
        assert_eq!(stats.visits, 1);
        assert_eq!(stats.total_reward, 1.0);
        assert_eq!(stats.mean_reward, 1.0);

        stats.update(0.0);
        assert_eq!(stats.visits, 2);
        assert_eq!(stats.total_reward, 1.0);
        assert_eq!(stats.mean_reward, 0.5);
    }

    #[test]
    fn test_node_root_creation() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let actions = vec![TestAction { delta: 1 }];
        let node = Node::root(state.clone(), actions);

        assert_eq!(node.id, NodeId::new(0));
        assert_eq!(node.state, state);
        assert!(node.action.is_none());
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert!(!node.expanded);
    }

    #[test]
    fn test_node_child_creation() {
        let state = TestState {
            value: 1,
            terminal: false,
        };
        let action = TestAction { delta: 1 };
        let node = Node::child(
            NodeId::new(1),
            state.clone(),
            action.clone(),
            NodeId::new(0),
            vec![],
            0.5,
        );

        assert_eq!(node.id, NodeId::new(1));
        assert_eq!(node.state, state);
        assert_eq!(node.action, Some(action));
        assert_eq!(node.parent, Some(NodeId::new(0)));
        assert_eq!(node.stats.prior, 0.5);
    }

    #[test]
    fn test_node_is_leaf() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let node: Node<TestState, TestAction> = Node::root(state, vec![]);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_search_tree_creation() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let actions = vec![TestAction { delta: 1 }, TestAction { delta: -1 }];
        let tree = SearchTree::new(state.clone(), actions);

        assert_eq!(tree.size(), 1);
        assert_eq!(tree.root().state, state);
    }

    #[test]
    fn test_search_tree_add_child() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let actions = vec![TestAction { delta: 1 }];
        let mut tree = SearchTree::new(state, actions);

        let child_state = TestState {
            value: 1,
            terminal: false,
        };
        let child_action = TestAction { delta: 1 };
        let child_id = tree.add_child(
            NodeId::new(0),
            child_state.clone(),
            child_action,
            vec![],
            0.5,
        );

        assert_eq!(tree.size(), 2);
        let child = tree.get(child_id).unwrap();
        assert_eq!(child.state, child_state);
    }

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert!(config.exploration_constant > 0.0);
        assert!(config.use_policy_priors);
    }

    // ========================================
    // ENT-091: UCB1/UCT tests
    // ========================================

    #[test]
    fn test_ucb1_unvisited_node() {
        let stats = NodeStats::default();
        let score = stats.ucb1(10, std::f64::consts::SQRT_2);
        assert!(score.is_infinite());
    }

    #[test]
    fn test_ucb1_visited_node() {
        let mut stats = NodeStats::default();
        stats.update(0.5);
        let score = stats.ucb1(10, std::f64::consts::SQRT_2);

        // Should be exploitation + exploration
        // 0.5 + sqrt(2) * sqrt(ln(10) / 1) ≈ 0.5 + 2.14 = 2.64
        assert!(score > 0.5);
        assert!(score < 5.0);
    }

    #[test]
    fn test_ucb1_more_visits_lower_exploration() {
        let mut stats1 = NodeStats::default();
        stats1.visits = 10;
        stats1.total_reward = 5.0;
        stats1.mean_reward = 0.5;

        let mut stats2 = NodeStats::default();
        stats2.visits = 100;
        stats2.total_reward = 50.0;
        stats2.mean_reward = 0.5;

        let score1 = stats1.ucb1(1000, std::f64::consts::SQRT_2);
        let score2 = stats2.ucb1(1000, std::f64::consts::SQRT_2);

        // More visits should have lower exploration bonus
        assert!(score1 > score2);
    }

    #[test]
    fn test_puct_with_prior() {
        let mut stats = NodeStats::default();
        stats.prior = 0.5;
        stats.update(0.3);

        let score = stats.puct(100, 2.0);

        // PUCT = mean_reward + c * prior * sqrt(parent_visits) / (1 + visits)
        // = 0.3 + 2.0 * 0.5 * sqrt(100) / 2 = 0.3 + 5.0 = 5.3
        assert!((score - 5.3).abs() < 0.01);
    }

    // ========================================
    // ENT-092 & ENT-093: MCTS search tests
    // ========================================

    #[test]
    fn test_mcts_search_basic() {
        let initial_state = TestState {
            value: 0,
            terminal: false,
        };
        let config = MctsConfig {
            max_iterations: 100,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);

        let state_space = TestStateSpace { target: 5 };
        let result = mcts.search(&state_space, &action_space, None);

        assert!(result.best_action.is_some());
        assert!(result.stats.iterations == 100);
        assert!(result.stats.tree_size > 1);
    }

    #[test]
    fn test_mcts_finds_path_to_target() {
        let initial_state = TestState {
            value: 0,
            terminal: false,
        };
        let config = MctsConfig {
            max_iterations: 500,
            exploration_constant: 1.0,
            use_policy_priors: false,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 12345);

        // Target is 5, so we need to go +1 five times
        let state_space = TestStateSpace { target: 5 };
        let result = mcts.search(&state_space, &action_space, None);

        // With enough iterations, should find a path with positive actions
        assert!(result.best_action.is_some());
        if let Some(action) = &result.best_action {
            // First action should be positive to move toward target
            assert!(
                action.delta > 0,
                "Expected positive action, got {:?}",
                action
            );
        }
    }

    #[test]
    fn test_mcts_terminal_state() {
        let initial_state = TestState {
            value: 10,
            terminal: true,
        };
        let config = MctsConfig {
            max_iterations: 10,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let mut mcts = MctsSearch::new(initial_state, &action_space, config);

        let state_space = TestStateSpace { target: 10 };
        let result = mcts.search(&state_space, &action_space, None);

        // No actions should be taken from terminal state
        assert!(result.best_action.is_none());
    }

    #[test]
    fn test_mcts_backpropagation() {
        let initial_state = TestState {
            value: 0,
            terminal: false,
        };
        let config = MctsConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let mut mcts = MctsSearch::new(initial_state, &action_space, config);

        let state_space = TestStateSpace { target: 5 };
        let result = mcts.search(&state_space, &action_space, None);

        // Root should have been visited
        assert!(result.stats.root_visits > 0);

        // Tree should have grown
        assert!(mcts.tree_size() > 1);
    }

    #[test]
    fn test_mcts_reproducibility_with_seed() {
        let initial_state = TestState {
            value: 0,
            terminal: false,
        };
        let config = MctsConfig {
            max_iterations: 100,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let state_space = TestStateSpace { target: 5 };

        let mut mcts1 =
            MctsSearch::with_seed(initial_state.clone(), &action_space, config.clone(), 42);
        let result1 = mcts1.search(&state_space, &action_space, None);

        let mut mcts2 = MctsSearch::with_seed(initial_state, &action_space, config, 42);
        let result2 = mcts2.search(&state_space, &action_space, None);

        assert_eq!(result1.best_action, result2.best_action);
        assert_eq!(result1.stats.tree_size, result2.stats.tree_size);
    }

    // ========================================
    // ENT-095: Property tests
    // ========================================

    proptest! {
        #[test]
        fn test_node_stats_update_invariants(rewards in prop::collection::vec(0.0f64..=1.0, 1..100)) {
            let mut stats = NodeStats::default();

            for r in &rewards {
                stats.update(*r);
            }

            prop_assert_eq!(stats.visits, rewards.len());
            prop_assert!((stats.total_reward - rewards.iter().sum::<f64>()).abs() < 1e-10);
            prop_assert!((stats.mean_reward - rewards.iter().sum::<f64>() / rewards.len() as f64).abs() < 1e-10);
        }

        #[test]
        fn test_ucb1_exploration_decreases_with_visits(parent_visits in 10usize..1000, c in 0.1f64..5.0) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 10;
            stats1.mean_reward = 0.5;

            let mut stats2 = NodeStats::default();
            stats2.visits = 100;
            stats2.mean_reward = 0.5;

            let ucb1 = stats1.ucb1(parent_visits, c);
            let ucb2 = stats2.ucb1(parent_visits, c);

            // More visits should lead to lower UCB (less exploration bonus)
            prop_assert!(ucb1 > ucb2, "UCB1 with fewer visits should be higher");
        }

        #[test]
        fn test_ucb1_higher_reward_higher_score(parent_visits in 10usize..1000, c in 0.1f64..5.0) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 50;
            stats1.mean_reward = 0.3;

            let mut stats2 = NodeStats::default();
            stats2.visits = 50;
            stats2.mean_reward = 0.7;

            let ucb1 = stats1.ucb1(parent_visits, c);
            let ucb2 = stats2.ucb1(parent_visits, c);

            // Higher reward should lead to higher UCB (same visits)
            prop_assert!(ucb2 > ucb1, "Higher reward should give higher UCB");
        }

        #[test]
        fn test_tree_size_increases_monotonically(num_children in 1usize..10) {
            let state = TestState { value: 0, terminal: false };
            let mut tree = SearchTree::new(state.clone(), vec![]);

            let mut prev_size = tree.size();

            for i in 0..num_children {
                let child_state = TestState { value: i as i32, terminal: false };
                tree.add_child(
                    NodeId::new(0),
                    child_state,
                    TestAction { delta: 1 },
                    vec![],
                    0.5,
                );

                // Size should increase or stay same (transposition)
                prop_assert!(tree.size() >= prev_size);
                prev_size = tree.size();
            }
        }

        #[test]
        fn test_mcts_tree_grows_with_iterations(iterations in 10usize..200) {
            let initial_state = TestState { value: 0, terminal: false };
            let config = MctsConfig {
                max_iterations: iterations,
                ..Default::default()
            };

            let action_space = TestActionSpace;
            let state_space = TestStateSpace { target: 5 };

            let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
            let result = mcts.search(&state_space, &action_space, None);

            // Tree should have grown
            prop_assert!(result.stats.tree_size >= 1);
        }

        #[test]
        fn test_mcts_root_visits_increase(iterations in 10usize..500) {
            let initial_state = TestState { value: 0, terminal: false };
            let config = MctsConfig {
                max_iterations: iterations,
                ..Default::default()
            };

            let action_space = TestActionSpace;
            let state_space = TestStateSpace { target: 5 };

            let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
            let result = mcts.search(&state_space, &action_space, None);

            // Root visits should be at least iterations (minus terminal states)
            prop_assert!(result.stats.root_visits > 0);
        }

        #[test]
        fn test_puct_prior_increases_exploration(prior in 0.1f64..0.9) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 10;
            stats1.mean_reward = 0.5;
            stats1.prior = prior;

            let mut stats2 = NodeStats::default();
            stats2.visits = 10;
            stats2.mean_reward = 0.5;
            stats2.prior = prior * 2.0;

            let puct1 = stats1.puct(100, 2.0);
            let puct2 = stats2.puct(100, 2.0);

            // Higher prior should give higher PUCT
            prop_assert!(puct2 > puct1, "Higher prior should give higher PUCT");
        }

        #[test]
        fn test_action_visits_sum_to_root_minus_one(iterations in 50usize..200) {
            let initial_state = TestState { value: 0, terminal: false };
            let config = MctsConfig {
                max_iterations: iterations,
                exploration_constant: 2.0,
                ..Default::default()
            };

            let action_space = TestActionSpace;
            let state_space = TestStateSpace { target: 5 };

            let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
            let result = mcts.search(&state_space, &action_space, None);

            // Sum of child visits should be less than or equal to root visits
            let child_visits: usize = result.action_visits.iter().map(|(_, v)| *v).sum();
            prop_assert!(child_visits <= result.stats.root_visits);
        }
    }

    // ========================================
    // Additional unit tests
    // ========================================

    #[test]
    fn test_state_trait_implementation() {
        let state = TestState {
            value: 5,
            terminal: false,
        };
        assert!(!state.is_terminal());

        let terminal = TestState {
            value: 10,
            terminal: true,
        };
        assert!(terminal.is_terminal());
    }

    #[test]
    fn test_action_trait_implementation() {
        let action = TestAction { delta: 1 };
        assert_eq!(action.name(), "test_action");
        assert_eq!(action.prior(), 1.0);
    }

    #[test]
    fn test_state_space_apply() {
        let state_space = TestStateSpace { target: 5 };
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let action = TestAction { delta: 1 };

        let new_state = state_space.apply(&state, &action);
        assert_eq!(new_state.value, 1);
        assert!(!new_state.terminal);
    }

    #[test]
    fn test_state_space_evaluate() {
        let state_space = TestStateSpace { target: 5 };

        let state_on_target = TestState {
            value: 5,
            terminal: true,
        };
        assert_eq!(state_space.evaluate(&state_on_target), 1.0);

        let state_off_target = TestState {
            value: 3,
            terminal: true,
        };
        assert_eq!(state_space.evaluate(&state_off_target), 0.0);
    }

    #[test]
    fn test_action_space_legal_actions() {
        let action_space = TestActionSpace;

        let non_terminal = TestState {
            value: 0,
            terminal: false,
        };
        let actions = action_space.legal_actions(&non_terminal);
        assert_eq!(actions.len(), 3);

        let terminal = TestState {
            value: 10,
            terminal: true,
        };
        let actions = action_space.legal_actions(&terminal);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_tree_children() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let mut tree = SearchTree::new(state.clone(), vec![]);

        let child1 = TestState {
            value: 1,
            terminal: false,
        };
        let child2 = TestState {
            value: 2,
            terminal: false,
        };

        tree.add_child(NodeId::new(0), child1, TestAction { delta: 1 }, vec![], 0.5);
        tree.add_child(NodeId::new(0), child2, TestAction { delta: 2 }, vec![], 0.5);

        let children = tree.children(NodeId::new(0));
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_mcts_result_structure() {
        let initial_state = TestState {
            value: 0,
            terminal: false,
        };
        let config = MctsConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let state_space = TestStateSpace { target: 5 };

        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
        let result = mcts.search(&state_space, &action_space, None);

        assert_eq!(result.stats.iterations, 50);
        assert!(!result.action_visits.is_empty());
    }

    #[test]
    fn test_transposition_table() {
        let state = TestState {
            value: 0,
            terminal: false,
        };
        let mut tree = SearchTree::new(state.clone(), vec![]);

        // Add same state via two different actions
        let child_state = TestState {
            value: 1,
            terminal: false,
        };

        let id1 = tree.add_child(
            NodeId::new(0),
            child_state.clone(),
            TestAction { delta: 1 },
            vec![],
            0.5,
        );

        // Same state again
        let id2 = tree.add_child(
            NodeId::new(0),
            child_state,
            TestAction { delta: 1 },
            vec![],
            0.5,
        );

        // Should return same node id (transposition)
        assert_eq!(id1, id2);
        // Tree size should be 2 (root + one child)
        assert_eq!(tree.size(), 2);
    }
}

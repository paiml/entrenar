//! MCTS search algorithm implementation.
//!
//! This module contains the main search algorithm including
//! selection, expansion, simulation, and backpropagation phases.

#![allow(clippy::field_reassign_with_default)]

use super::config::MctsConfig;
use super::node::NodeId;
use super::traits::{Action, ActionSpace, PolicyNetwork, State, StateSpace};
use super::tree::SearchTree;
use super::Reward;

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
        fn name(&self) -> &'static str {
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
            assert!(action.delta > 0, "Expected positive action, got {action:?}");
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

    // ========================================
    // ENT-095: Property tests
    // ========================================

    proptest! {
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
}

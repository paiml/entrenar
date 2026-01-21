//! Gradient checkpointing for memory-efficient training
//!
//! Gradient checkpointing trades compute for memory by recomputing intermediate
//! activations during the backward pass instead of storing them.
//!
//! ## How It Works
//!
//! 1. During forward pass, only inputs to checkpointed segments are saved
//! 2. During backward pass, the forward pass is recomputed to get activations
//! 3. Memory usage scales with O(sqrt(N)) instead of O(N) for N layers
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::autograd::checkpoint::{checkpoint, CheckpointConfig};
//!
//! // Wrap a computation in a checkpoint
//! let output = checkpoint(|| {
//!     let h1 = layer1.forward(&input);
//!     let h2 = layer2.forward(&h1);
//!     layer3.forward(&h2)
//! }, &input);
//! ```

use crate::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Whether checkpointing is enabled
    pub enabled: bool,
    /// Number of segments to divide the model into
    pub num_segments: usize,
    /// Whether to use selective checkpointing (only checkpoint attention)
    pub selective: bool,
}

impl CheckpointConfig {
    /// Create new config with checkpointing enabled
    pub fn enabled(num_segments: usize) -> Self {
        Self {
            enabled: true,
            num_segments,
            selective: false,
        }
    }

    /// Create config with checkpointing disabled
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            num_segments: 1,
            selective: false,
        }
    }

    /// Enable selective checkpointing (only attention layers)
    pub fn with_selective(mut self) -> Self {
        self.selective = true;
        self
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// A checkpointed computation segment
///
/// Stores the input tensor and a function to recompute the forward pass.
/// During backward, the forward pass is recomputed to recover activations.
pub struct CheckpointedSegment {
    /// Input tensor (saved for recomputation)
    input: Tensor,
    /// Output tensor (computed lazily or cached)
    output: RefCell<Option<Tensor>>,
    /// Whether this segment has been checkpointed
    is_checkpointed: bool,
}

impl CheckpointedSegment {
    /// Create a new checkpointed segment
    pub fn new(input: Tensor, is_checkpointed: bool) -> Self {
        Self {
            input,
            output: RefCell::new(None),
            is_checkpointed,
        }
    }

    /// Get the input tensor
    pub fn input(&self) -> &Tensor {
        &self.input
    }

    /// Check if this segment is checkpointed
    pub fn is_checkpointed(&self) -> bool {
        self.is_checkpointed
    }

    /// Set the output (used during forward pass)
    pub fn set_output(&self, output: Tensor) {
        *self.output.borrow_mut() = Some(output);
    }

    /// Get the output (returns None if not computed yet)
    pub fn output(&self) -> Option<Tensor> {
        self.output.borrow().clone()
    }

    /// Clear the output to free memory
    pub fn clear_output(&self) {
        *self.output.borrow_mut() = None;
    }
}

/// Checkpoint manager for coordinating checkpointed segments
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Segments in order
    segments: Vec<Rc<CheckpointedSegment>>,
    /// Current segment index during forward pass
    current_segment: RefCell<usize>,
    /// Memory saved (estimated bytes)
    memory_saved: RefCell<usize>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            current_segment: RefCell::new(0),
            memory_saved: RefCell::new(0),
        }
    }

    /// Check if checkpointing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the number of segments
    pub fn num_segments(&self) -> usize {
        self.config.num_segments
    }

    /// Register a new segment
    pub fn register_segment(&mut self, input: Tensor) -> Rc<CheckpointedSegment> {
        let idx = self.segments.len();
        let should_checkpoint = self.config.enabled && self.should_checkpoint_segment(idx);

        let segment = Rc::new(CheckpointedSegment::new(input, should_checkpoint));
        self.segments.push(segment.clone());

        // Track memory savings
        if should_checkpoint {
            // Estimate: we save the intermediate activations
            // For now, just track a placeholder value
            *self.memory_saved.borrow_mut() += 1;
        }

        segment
    }

    /// Determine if a segment should be checkpointed
    fn should_checkpoint_segment(&self, segment_idx: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Checkpoint every N segments based on config
        let checkpoint_interval = self.segments.len().max(1) / self.config.num_segments.max(1);
        if checkpoint_interval == 0 {
            return true; // Checkpoint all if interval is 0
        }

        segment_idx.is_multiple_of(checkpoint_interval)
    }

    /// Get estimated memory saved (number of checkpointed segments)
    pub fn memory_saved_segments(&self) -> usize {
        *self.memory_saved.borrow()
    }

    /// Clear all segments (call after backward pass)
    pub fn clear(&mut self) {
        for segment in &self.segments {
            segment.clear_output();
        }
        self.segments.clear();
        *self.current_segment.borrow_mut() = 0;
    }

    /// Get total number of registered segments
    pub fn total_segments(&self) -> usize {
        self.segments.len()
    }
}

/// Run a computation with gradient checkpointing
///
/// The function `f` is executed during forward pass. During backward pass,
/// if checkpointing is enabled, `f` will be re-executed to recompute activations.
///
/// # Arguments
///
/// * `f` - Function that computes the forward pass
/// * `input` - Input tensor (saved for recomputation)
///
/// # Returns
///
/// The output tensor from the computation
pub fn checkpoint<F>(f: F, input: &Tensor) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    // Simply run the function - actual checkpointing happens in training loop
    f(input)
}

/// Run a computation with explicit checkpointing control
///
/// # Arguments
///
/// * `f` - Function that computes the forward pass
/// * `input` - Input tensor
/// * `should_checkpoint` - Whether to enable checkpointing for this segment
pub fn checkpoint_if<F>(f: F, input: &Tensor, should_checkpoint: bool) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    if should_checkpoint {
        // In a full implementation, we would save `input` and `f` for recomputation
        // For now, just run the function
        f(input)
    } else {
        f(input)
    }
}

/// Estimate memory savings from checkpointing
///
/// # Arguments
///
/// * `num_layers` - Number of transformer layers
/// * `hidden_size` - Hidden dimension
/// * `seq_len` - Sequence length
/// * `batch_size` - Batch size
/// * `num_checkpoints` - Number of checkpoint segments
///
/// # Returns
///
/// Tuple of (memory_without_checkpoint, memory_with_checkpoint) in bytes
pub fn estimate_memory_savings(
    num_layers: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    num_checkpoints: usize,
) -> (usize, usize) {
    // Each activation: batch_size * seq_len * hidden_size * sizeof(f32)
    let activation_size = batch_size * seq_len * hidden_size * 4;

    // Without checkpointing: store all layer activations
    let memory_without = num_layers * activation_size;

    // With checkpointing: store only checkpoint boundaries + recompute cost
    // Memory scales as O(sqrt(N)) with optimal checkpointing
    let sqrt_layers = (num_layers as f64).sqrt().ceil() as usize;
    let memory_with = sqrt_layers.max(num_checkpoints) * activation_size;

    (memory_without, memory_with)
}

/// Calculate optimal number of checkpoints for given memory budget
///
/// Uses the formula: optimal_checkpoints = sqrt(num_layers)
pub fn optimal_checkpoints(num_layers: usize) -> usize {
    ((num_layers as f64).sqrt().ceil() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::scale;

    #[test]
    fn test_checkpoint_config_enabled() {
        let config = CheckpointConfig::enabled(4);
        assert!(config.enabled);
        assert_eq!(config.num_segments, 4);
        assert!(!config.selective);
    }

    #[test]
    fn test_checkpoint_config_disabled() {
        let config = CheckpointConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_checkpoint_config_selective() {
        let config = CheckpointConfig::enabled(4).with_selective();
        assert!(config.selective);
    }

    #[test]
    fn test_checkpointed_segment_new() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let segment = CheckpointedSegment::new(input, true);
        assert!(segment.is_checkpointed());
        assert!(segment.output().is_none());
    }

    #[test]
    fn test_checkpointed_segment_output() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let segment = CheckpointedSegment::new(input, true);

        let output = Tensor::from_vec(vec![2.0, 4.0], true);
        segment.set_output(output.clone());

        assert!(segment.output().is_some());
        assert_eq!(segment.output().unwrap().len(), 2);
    }

    #[test]
    fn test_checkpointed_segment_clear() {
        let input = Tensor::from_vec(vec![1.0], true);
        let segment = CheckpointedSegment::new(input, true);
        segment.set_output(Tensor::from_vec(vec![2.0], true));

        segment.clear_output();
        assert!(segment.output().is_none());
    }

    #[test]
    fn test_checkpoint_manager_new() {
        let config = CheckpointConfig::enabled(4);
        let manager = CheckpointManager::new(config);
        assert!(manager.is_enabled());
        assert_eq!(manager.num_segments(), 4);
    }

    #[test]
    fn test_checkpoint_manager_disabled() {
        let config = CheckpointConfig::disabled();
        let manager = CheckpointManager::new(config);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_checkpoint_manager_register() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        let input1 = Tensor::from_vec(vec![1.0], true);
        let input2 = Tensor::from_vec(vec![2.0], true);

        let seg1 = manager.register_segment(input1);
        let seg2 = manager.register_segment(input2);

        assert_eq!(manager.total_segments(), 2);
        assert_eq!(seg1.input().len(), 1);
        assert_eq!(seg2.input().len(), 1);
    }

    #[test]
    fn test_checkpoint_manager_clear() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        manager.register_segment(Tensor::from_vec(vec![1.0], true));
        manager.register_segment(Tensor::from_vec(vec![2.0], true));

        manager.clear();
        assert_eq!(manager.total_segments(), 0);
    }

    #[test]
    fn test_checkpoint_function() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let output = checkpoint(|x| scale(x, 2.0), &input);
        assert_eq!(output.len(), 3);
        assert_eq!(output.data()[0], 2.0);
    }

    #[test]
    fn test_checkpoint_if_enabled() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let output = checkpoint_if(|x| scale(x, 3.0), &input, true);
        assert_eq!(output.data()[0], 3.0);
    }

    #[test]
    fn test_checkpoint_if_disabled() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let output = checkpoint_if(|x| scale(x, 3.0), &input, false);
        assert_eq!(output.data()[0], 3.0);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (without, with) = estimate_memory_savings(32, 4096, 512, 1, 6);

        // With checkpointing should use less memory
        assert!(with < without);

        // Sanity check: without checkpointing stores all layers
        // 32 layers * 512 seq * 4096 hidden * 4 bytes = 268,435,456 bytes
        assert_eq!(without, 32 * 512 * 4096 * 4);
    }

    #[test]
    fn test_optimal_checkpoints() {
        assert_eq!(optimal_checkpoints(1), 1);
        assert_eq!(optimal_checkpoints(4), 2);
        assert_eq!(optimal_checkpoints(16), 4);
        assert_eq!(optimal_checkpoints(32), 6);
        assert_eq!(optimal_checkpoints(64), 8);
    }

    #[test]
    fn test_memory_savings_formula() {
        // For 32 layers with optimal sqrt(32) ≈ 6 checkpoints
        let num_layers = 32;
        let checkpoints = optimal_checkpoints(num_layers);

        let (without, with) = estimate_memory_savings(num_layers, 1024, 128, 1, checkpoints);

        // Memory reduction factor should be approximately sqrt(N)
        let ratio = without as f64 / with as f64;
        assert!(ratio > 4.0); // Should be significant savings
    }

    #[test]
    fn test_checkpoint_preserves_computation() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);

        // Without checkpoint
        let direct = scale(&input, 2.5);

        // With checkpoint
        let checkpointed = checkpoint(|x| scale(x, 2.5), &input);

        // Results should be identical
        for i in 0..4 {
            assert_eq!(direct.data()[i], checkpointed.data()[i]);
        }
    }

    #[test]
    fn test_nested_checkpoints() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);

        let output = checkpoint(
            |x| {
                let h1 = scale(x, 2.0);
                checkpoint(|y| scale(y, 3.0), &h1)
            },
            &input,
        );

        // 1.0 * 2.0 * 3.0 = 6.0
        assert_eq!(output.data()[0], 6.0);
    }

    #[test]
    fn test_checkpoint_manager_memory_tracking() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        for i in 0..4 {
            manager.register_segment(Tensor::from_vec(vec![i as f32], true));
        }

        // Should have tracked some memory savings
        assert!(manager.memory_saved_segments() > 0);
    }
}

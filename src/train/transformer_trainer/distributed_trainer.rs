//! Distributed CUDA trainer for data-parallel pretraining.
//!
//! Wraps `CudaTransformerTrainer` with per-block gradient accumulation
//! and AllReduce communication for multi-GPU / multi-node DDP.
//!
//! # Architecture
//!
//! ```text
//! DistributedCudaTrainer
//! ├── trainer: CudaTransformerTrainer     (local GPU training)
//! ├── comm: DistributedComm               (local channels or TCP)
//! ├── block_grad_accum: PerBlockGradientAccumulator
//! └── dist_config: DistributedTrainConfig
//! ```
//!
//! # Training Step (DDP)
//!
//! 1. Each worker runs forward on its data shard
//! 2. Backward: store per-block gradients in accum buffers
//! 3. Per-block AllReduce (reverse order, overlapping comm+compute):
//!    - AllReduce block[i] gradients across workers
//!    - Optimizer step for block[i] with averaged gradients
//! 4. AllReduce + optimizer for LM head, final norm, embedding
//!
//! # Contract
//!
//! C-DDP-001: After AllReduce + optimizer step, all workers hold identical weights.

#[cfg(feature = "cuda")]
use std::sync::mpsc;

#[cfg(feature = "cuda")]
use super::config::DistributedTrainConfig;
#[cfg(feature = "cuda")]
use super::grad_accumulator::PerBlockGradientAccumulator;

/// Communication backend for distributed training.
#[cfg(feature = "cuda")]
pub enum DistributedComm {
    /// Single-machine multi-GPU via crossbeam channels.
    ///
    /// Each worker has a send/recv pair for each other worker.
    /// AllReduce is done by sending gradients to rank 0,
    /// averaging, and broadcasting back.
    Local {
        /// Send gradient to coordinator
        tx: mpsc::Sender<GradientMessage>,
        /// Receive averaged gradient from coordinator
        rx: mpsc::Receiver<GradientMessage>,
    },
    /// Multi-node via TCP using the existing WorkerClient/GradientServer.
    Remote {
        /// TCP client for gradient exchange
        client: crate::finetune::WorkerClient,
    },
}

/// Message types for local (channel-based) gradient exchange.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum GradientMessage {
    /// Per-block gradient from a worker
    BlockGradient {
        block_idx: usize,
        gradients: Vec<f32>,
        component_sizes: Vec<u32>,
    },
    /// Averaged per-block gradient from coordinator
    AveragedBlockGradient {
        block_idx: usize,
        gradients: Vec<f32>,
        component_sizes: Vec<u32>,
    },
    /// Non-block gradient (LM head, final norm, embedding)
    NonBlockGradient {
        component: u8,
        gradients: Vec<f32>,
    },
    /// Averaged non-block gradient
    AveragedNonBlockGradient {
        component: u8,
        gradients: Vec<f32>,
    },
    /// Synchronization barrier
    Barrier,
}

/// Distributed CUDA trainer for data-parallel pretraining.
///
/// Wraps a single-GPU `CudaTransformerTrainer` with communication
/// and gradient averaging logic. The actual CUDA operations remain
/// in the underlying trainer — this layer only handles:
///
/// 1. Downloading per-block gradients from GPU to CPU accumulation buffers
/// 2. AllReducing gradients across workers
/// 3. Uploading averaged gradients back to GPU for optimizer step
///
/// # Safety
///
/// C-STREAMSYNC-001 applies: stream.synchronize() before all D2H transfers
/// is handled by the underlying CudaTransformerTrainer.
#[cfg(feature = "cuda")]
pub struct DistributedCudaTrainer {
    /// Underlying single-GPU trainer
    trainer: super::cuda_trainer::CudaTransformerTrainer,
    /// Communication backend
    comm: DistributedComm,
    /// Per-block gradient accumulation buffers (CPU-side)
    block_grad_accum: PerBlockGradientAccumulator,
    /// Distributed configuration
    dist_config: DistributedTrainConfig,
    /// Current training step
    step: usize,
}

#[cfg(feature = "cuda")]
impl DistributedCudaTrainer {
    /// Create a new distributed trainer.
    ///
    /// # Arguments
    /// * `trainer` - Pre-initialized single-GPU trainer
    /// * `comm` - Communication backend (local channels or TCP)
    /// * `dist_config` - Distributed training configuration
    /// * `block_sizes` - Per-block gradient component sizes (from model architecture)
    /// * `vocab_size` - For LM head and embedding gradient buffers
    /// * `hidden_size` - For final norm gradient buffer
    /// * `num_blocks` - Number of transformer layers
    pub fn new(
        trainer: super::cuda_trainer::CudaTransformerTrainer,
        comm: DistributedComm,
        dist_config: DistributedTrainConfig,
        block_sizes: [usize; super::grad_accumulator::BLOCK_GRAD_COMPONENTS],
        vocab_size: usize,
        hidden_size: usize,
        num_blocks: usize,
    ) -> Self {
        let block_grad_accum =
            PerBlockGradientAccumulator::new(num_blocks, block_sizes, vocab_size, hidden_size);

        Self {
            trainer,
            comm,
            block_grad_accum,
            dist_config,
            step: 0,
        }
    }

    /// Get the distributed configuration.
    pub fn dist_config(&self) -> &DistributedTrainConfig {
        &self.dist_config
    }

    /// Get the current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get a reference to the underlying trainer.
    pub fn trainer(&self) -> &super::cuda_trainer::CudaTransformerTrainer {
        &self.trainer
    }

    /// Get a mutable reference to the underlying trainer.
    pub fn trainer_mut(&mut self) -> &mut super::cuda_trainer::CudaTransformerTrainer {
        &mut self.trainer
    }

    /// Get a reference to the gradient accumulator.
    pub fn grad_accum(&self) -> &PerBlockGradientAccumulator {
        &self.block_grad_accum
    }

    /// Get a mutable reference to the gradient accumulator.
    pub fn grad_accum_mut(&mut self) -> &mut PerBlockGradientAccumulator {
        &mut self.block_grad_accum
    }

    /// Zero all gradient accumulation buffers (call at start of each step).
    pub fn zero_grad_accum(&mut self) {
        self.block_grad_accum.zero_all();
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Check if this worker is the coordinator (rank 0).
    pub fn is_coordinator(&self) -> bool {
        self.dist_config.rank == 0
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.dist_config.world_size
    }

    /// Get rank.
    pub fn rank(&self) -> usize {
        self.dist_config.rank
    }
}

/// Create a local communication pair for single-machine multi-GPU training.
///
/// Returns (coordinator_comm, worker_comms) where worker_comms[i] is for worker i.
/// The coordinator aggregates gradients and broadcasts averages.
#[cfg(feature = "cuda")]
pub fn create_local_comm_pair() -> (
    (mpsc::Sender<GradientMessage>, mpsc::Receiver<GradientMessage>),
    (mpsc::Sender<GradientMessage>, mpsc::Receiver<GradientMessage>),
) {
    let (tx_to_coord, rx_at_coord) = mpsc::channel();
    let (tx_to_worker, rx_at_worker) = mpsc::channel();
    ((tx_to_worker, rx_at_coord), (tx_to_coord, rx_at_worker))
}

#[cfg(test)]
mod tests {
    // DistributedCudaTrainer requires CUDA feature, so we test only
    // the non-CUDA parts here. Full integration tests require CUDA hardware.

    #[test]
    fn test_module_compiles() {
        // Smoke test: this module exists and compiles
        assert!(true);
    }
}

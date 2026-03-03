//! TCP worker client for distributed training (worker side)
//!
//! The `WorkerClient` runs on each worker node and:
//! 1. Connects to the coordinator
//! 2. Receives shard assignments per training step
//! 3. Computes forward/backward locally
//! 4. Sends gradients to coordinator
//! 5. Receives averaged gradients and applies optimizer step
//!
//! # Contract: F-DP-004 (Backend Fallback)
//!
//! The worker uses `forward_hidden_dispatch()` which falls back:
//! CUDA → wgpu → CPU. Training always proceeds regardless of GPU availability.

use super::distributed::{DistributedConfig, WireMessage};
use super::gradient_server::{read_wire_message, send_wire_message};
use std::net::TcpStream;

/// Worker client that connects to the coordinator.
pub struct WorkerClient {
    config: DistributedConfig,
    stream: TcpStream,
    worker_id: u32,
    total_workers: u32,
}

/// Shard assignment received from coordinator.
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    pub step: u64,
    pub shard_start: usize,
    pub shard_end: usize,
}

/// Averaged gradient received from coordinator after AllReduce.
#[derive(Debug, Clone)]
pub struct AveragedResult {
    pub step: u64,
    pub gradients: Vec<f32>,
    pub global_loss: f32,
}

impl WorkerClient {
    /// Connect to the coordinator and complete the join handshake.
    ///
    /// # Arguments
    /// * `config` - Worker configuration with coordinator address
    /// * `gpu_count` - Number of GPUs this worker has
    /// * `backend` - Backend name (e.g., "wgpu", "cuda", "cpu")
    ///
    /// # Errors
    /// Returns error if connection or handshake fails.
    pub fn connect(
        config: DistributedConfig,
        gpu_count: u32,
        backend: &str,
    ) -> Result<Self, String> {
        let coord_addr = config
            .coordinator_addr
            .ok_or_else(|| "worker config must have coordinator_addr".to_string())?;

        eprintln!("[worker {}] Connecting to coordinator at {coord_addr}...", config.node_id);

        let stream = TcpStream::connect(coord_addr)
            .map_err(|e| format!("failed to connect to {coord_addr}: {e}"))?;

        // Send JoinRequest
        let join = WireMessage::JoinRequest {
            node_id: config.node_id.clone(),
            gpu_count,
            backend: backend.to_string(),
        };
        send_wire_message(&stream, &join)?;

        // Read JoinAccepted
        let response = read_wire_message(&stream)?;
        match response {
            WireMessage::JoinAccepted {
                worker_id,
                total_workers,
            } => {
                eprintln!(
                    "[worker {}] Joined as worker {worker_id}/{total_workers}",
                    config.node_id
                );
                Ok(Self {
                    config,
                    stream,
                    worker_id,
                    total_workers,
                })
            }
            other => Err(format!("expected JoinAccepted, got {other:?}")),
        }
    }

    /// Receive shard assignment for the next training step.
    ///
    /// Returns `None` if the coordinator sends a Shutdown message.
    ///
    /// # Errors
    /// Returns error on communication failure.
    pub fn receive_shard(&self) -> Result<Option<ShardAssignment>, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::ShardAssignment {
                step,
                shard_start,
                shard_end,
            } => Ok(Some(ShardAssignment {
                step,
                shard_start,
                shard_end,
            })),
            WireMessage::Shutdown => {
                eprintln!(
                    "[worker {}] Received shutdown from coordinator",
                    self.config.node_id
                );
                Ok(None)
            }
            other => Err(format!("expected ShardAssignment or Shutdown, got {other:?}")),
        }
    }

    /// Send computed gradients to the coordinator.
    ///
    /// # Arguments
    /// * `step` - Training step number
    /// * `gradients` - Gradient vector (flattened LoRA params + classifier head)
    /// * `loss` - Average loss for this shard
    /// * `correct` - Number of correct predictions
    /// * `total` - Total samples in shard
    ///
    /// # Errors
    /// Returns error on send failure.
    pub fn send_gradients(
        &self,
        step: u64,
        gradients: Vec<f32>,
        loss: f32,
        correct: usize,
        total: usize,
    ) -> Result<(), String> {
        let msg = WireMessage::GradientPayload {
            step,
            worker_id: self.worker_id,
            gradients,
            loss,
            correct,
            total,
        };
        send_wire_message(&self.stream, &msg)
    }

    /// Receive averaged gradients from coordinator after AllReduce.
    ///
    /// # Errors
    /// Returns error on communication failure.
    pub fn receive_averaged(&self) -> Result<AveragedResult, String> {
        let msg = read_wire_message(&self.stream)?;
        match msg {
            WireMessage::AveragedGradient {
                step,
                gradients,
                global_loss,
            } => Ok(AveragedResult {
                step,
                gradients,
                global_loss,
            }),
            WireMessage::Shutdown => Err("shutdown during AllReduce".to_string()),
            other => Err(format!("expected AveragedGradient, got {other:?}")),
        }
    }

    /// This worker's assigned ID
    #[must_use]
    pub fn worker_id(&self) -> u32 {
        self.worker_id
    }

    /// Total number of workers in the cluster
    #[must_use]
    pub fn total_workers(&self) -> u32 {
        self.total_workers
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use super::super::distributed::DistributedConfig;
    use super::super::gradient_server::GradientServer;
    use std::thread;

    #[test]
    fn test_worker_connect_and_join() {
        let server_config = DistributedConfig::coordinator("127.0.0.1:0".parse().unwrap(), 1);
        let mut server = GradientServer::bind(server_config).unwrap();
        let addr = server.local_addr();

        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cpu").unwrap();
            assert_eq!(client.worker_id(), 0);
            assert_eq!(client.total_workers(), 1);
            client
        });

        server.wait_for_workers().unwrap();
        let _client = handle.join().unwrap();
    }

    #[test]
    fn test_worker_full_training_step() {
        let server_config = DistributedConfig::coordinator("127.0.0.1:0".parse().unwrap(), 1);
        let mut server = GradientServer::bind(server_config).unwrap();
        let addr = server.local_addr();

        let handle = thread::spawn(move || {
            let worker_config = DistributedConfig::worker(addr);
            let client = WorkerClient::connect(worker_config, 1, "cpu").unwrap();

            // Receive shard
            let shard = client.receive_shard().unwrap().expect("should get shard");
            assert_eq!(shard.step, 0);
            assert_eq!(shard.shard_start, 0);
            assert_eq!(shard.shard_end, 50);

            // Send gradients
            client
                .send_gradients(0, vec![1.0, 2.0, 3.0], 0.5, 48, 50)
                .unwrap();

            // Receive averaged
            let avg = client.receive_averaged().unwrap();
            assert_eq!(avg.step, 0);
            assert_eq!(avg.gradients, vec![1.0, 2.0, 3.0]); // Single worker, no averaging
            assert!((avg.global_loss - 0.5).abs() < 1e-5);

            client
        });

        server.wait_for_workers().unwrap();
        server.set_total_samples(50);
        server.send_shard_assignments(0).unwrap();
        let result = server.collect_and_reduce(0).unwrap();
        server.broadcast_averaged(0, &result).unwrap();

        let _client = handle.join().unwrap();
    }
}

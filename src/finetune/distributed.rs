//! Distributed training configuration and protocol (SPEC-DIST-2026-001 Phase 2)
//!
//! Defines the coordination protocol for multi-node heterogeneous training
//! across CUDA and wgpu backends over TCP.
//!
//! # Architecture
//!
//! ```text
//! Coordinator (intel:9000)
//!   ├── Worker 0: intel:gpu0 (wgpu)
//!   ├── Worker 1: intel:gpu1 (wgpu)
//!   └── Worker 2: lambda:gpu0 (CUDA)
//! ```
//!
//! # Protocol
//!
//! 1. Workers connect to coordinator via TCP
//! 2. Coordinator broadcasts model config + initial weights
//! 3. Per step: coordinator sends shard assignment, workers compute gradients,
//!    coordinator AllReduces, broadcasts averaged gradients
//! 4. Workers detect coordinator failure via heartbeat timeout
//!
//! # Contract: F-DP-001 (Weight Consistency)
//!
//! After AllReduce + optimizer step, all workers hold identical LoRA weights.

use std::fmt;
use std::net::SocketAddr;

/// Role of a node in distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Coordinates training: accepts workers, shards data, AllReduces gradients
    Coordinator,
    /// Computes forward/backward on assigned shard, sends gradients to coordinator
    Worker,
}

impl fmt::Display for NodeRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinator => write!(f, "coordinator"),
            Self::Worker => write!(f, "worker"),
        }
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Role of this node
    pub role: NodeRole,
    /// Address to bind (coordinator) or connect to (worker)
    pub bind_addr: SocketAddr,
    /// Coordinator address (workers only; coordinators use bind_addr)
    pub coordinator_addr: Option<SocketAddr>,
    /// Expected number of workers (coordinator only, for barrier)
    pub expect_workers: usize,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Heartbeat timeout in milliseconds (detect failure after this)
    pub heartbeat_timeout_ms: u64,
    /// Node identifier (auto-assigned from hostname + pid)
    pub node_id: String,
}

impl DistributedConfig {
    /// Create a coordinator config.
    ///
    /// # Arguments
    /// * `bind_addr` - Address to listen on (e.g., `0.0.0.0:9000`)
    /// * `expect_workers` - Total worker count (including coordinator's own GPUs)
    #[must_use]
    pub fn coordinator(bind_addr: SocketAddr, expect_workers: usize) -> Self {
        Self {
            role: NodeRole::Coordinator,
            bind_addr,
            coordinator_addr: None,
            expect_workers,
            heartbeat_interval_ms: 5000,
            heartbeat_timeout_ms: 30000,
            node_id: Self::default_node_id(),
        }
    }

    /// Create a worker config.
    ///
    /// # Arguments
    /// * `coordinator_addr` - Address of the coordinator (e.g., `intel:9000`)
    #[must_use]
    pub fn worker(coordinator_addr: SocketAddr) -> Self {
        Self {
            role: NodeRole::Worker,
            bind_addr: "0.0.0.0:0".parse().expect("valid addr"),
            coordinator_addr: Some(coordinator_addr),
            expect_workers: 0,
            heartbeat_interval_ms: 5000,
            heartbeat_timeout_ms: 30000,
            node_id: Self::default_node_id(),
        }
    }

    /// Check if this node is a coordinator
    #[must_use]
    pub fn is_coordinator(&self) -> bool {
        self.role == NodeRole::Coordinator
    }

    fn default_node_id() -> String {
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let pid = std::process::id();
        format!("{hostname}-{pid}")
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self::coordinator("0.0.0.0:9000".parse().expect("valid addr"), 1)
    }
}

// ─── Wire Protocol ───────────────────────────────────────────────────────────

/// Messages sent between coordinator and workers over TCP.
///
/// Each message is length-prefixed: `[u32 big-endian length][payload bytes]`.
/// Payload is bincode-serialized `WireMessage`.
#[derive(Debug, Clone)]
pub enum WireMessage {
    /// Worker → Coordinator: request to join training
    JoinRequest {
        node_id: String,
        gpu_count: u32,
        backend: String,
    },
    /// Coordinator → Worker: join accepted with assigned worker ID
    JoinAccepted {
        worker_id: u32,
        total_workers: u32,
    },
    /// Coordinator → Worker: here is your shard for this step
    ShardAssignment {
        step: u64,
        shard_start: usize,
        shard_end: usize,
    },
    /// Worker → Coordinator: gradient data for this step
    GradientPayload {
        step: u64,
        worker_id: u32,
        /// Serialized f32 gradient vector (LoRA params + classifier head)
        gradients: Vec<f32>,
        loss: f32,
        correct: usize,
        total: usize,
    },
    /// Coordinator → Worker: averaged gradient for optimizer step
    AveragedGradient {
        step: u64,
        gradients: Vec<f32>,
        global_loss: f32,
    },
    /// Bidirectional: heartbeat ping/pong
    Heartbeat {
        node_id: String,
        timestamp_ms: u64,
    },
    /// Coordinator → Worker: training complete, shut down
    Shutdown,
}

impl WireMessage {
    /// Serialize this message to bytes (length-prefixed).
    ///
    /// Format: `[4 bytes big-endian length][payload]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload = self.serialize_payload();
        let len = payload.len() as u32;
        let mut buf = Vec::with_capacity(4 + payload.len());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&payload);
        buf
    }

    /// Deserialize from a complete payload (without length prefix).
    ///
    /// # Errors
    /// Returns error if payload is malformed.
    pub fn from_payload(payload: &[u8]) -> Result<Self, String> {
        Self::deserialize_payload(payload)
    }

    // Simple binary serialization (avoiding serde dependency for wire protocol)
    fn serialize_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            Self::JoinRequest {
                node_id,
                gpu_count,
                backend,
            } => {
                buf.push(0x01);
                write_string(&mut buf, node_id);
                buf.extend_from_slice(&gpu_count.to_le_bytes());
                write_string(&mut buf, backend);
            }
            Self::JoinAccepted {
                worker_id,
                total_workers,
            } => {
                buf.push(0x02);
                buf.extend_from_slice(&worker_id.to_le_bytes());
                buf.extend_from_slice(&total_workers.to_le_bytes());
            }
            Self::ShardAssignment {
                step,
                shard_start,
                shard_end,
            } => {
                buf.push(0x03);
                buf.extend_from_slice(&step.to_le_bytes());
                buf.extend_from_slice(&(*shard_start as u64).to_le_bytes());
                buf.extend_from_slice(&(*shard_end as u64).to_le_bytes());
            }
            Self::GradientPayload {
                step,
                worker_id,
                gradients,
                loss,
                correct,
                total,
            } => {
                buf.push(0x04);
                buf.extend_from_slice(&step.to_le_bytes());
                buf.extend_from_slice(&worker_id.to_le_bytes());
                buf.extend_from_slice(&(gradients.len() as u64).to_le_bytes());
                for &g in gradients {
                    buf.extend_from_slice(&g.to_le_bytes());
                }
                buf.extend_from_slice(&loss.to_le_bytes());
                buf.extend_from_slice(&(*correct as u64).to_le_bytes());
                buf.extend_from_slice(&(*total as u64).to_le_bytes());
            }
            Self::AveragedGradient {
                step,
                gradients,
                global_loss,
            } => {
                buf.push(0x05);
                buf.extend_from_slice(&step.to_le_bytes());
                buf.extend_from_slice(&(gradients.len() as u64).to_le_bytes());
                for &g in gradients {
                    buf.extend_from_slice(&g.to_le_bytes());
                }
                buf.extend_from_slice(&global_loss.to_le_bytes());
            }
            Self::Heartbeat {
                node_id,
                timestamp_ms,
            } => {
                buf.push(0x06);
                write_string(&mut buf, node_id);
                buf.extend_from_slice(&timestamp_ms.to_le_bytes());
            }
            Self::Shutdown => {
                buf.push(0x07);
            }
        }
        buf
    }

    fn deserialize_payload(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Err("empty payload".to_string());
        }
        let tag = data[0];
        let rest = &data[1..];
        match tag {
            0x01 => decode_join_request(rest),
            0x02 => decode_join_accepted(rest),
            0x03 => decode_shard_assignment(rest),
            0x04 => decode_gradient_payload(rest),
            0x05 => decode_averaged_gradient(rest),
            0x06 => decode_heartbeat(rest),
            0x07 => Ok(Self::Shutdown),
            other => Err(format!("unknown message tag: 0x{other:02x}")),
        }
    }
}

fn decode_join_request(rest: &[u8]) -> Result<WireMessage, String> {
    let (node_id, rest) = read_string(rest)?;
    if rest.len() < 4 {
        return Err("truncated JoinRequest".to_string());
    }
    let gpu_count = u32::from_le_bytes(rest[..4].try_into().expect("4 bytes"));
    let (backend, _) = read_string(&rest[4..])?;
    Ok(WireMessage::JoinRequest { node_id, gpu_count, backend })
}

fn decode_join_accepted(rest: &[u8]) -> Result<WireMessage, String> {
    if rest.len() < 8 {
        return Err("truncated JoinAccepted".to_string());
    }
    let worker_id = u32::from_le_bytes(rest[..4].try_into().expect("4 bytes"));
    let total_workers = u32::from_le_bytes(rest[4..8].try_into().expect("4 bytes"));
    Ok(WireMessage::JoinAccepted { worker_id, total_workers })
}

fn decode_shard_assignment(rest: &[u8]) -> Result<WireMessage, String> {
    if rest.len() < 24 {
        return Err("truncated ShardAssignment".to_string());
    }
    let step = u64::from_le_bytes(rest[..8].try_into().expect("8 bytes"));
    let shard_start = u64::from_le_bytes(rest[8..16].try_into().expect("8 bytes")) as usize;
    let shard_end = u64::from_le_bytes(rest[16..24].try_into().expect("8 bytes")) as usize;
    Ok(WireMessage::ShardAssignment { step, shard_start, shard_end })
}

fn decode_gradient_payload(rest: &[u8]) -> Result<WireMessage, String> {
    if rest.len() < 20 {
        return Err("truncated GradientPayload header".to_string());
    }
    let step = u64::from_le_bytes(rest[..8].try_into().expect("8 bytes"));
    let worker_id = u32::from_le_bytes(rest[8..12].try_into().expect("4 bytes"));
    let grad_len = u64::from_le_bytes(rest[12..20].try_into().expect("8 bytes")) as usize;
    let grad_bytes = grad_len * 4;
    if rest.len() < 20 + grad_bytes + 4 + 8 + 8 {
        return Err("truncated GradientPayload data".to_string());
    }
    let gradients = read_f32_vec(rest, 20, grad_len);
    let tail = &rest[20 + grad_bytes..];
    let loss = f32::from_le_bytes(tail[..4].try_into().expect("4 bytes"));
    let correct = u64::from_le_bytes(tail[4..12].try_into().expect("8 bytes")) as usize;
    let total = u64::from_le_bytes(tail[12..20].try_into().expect("8 bytes")) as usize;
    Ok(WireMessage::GradientPayload { step, worker_id, gradients, loss, correct, total })
}

fn decode_averaged_gradient(rest: &[u8]) -> Result<WireMessage, String> {
    if rest.len() < 16 {
        return Err("truncated AveragedGradient header".to_string());
    }
    let step = u64::from_le_bytes(rest[..8].try_into().expect("8 bytes"));
    let grad_len = u64::from_le_bytes(rest[8..16].try_into().expect("8 bytes")) as usize;
    let grad_bytes = grad_len * 4;
    if rest.len() < 16 + grad_bytes + 4 {
        return Err("truncated AveragedGradient data".to_string());
    }
    let gradients = read_f32_vec(rest, 16, grad_len);
    let global_loss = f32::from_le_bytes(
        rest[16 + grad_bytes..16 + grad_bytes + 4].try_into().expect("4 bytes"),
    );
    Ok(WireMessage::AveragedGradient { step, gradients, global_loss })
}

fn decode_heartbeat(rest: &[u8]) -> Result<WireMessage, String> {
    let (node_id, rest) = read_string(rest)?;
    if rest.len() < 8 {
        return Err("truncated Heartbeat".to_string());
    }
    let timestamp_ms = u64::from_le_bytes(rest[..8].try_into().expect("8 bytes"));
    Ok(WireMessage::Heartbeat { node_id, timestamp_ms })
}

fn read_f32_vec(data: &[u8], offset: usize, count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * 4;
        let val = f32::from_le_bytes(data[start..start + 4].try_into().expect("4 bytes"));
        result.push(val);
    }
    result
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_string(data: &[u8]) -> Result<(String, &[u8]), String> {
    if data.len() < 4 {
        return Err("truncated string length".to_string());
    }
    let len = u32::from_le_bytes(data[..4].try_into().expect("4 bytes")) as usize;
    if data.len() < 4 + len {
        return Err("truncated string data".to_string());
    }
    let s = String::from_utf8(data[4..4 + len].to_vec())
        .map_err(|e| format!("invalid utf8: {e}"))?;
    Ok((s, &data[4 + len..]))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_coordinator_config() {
        let config = DistributedConfig::coordinator("0.0.0.0:9000".parse().unwrap(), 3);
        assert!(config.is_coordinator());
        assert_eq!(config.role, NodeRole::Coordinator);
        assert_eq!(config.expect_workers, 3);
        assert!(config.coordinator_addr.is_none());
    }

    #[test]
    fn test_worker_config() {
        let config = DistributedConfig::worker("192.168.50.100:9000".parse().unwrap());
        assert!(!config.is_coordinator());
        assert_eq!(config.role, NodeRole::Worker);
        assert_eq!(
            config.coordinator_addr,
            Some("192.168.50.100:9000".parse().unwrap())
        );
    }

    #[test]
    fn test_default_config() {
        let config = DistributedConfig::default();
        assert!(config.is_coordinator());
        assert_eq!(config.expect_workers, 1);
    }

    #[test]
    fn test_node_role_display() {
        assert_eq!(NodeRole::Coordinator.to_string(), "coordinator");
        assert_eq!(NodeRole::Worker.to_string(), "worker");
    }

    #[test]
    fn test_node_id_not_empty() {
        let config = DistributedConfig::default();
        assert!(!config.node_id.is_empty());
    }

    // ── Wire protocol round-trip tests ───────────────────────────────────

    #[test]
    fn test_wire_join_request_roundtrip() {
        let msg = WireMessage::JoinRequest {
            node_id: "intel-1234".to_string(),
            gpu_count: 2,
            backend: "wgpu".to_string(),
        };
        let bytes = msg.to_bytes();
        // Skip 4-byte length prefix
        let payload = &bytes[4..];
        let decoded = WireMessage::from_payload(payload).unwrap();
        match decoded {
            WireMessage::JoinRequest {
                node_id,
                gpu_count,
                backend,
            } => {
                assert_eq!(node_id, "intel-1234");
                assert_eq!(gpu_count, 2);
                assert_eq!(backend, "wgpu");
            }
            other => panic!("expected JoinRequest, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_join_accepted_roundtrip() {
        let msg = WireMessage::JoinAccepted {
            worker_id: 1,
            total_workers: 3,
        };
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::JoinAccepted {
                worker_id,
                total_workers,
            } => {
                assert_eq!(worker_id, 1);
                assert_eq!(total_workers, 3);
            }
            other => panic!("expected JoinAccepted, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_shard_assignment_roundtrip() {
        let msg = WireMessage::ShardAssignment {
            step: 42,
            shard_start: 100,
            shard_end: 200,
        };
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::ShardAssignment {
                step,
                shard_start,
                shard_end,
            } => {
                assert_eq!(step, 42);
                assert_eq!(shard_start, 100);
                assert_eq!(shard_end, 200);
            }
            other => panic!("expected ShardAssignment, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_gradient_payload_roundtrip() {
        let grads = vec![1.0f32, 2.0, 3.0, -0.5, 0.0];
        let msg = WireMessage::GradientPayload {
            step: 10,
            worker_id: 2,
            gradients: grads.clone(),
            loss: 0.456,
            correct: 8,
            total: 10,
        };
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::GradientPayload {
                step,
                worker_id,
                gradients,
                loss,
                correct,
                total,
            } => {
                assert_eq!(step, 10);
                assert_eq!(worker_id, 2);
                assert_eq!(gradients, grads);
                assert!((loss - 0.456).abs() < 1e-6);
                assert_eq!(correct, 8);
                assert_eq!(total, 10);
            }
            other => panic!("expected GradientPayload, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_averaged_gradient_roundtrip() {
        let grads = vec![0.5f32, 1.0, 1.5];
        let msg = WireMessage::AveragedGradient {
            step: 5,
            gradients: grads.clone(),
            global_loss: 0.789,
        };
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::AveragedGradient {
                step,
                gradients,
                global_loss,
            } => {
                assert_eq!(step, 5);
                assert_eq!(gradients, grads);
                assert!((global_loss - 0.789).abs() < 1e-6);
            }
            other => panic!("expected AveragedGradient, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_heartbeat_roundtrip() {
        let msg = WireMessage::Heartbeat {
            node_id: "lambda-5678".to_string(),
            timestamp_ms: 1_709_000_000_000,
        };
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::Heartbeat {
                node_id,
                timestamp_ms,
            } => {
                assert_eq!(node_id, "lambda-5678");
                assert_eq!(timestamp_ms, 1_709_000_000_000);
            }
            other => panic!("expected Heartbeat, got {other:?}"),
        }
    }

    #[test]
    fn test_wire_shutdown_roundtrip() {
        let msg = WireMessage::Shutdown;
        let bytes = msg.to_bytes();
        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        assert!(matches!(decoded, WireMessage::Shutdown));
    }

    #[test]
    fn test_wire_empty_payload_error() {
        let result = WireMessage::from_payload(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_wire_unknown_tag_error() {
        let result = WireMessage::from_payload(&[0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn test_wire_large_gradient_roundtrip() {
        // Simulate real LoRA gradient size: ~1.3M params
        let grad_len = 1_378_050;
        let grads: Vec<f32> = (0..grad_len).map(|i| (i as f32) * 0.0001).collect();
        let msg = WireMessage::GradientPayload {
            step: 100,
            worker_id: 0,
            gradients: grads.clone(),
            loss: 0.123,
            correct: 95,
            total: 100,
        };
        let bytes = msg.to_bytes();
        // Verify size: 4 (len prefix) + 1 (tag) + 8 (step) + 4 (worker_id) +
        //   8 (grad_len) + grad_len*4 + 4 (loss) + 8 (correct) + 8 (total)
        let expected_size = 4 + 1 + 8 + 4 + 8 + grad_len * 4 + 4 + 8 + 8;
        assert_eq!(bytes.len(), expected_size);

        let decoded = WireMessage::from_payload(&bytes[4..]).unwrap();
        match decoded {
            WireMessage::GradientPayload {
                gradients, loss, ..
            } => {
                assert_eq!(gradients.len(), grad_len);
                assert!((loss - 0.123).abs() < 1e-6);
            }
            other => panic!("expected GradientPayload, got {other:?}"),
        }
    }
}

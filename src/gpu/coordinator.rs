//! Checkpoint coordination for multi-node adapter training (GPU-SHARE Phase 3, §3.4).
//!
//! The coordinator polls each node's checkpoint directory for metadata,
//! compares val_loss across adapters, maintains a leaderboard, and identifies
//! the best adapter at end of training.
//!
//! Remote nodes are polled via `cat checkpoint_dir/best/metadata.json` over SSH.
//! Local nodes read the file directly.

use super::cluster::{ClusterConfig, NodeConfig, Transport};
use super::placement::PlacementDecision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata written by `save_adapter_checkpoint()` in multi_adapter_pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub adapter_idx: usize,
    pub epoch: usize,
    pub avg_loss: f32,
    #[serde(default)]
    pub val_loss: Option<f32>,
    #[serde(default)]
    pub node_name: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// Status of a single adapter across the cluster.
#[derive(Debug, Clone)]
pub struct AdapterStatus {
    /// Adapter index.
    pub adapter_idx: usize,
    /// Node name where this adapter is running.
    pub node_name: String,
    /// Checkpoint directory on the remote node.
    pub checkpoint_dir: PathBuf,
    /// Latest checkpoint metadata (if available).
    pub latest: Option<CheckpointMetadata>,
}

/// Leaderboard entry ranking adapters by loss.
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub adapter_idx: usize,
    pub node_name: String,
    pub epoch: usize,
    pub loss: f32,
}

/// Coordinator for multi-node training checkpoint polling.
pub struct CheckpointCoordinator {
    /// Adapter statuses indexed by adapter_idx.
    pub adapters: HashMap<usize, AdapterStatus>,
    /// Poll interval in seconds.
    pub poll_interval_secs: u64,
    /// Cluster configuration reference.
    cluster: ClusterConfig,
}

impl CheckpointCoordinator {
    /// Create a new coordinator from placement decisions.
    pub fn new(
        cluster: ClusterConfig,
        placements: &[PlacementDecision],
        checkpoint_dirs: &HashMap<usize, PathBuf>,
        poll_interval_secs: u64,
    ) -> Self {
        let mut adapters = HashMap::new();
        for p in placements {
            let checkpoint_dir = checkpoint_dirs
                .get(&p.adapter_idx)
                .cloned()
                .unwrap_or_else(|| PathBuf::from(format!("checkpoints/adapter-{}", p.adapter_idx)));
            adapters.insert(
                p.adapter_idx,
                AdapterStatus {
                    adapter_idx: p.adapter_idx,
                    node_name: p.node_name.clone(),
                    checkpoint_dir,
                    latest: None,
                },
            );
        }
        Self {
            adapters,
            poll_interval_secs,
            cluster,
        }
    }

    /// Poll all adapters for their latest checkpoint metadata.
    ///
    /// For local nodes, reads the file directly.
    /// For SSH nodes, returns an error (SSH transport via forjar not yet available).
    pub fn poll_all(&mut self) -> Vec<PollResult> {
        let mut results = Vec::new();
        let adapter_list: Vec<(usize, String, PathBuf)> = self
            .adapters
            .values()
            .map(|a| (a.adapter_idx, a.node_name.clone(), a.checkpoint_dir.clone()))
            .collect();

        for (idx, node_name, checkpoint_dir) in adapter_list {
            let result = self.poll_adapter(idx, &node_name, &checkpoint_dir);
            results.push(result);
        }
        results
    }

    fn poll_adapter(
        &mut self,
        adapter_idx: usize,
        node_name: &str,
        checkpoint_dir: &Path,
    ) -> PollResult {
        let node = self.cluster.find_node(node_name);
        let transport = node.map_or(Transport::Local, |n| n.transport);

        let metadata = match transport {
            Transport::Local => read_local_metadata(checkpoint_dir),
            Transport::Ssh => {
                let host = node.map_or("unknown", |n| &n.host);
                let user = node.and_then(|n| n.user.as_deref());
                read_ssh_metadata(host, user, checkpoint_dir)
            }
        };

        match metadata {
            Ok(meta) => {
                if let Some(status) = self.adapters.get_mut(&adapter_idx) {
                    status.latest = Some(meta.clone());
                }
                PollResult::Ok {
                    adapter_idx,
                    metadata: meta,
                }
            }
            Err(e) => PollResult::Error {
                adapter_idx,
                node_name: node_name.to_string(),
                error: e,
            },
        }
    }

    /// Generate leaderboard sorted by loss (ascending).
    pub fn leaderboard(&self) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self
            .adapters
            .values()
            .filter_map(|a| {
                a.latest.as_ref().map(|meta| LeaderboardEntry {
                    rank: 0,
                    adapter_idx: a.adapter_idx,
                    node_name: a.node_name.clone(),
                    epoch: meta.epoch,
                    loss: meta.val_loss.unwrap_or(meta.avg_loss),
                })
            })
            .collect();

        entries.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal));
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }
        entries
    }

    /// Find the best adapter (lowest loss).
    pub fn best_adapter(&self) -> Option<&AdapterStatus> {
        let board = self.leaderboard();
        board
            .first()
            .and_then(|entry| self.adapters.get(&entry.adapter_idx))
    }

    /// Format leaderboard as a human-readable string.
    pub fn format_leaderboard(&self) -> String {
        let board = self.leaderboard();
        if board.is_empty() {
            return "No checkpoints available yet.".to_string();
        }
        let mut out = String::from("Adapter Leaderboard:\n");
        out.push_str("  Rank | Adapter | Node       | Epoch | Loss\n");
        out.push_str("  -----+---------+------------+-------+--------\n");
        for entry in &board {
            out.push_str(&format!(
                "  {:>4} | {:>7} | {:<10} | {:>5} | {:.4}\n",
                entry.rank, entry.adapter_idx, entry.node_name, entry.epoch, entry.loss
            ));
        }
        out
    }
}

/// Result of polling a single adapter's checkpoint.
#[derive(Debug)]
pub enum PollResult {
    Ok {
        adapter_idx: usize,
        metadata: CheckpointMetadata,
    },
    Error {
        adapter_idx: usize,
        node_name: String,
        error: String,
    },
}

/// Read checkpoint metadata from a local path.
fn read_local_metadata(checkpoint_dir: &Path) -> Result<CheckpointMetadata, String> {
    let best_meta = checkpoint_dir.join("best").join("metadata.json");
    let contents = std::fs::read_to_string(&best_meta)
        .map_err(|e| format!("failed to read {}: {e}", best_meta.display()))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("failed to parse {}: {e}", best_meta.display()))
}

/// Read checkpoint metadata from a remote node via SSH.
///
/// Currently returns a descriptive error — forjar SSH transport integration
/// is a separate dependency (Phase 3 external).
fn read_ssh_metadata(host: &str, user: Option<&str>, checkpoint_dir: &Path) -> Result<CheckpointMetadata, String> {
    let _user_prefix = user.map_or_else(String::new, |u| format!("{u}@"));
    let _path = checkpoint_dir.join("best").join("metadata.json");
    Err(format!(
        "SSH transport not yet available (forjar dependency): {}{}:{}",
        _user_prefix,
        host,
        _path.display()
    ))
}

/// Build a remote launch command for an adapter job on a node.
///
/// Returns the shell command that would be executed on the remote node
/// to start training.
pub fn build_launch_command(
    node: &NodeConfig,
    model_path: &Path,
    data_path: &Path,
    checkpoint_dir: &Path,
    rank: u32,
    epochs: u32,
) -> String {
    let base = format!(
        "apr finetune {} --task instruct --method qlora --quantize-nf4 \
         --data {} --output {} --rank {rank} --epochs {epochs}",
        model_path.display(),
        data_path.display(),
        checkpoint_dir.display(),
    );

    match node.transport {
        Transport::Local => base,
        Transport::Ssh => {
            let user_prefix = node
                .user
                .as_ref()
                .map_or_else(String::new, |u| format!("{u}@"));
            format!("ssh {user_prefix}{} '{base}'", node.host)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::fs;

    fn test_cluster() -> ClusterConfig {
        ClusterConfig::from_yaml(
            r#"
nodes:
  - name: desktop
    host: localhost
    gpus:
      - uuid: GPU-abcd-1234
        type: rtx-4090
        vram_mb: 24564
    max_adapters: 3
  - name: jetson
    host: jetson.local
    transport: ssh
    gpus:
      - uuid: GPU-efgh-5678
        type: jetson-orin
        vram_mb: 8192
        memory_type: unified
    max_adapters: 1
"#,
        )
        .unwrap()
    }

    fn test_placements() -> Vec<PlacementDecision> {
        vec![
            PlacementDecision {
                adapter_idx: 0,
                node_name: "desktop".to_string(),
                score: 2.5,
            },
            PlacementDecision {
                adapter_idx: 1,
                node_name: "desktop".to_string(),
                score: 1.2,
            },
            PlacementDecision {
                adapter_idx: 2,
                node_name: "jetson".to_string(),
                score: 0.3,
            },
        ]
    }

    #[test]
    fn test_coordinator_creation() {
        let cluster = test_cluster();
        let placements = test_placements();
        let dirs = HashMap::new();
        let coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        assert_eq!(coord.adapters.len(), 3);
        assert_eq!(coord.poll_interval_secs, 300);
    }

    #[test]
    fn test_empty_leaderboard() {
        let cluster = test_cluster();
        let coord = CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        let board = coord.leaderboard();
        assert!(board.is_empty());
        assert!(coord.best_adapter().is_none());
    }

    #[test]
    fn test_leaderboard_with_data() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        // Manually set latest metadata
        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 3,
            avg_loss: 0.5,
            val_loss: Some(0.45),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        });
        coord.adapters.get_mut(&1).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 1,
            epoch: 3,
            avg_loss: 0.8,
            val_loss: Some(0.75),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        });
        coord.adapters.get_mut(&2).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 2,
            epoch: 2,
            avg_loss: 0.3,
            val_loss: Some(0.28),
            node_name: Some("jetson".to_string()),
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 3);
        assert_eq!(board[0].adapter_idx, 2); // 0.28 lowest
        assert_eq!(board[0].rank, 1);
        assert_eq!(board[1].adapter_idx, 0); // 0.45
        assert_eq!(board[2].adapter_idx, 1); // 0.75

        let best = coord.best_adapter().unwrap();
        assert_eq!(best.adapter_idx, 2);
    }

    #[test]
    fn test_poll_local_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let best_dir = dir.path().join("best");
        fs::create_dir_all(&best_dir).unwrap();
        let meta = CheckpointMetadata {
            adapter_idx: 0,
            epoch: 5,
            avg_loss: 0.42,
            val_loss: Some(0.39),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        };
        fs::write(
            best_dir.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let cluster = test_cluster();
        let placements = vec![PlacementDecision {
            adapter_idx: 0,
            node_name: "desktop".to_string(),
            score: 2.5,
        }];
        let mut dirs = HashMap::new();
        dirs.insert(0, dir.path().to_path_buf());

        let mut coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        let results = coord.poll_all();

        assert_eq!(results.len(), 1);
        match &results[0] {
            PollResult::Ok {
                adapter_idx,
                metadata,
            } => {
                assert_eq!(*adapter_idx, 0);
                assert_eq!(metadata.epoch, 5);
                assert!((metadata.avg_loss - 0.42).abs() < f32::EPSILON);
            }
            PollResult::Error { error, .. } => panic!("unexpected error: {error}"),
        }
    }

    #[test]
    fn test_poll_ssh_returns_error() {
        let cluster = test_cluster();
        let placements = vec![PlacementDecision {
            adapter_idx: 2,
            node_name: "jetson".to_string(),
            score: 0.3,
        }];
        let mut coord =
            CheckpointCoordinator::new(cluster, &placements, &HashMap::new(), 300);
        let results = coord.poll_all();

        assert_eq!(results.len(), 1);
        match &results[0] {
            PollResult::Error { error, .. } => {
                assert!(error.contains("SSH transport not yet available"));
            }
            PollResult::Ok { .. } => panic!("expected SSH error"),
        }
    }

    #[test]
    fn test_format_leaderboard() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 2,
            avg_loss: 0.5,
            val_loss: None,
            node_name: None,
            timestamp: None,
        });

        let display = coord.format_leaderboard();
        assert!(display.contains("Adapter Leaderboard"));
        assert!(display.contains("0.5000"));
    }

    #[test]
    fn test_build_launch_command_local() {
        let node = NodeConfig {
            name: "desktop".to_string(),
            host: "localhost".to_string(),
            transport: Transport::Local,
            user: None,
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let cmd = build_launch_command(
            &node,
            Path::new("model.apr"),
            Path::new("data.jsonl"),
            Path::new("/tmp/ckpt"),
            16,
            3,
        );
        assert!(cmd.starts_with("apr finetune model.apr"));
        assert!(cmd.contains("--rank 16"));
        assert!(cmd.contains("--epochs 3"));
        assert!(!cmd.contains("ssh"));
    }

    #[test]
    fn test_build_launch_command_ssh() {
        let node = NodeConfig {
            name: "jetson".to_string(),
            host: "jetson.local".to_string(),
            transport: Transport::Ssh,
            user: Some("noah".to_string()),
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let cmd = build_launch_command(
            &node,
            Path::new("model.apr"),
            Path::new("data.jsonl"),
            Path::new("/tmp/ckpt"),
            16,
            3,
        );
        assert!(cmd.starts_with("ssh noah@jetson.local"));
        assert!(cmd.contains("apr finetune model.apr"));
    }
}

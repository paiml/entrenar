//! WGPU LoRA checkpoint save/load
//!
//! Serializes LoRA adapter weights + optimizer state to JSON for
//! checkpoint/resume during Qwen3-4B NF4 QLoRA training.
//!
//! # Contract: wgpu-transformer-trainer-v1.yaml (C-WGPU-CKPT-001, C-WGPU-CKPT-002)
//!
//! - C-WGPU-CKPT-001: Round-trip preserves all weights bit-identically
//! - C-WGPU-CKPT-002: Rejects dimension mismatch on load

/// LoRA checkpoint: serializable LoRA adapter weights + optimizer state
///
/// Used for saving/loading training checkpoints. Contains all trainable
/// state needed to resume training or run inference with the adapter.
#[cfg(feature = "gpu")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LoraCheckpoint {
    /// Training step at checkpoint
    pub step: u32,
    /// LoRA rank
    pub rank: u32,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Per-layer Q adapter A matrices [rank, in_dim]
    pub q_a: Vec<Vec<f32>>,
    /// Per-layer Q adapter B matrices [out_dim, rank]
    pub q_b: Vec<Vec<f32>>,
    /// Per-layer V adapter A matrices
    pub v_a: Vec<Vec<f32>>,
    /// Per-layer V adapter B matrices
    pub v_b: Vec<Vec<f32>>,
    /// Per-layer Q optimizer state (m_a, v_a, m_b, v_b)
    pub q_opt: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Per-layer V optimizer state
    pub v_opt: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Adapter dimensions: (in_dim, q_out_dim, v_out_dim)
    pub dims: (u32, u32, u32),
    /// Training loss at checkpoint
    pub loss: f32,
    /// Model name (for verification)
    pub model_name: String,
}

/// Save a LoRA checkpoint from model state
///
/// # Contract (C-WGPU-CKPT-001)
///
/// - **Precondition**: output_dir exists or can be created
/// - **Postcondition**: checkpoint file is valid JSON, round-trip preserves all weights
#[cfg(feature = "gpu")]
pub fn save_lora_checkpoint(
    lora_q: &[super::wgpu_nf4::LoraAdapter],
    lora_v: &[super::wgpu_nf4::LoraAdapter],
    hidden_size: usize,
    output_dir: &std::path::Path,
    step: u32,
    loss: f32,
    rank: u32,
    alpha: f32,
) -> Result<std::path::PathBuf, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Cannot create output dir: {e}"))?;

    let checkpoint = LoraCheckpoint {
        step,
        rank,
        alpha,
        q_a: lora_q.iter().map(|l| l.a.clone()).collect(),
        q_b: lora_q.iter().map(|l| l.b.clone()).collect(),
        v_a: lora_v.iter().map(|l| l.a.clone()).collect(),
        v_b: lora_v.iter().map(|l| l.b.clone()).collect(),
        q_opt: lora_q
            .iter()
            .map(|l| (l.m_a.clone(), l.v_a.clone(), l.m_b.clone(), l.v_b.clone()))
            .collect(),
        v_opt: lora_v
            .iter()
            .map(|l| (l.m_a.clone(), l.v_a.clone(), l.m_b.clone(), l.v_b.clone()))
            .collect(),
        dims: (
            hidden_size as u32,
            lora_q.first().map_or(0, |l| l.out_dim),
            lora_v.first().map_or(0, |l| l.out_dim),
        ),
        loss,
        model_name: "qwen3-4b".to_string(),
    };

    let filename = format!("lora-checkpoint-step{step}.json");
    let path = output_dir.join(&filename);
    let json =
        serde_json::to_string(&checkpoint).map_err(|e| format!("Serialize checkpoint: {e}"))?;
    std::fs::write(&path, &json).map_err(|e| format!("Write checkpoint: {e}"))?;

    let size_mb = json.len() as f64 / 1024.0 / 1024.0;
    eprintln!("  Saved checkpoint: {} ({size_mb:.1} MB)", path.display());

    Ok(path)
}

/// Load a LoRA checkpoint and restore adapter weights
///
/// # Contract (C-WGPU-CKPT-002)
///
/// - **Precondition**: checkpoint file exists and is valid JSON
/// - **Postcondition**: model state matches checkpoint exactly (bit-identical weights)
#[cfg(feature = "gpu")]
pub fn load_lora_checkpoint(
    lora_q: &mut [super::wgpu_nf4::LoraAdapter],
    lora_v: &mut [super::wgpu_nf4::LoraAdapter],
    num_layers: usize,
    hidden_size: usize,
    checkpoint_path: &std::path::Path,
) -> Result<(u32, f32), String> {
    let json = std::fs::read_to_string(checkpoint_path)
        .map_err(|e| format!("Read checkpoint: {e}"))?;
    let ckpt: LoraCheckpoint =
        serde_json::from_str(&json).map_err(|e| format!("Parse checkpoint: {e}"))?;

    // Validate dimensions match
    if ckpt.q_a.len() != num_layers {
        return Err(format!(
            "Checkpoint has {} layers, model has {}",
            ckpt.q_a.len(),
            num_layers
        ));
    }

    let expected_in = hidden_size as u32;
    if ckpt.dims.0 != expected_in {
        return Err(format!(
            "Checkpoint hidden_size={}, model has {expected_in}",
            ckpt.dims.0
        ));
    }

    // Restore LoRA weights
    for (i, lora) in lora_q.iter_mut().enumerate() {
        lora.a = ckpt.q_a[i].clone();
        lora.b = ckpt.q_b[i].clone();
        if i < ckpt.q_opt.len() {
            let (ref ma, ref va, ref mb, ref vb) = ckpt.q_opt[i];
            lora.m_a = ma.clone();
            lora.v_a = va.clone();
            lora.m_b = mb.clone();
            lora.v_b = vb.clone();
        }
    }
    for (i, lora) in lora_v.iter_mut().enumerate() {
        lora.a = ckpt.v_a[i].clone();
        lora.b = ckpt.v_b[i].clone();
        if i < ckpt.v_opt.len() {
            let (ref ma, ref va, ref mb, ref vb) = ckpt.v_opt[i];
            lora.m_a = ma.clone();
            lora.v_a = va.clone();
            lora.m_b = mb.clone();
            lora.v_b = vb.clone();
        }
    }

    eprintln!(
        "  Loaded checkpoint: step={}, loss={:.3}, {} layers",
        ckpt.step, ckpt.loss, ckpt.q_a.len()
    );

    Ok((ckpt.step, ckpt.loss))
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::train::transformer_trainer::wgpu_nf4::LoraAdapter;

    /// FALSIFY-CKPT-001: Checkpoint round-trip preserves all LoRA weights bit-identically
    #[test]
    fn test_checkpoint_round_trip() {
        let rank = 4u32;
        let alpha = 8.0f32;
        let hidden = 8usize;
        let q_dim = 8usize;
        let kv_dim = 8usize;

        let lora_q: Vec<LoraAdapter> = (0..2)
            .map(|_| LoraAdapter::new(rank, hidden as u32, q_dim as u32))
            .collect();
        let lora_v: Vec<LoraAdapter> = (0..2)
            .map(|_| LoraAdapter::new(rank, hidden as u32, kv_dim as u32))
            .collect();

        // Save checkpoint
        let tmpdir = std::env::temp_dir().join("entrenar-ckpt-test");
        let ckpt_path =
            save_lora_checkpoint(&lora_q, &lora_v, hidden, &tmpdir, 42, 3.14, rank, alpha)
                .expect("save_checkpoint");

        assert!(ckpt_path.exists(), "Checkpoint file must exist");

        // Load into fresh adapters
        let mut lora_q2: Vec<LoraAdapter> = (0..2)
            .map(|_| LoraAdapter::new(rank, hidden as u32, q_dim as u32))
            .collect();
        let mut lora_v2: Vec<LoraAdapter> = (0..2)
            .map(|_| LoraAdapter::new(rank, hidden as u32, kv_dim as u32))
            .collect();

        let (step, loss) =
            load_lora_checkpoint(&mut lora_q2, &mut lora_v2, 2, hidden, &ckpt_path)
                .expect("load_checkpoint");
        assert_eq!(step, 42);
        assert!((loss - 3.14).abs() < 1e-5);

        // Verify bit-identical LoRA weights
        for i in 0..2 {
            assert_eq!(lora_q[i].a, lora_q2[i].a, "Q adapter A layer {i} mismatch");
            assert_eq!(lora_q[i].b, lora_q2[i].b, "Q adapter B layer {i} mismatch");
            assert_eq!(lora_v[i].a, lora_v2[i].a, "V adapter A layer {i} mismatch");
            assert_eq!(lora_v[i].b, lora_v2[i].b, "V adapter B layer {i} mismatch");
            assert_eq!(lora_q[i].m_a, lora_q2[i].m_a);
            assert_eq!(lora_q[i].v_a, lora_q2[i].v_a);
        }

        let _ = std::fs::remove_dir_all(&tmpdir);
        eprintln!("Checkpoint round-trip: PASS (step={step}, loss={loss:.3})");
    }

    /// FALSIFY-CKPT-002: Checkpoint rejects dimension mismatch
    #[test]
    fn test_checkpoint_dimension_mismatch() {
        let rank = 4u32;

        let lora_q = vec![LoraAdapter::new(rank, 8, 8)];
        let lora_v = vec![LoraAdapter::new(rank, 8, 8)];

        let tmpdir = std::env::temp_dir().join("entrenar-ckpt-mismatch");
        let ckpt_path = save_lora_checkpoint(&lora_q, &lora_v, 8, &tmpdir, 1, 5.0, rank, 8.0)
            .expect("save");

        // Try loading into model with different hidden_size
        let mut lora_q2 = vec![LoraAdapter::new(rank, 16, 16), LoraAdapter::new(rank, 16, 16)];
        let mut lora_v2 = vec![LoraAdapter::new(rank, 16, 16), LoraAdapter::new(rank, 16, 16)];

        let result = load_lora_checkpoint(&mut lora_q2, &mut lora_v2, 2, 16, &ckpt_path);
        assert!(result.is_err(), "Should reject dimension mismatch");

        let _ = std::fs::remove_dir_all(&tmpdir);
        eprintln!("Checkpoint dimension mismatch rejection: PASS");
    }
}

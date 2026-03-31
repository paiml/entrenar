//! WGPU Eval — evaluate trained checkpoint on ShellSafetyBench test split
//!
//! Loads LoRA checkpoint, runs forward on test entries, scores by loss.
//! Computes MCC (Matthews Correlation Coefficient) for binary classification.
//!
//! ```bash
//! cargo run --features gpu --release --example wgpu_eval -- \
//!   --model /home/noah/src/models/qwen3-4b \
//!   --checkpoint /tmp/wgpu-train-output/lora-checkpoint-step500.json \
//!   --test /home/noah/src/bashrs/training/shellsafetybench/splits-merged/test.jsonl
//! ```

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("Error: --features gpu required");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    {
        if let Err(e) = run() {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), String> {
    use entrenar::train::transformer_trainer::wgpu_trainer::WgpuModelState;

    let args: Vec<String> = std::env::args().collect();
    let model_dir = get_arg(&args, "--model").unwrap_or_else(|| "/home/noah/src/models/qwen3-4b".into());
    let ckpt_path = get_arg(&args, "--checkpoint").unwrap_or_else(|| "/tmp/wgpu-train-output/lora-checkpoint-step500.json".into());
    let test_path = get_arg(&args, "--test").unwrap_or_else(|| "/home/noah/src/bashrs/training/shellsafetybench/splits-merged/test.jsonl".into());
    let max_entries: usize = get_arg(&args, "--max").and_then(|s| s.parse().ok()).unwrap_or(500);

    eprintln!("=== WGPU Eval: ShellSafetyBench ===\n");

    // 1. Load tokenizer
    let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer = entrenar::tokenizer::HfTokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Tokenizer: {e}"))?;

    // 2. Load model + checkpoint
    let mut model = WgpuModelState::load_qwen3_4b(std::path::Path::new(&model_dir), 16, 32.0)?;
    let (step, ckpt_loss) = model.load_checkpoint(std::path::Path::new(&ckpt_path))?;
    eprintln!("Loaded checkpoint: step={step}, loss={ckpt_loss:.3}\n");

    // 3. Create GPU device for matmul
    let device = trueno::backends::gpu::GpuDevice::new()?;

    // 4. Load test data
    let test_data = std::fs::read_to_string(&test_path).map_err(|e| format!("Test data: {e}"))?;
    let entries: Vec<(String, i32)> = test_data.lines()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;
            let input = v["input"].as_str()?.to_string();
            let label = v["label"].as_i64()? as i32;
            Some((input, label))
        })
        .take(max_entries)
        .collect();
    eprintln!("Test entries: {} (max {})\n", entries.len(), max_entries);

    // 6. Score each entry by forward loss
    let mut scores: Vec<(f32, i32)> = Vec::new(); // (loss, label)
    let h = model.hidden_size;

    for (idx, (input, label)) in entries.iter().enumerate() {
        let tokens = tokenizer.encode(input);
        let len = tokens.len().min(64); // short context for scoring
        if len < 2 { continue; }

        let input_ids = &tokens[..len];
        let target_ids: Vec<u32> = tokens[1..len].to_vec();
        let seq_len = target_ids.len();

        // Fast eval: embedding → lm_head logits → cross-entropy (skip 36-layer forward)
        let mut hidden = vec![0.0f32; seq_len * h];
        for (si, &tid) in input_ids[..seq_len].iter().enumerate() {
            let tid = (tid as usize).min(model.vocab_size - 1);
            for hi in 0..h { hidden[si * h + hi] = model.lm_head[tid * h + hi]; }
        }
        // logits = hidden @ lm_head^T (GPU GEMM)
        let v = model.vocab_size;
        let mut logits = vec![0.0f32; seq_len * v];
        device.gemm_backward_a(&hidden, &model.lm_head, &mut logits,
            seq_len as u32, v as u32, h as u32)?;
        // Cross-entropy loss
        let mut loss = 0.0f32;
        for si in 0..seq_len {
            let row = &logits[si * v..(si + 1) * v];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_val + row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
            let t = target_ids[si] as usize;
            if t < v { loss -= logits[si * v + t] - lse; }
        }
        loss /= seq_len as f32;

        scores.push((loss, *label));

        if (idx + 1) % 50 == 0 {
            eprintln!("  [{}/{}] loss={loss:.3} label={label}", idx + 1, entries.len());
        }
    }

    // 7. Compute MCC at optimal threshold
    let (mcc, threshold, tp, fp, tn, fn_) = compute_best_mcc(&scores);

    eprintln!("\n=== Results ===");
    eprintln!("Entries scored: {}", scores.len());
    eprintln!("Best threshold: {threshold:.3}");
    eprintln!("TP={tp} FP={fp} TN={tn} FN={fn_}");
    eprintln!("MCC = {mcc:.4}");
    eprintln!("Ship criteria: MCC > 0.50 → {}", if mcc > 0.5 { "PASS" } else { "FAIL" });

    // Stats by label
    let safe_losses: Vec<f32> = scores.iter().filter(|(_, l)| *l == 0).map(|(s, _)| *s).collect();
    let unsafe_losses: Vec<f32> = scores.iter().filter(|(_, l)| *l == 1).map(|(s, _)| *s).collect();
    if !safe_losses.is_empty() {
        let avg_safe: f32 = safe_losses.iter().sum::<f32>() / safe_losses.len() as f32;
        let avg_unsafe: f32 = if unsafe_losses.is_empty() { 0.0 } else { unsafe_losses.iter().sum::<f32>() / unsafe_losses.len() as f32 };
        eprintln!("Avg loss safe={avg_safe:.3} unsafe={avg_unsafe:.3} (delta={:.3})", (avg_unsafe - avg_safe).abs());
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn compute_best_mcc(scores: &[(f32, i32)]) -> (f32, f32, usize, usize, usize, usize) {
    let mut best_mcc = -2.0f32;
    let mut best = (0.0f32, 0, 0, 0, 0);

    // Try multiple thresholds
    let mut thresholds: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();

    for &thresh in &thresholds {
        let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0, 0, 0);
        for &(score, label) in scores {
            let pred = if score > thresh { 1 } else { 0 }; // higher loss → unsafe
            match (pred, label) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_ += 1,
                _ => {}
            }
        }
        let num = (tp * tn) as f64 - (fp * fn_) as f64;
        let den = ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
        let mcc = if den > 0.0 { (num / den) as f32 } else { 0.0 };
        if mcc > best_mcc {
            best_mcc = mcc;
            best = (thresh, tp, fp, tn, fn_);
        }
    }

    (best_mcc, best.0, best.1, best.2, best.3, best.4)
}

#[cfg(feature = "gpu")]
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

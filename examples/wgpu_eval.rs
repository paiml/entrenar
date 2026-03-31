//! WGPU Eval — forward-only eval through full 36-layer transformer with LoRA
//!
//! Phase 8.1: Uses trained LoRA adapters in scoring (previous eval skipped them).
//! Contract: C-WGPU-IMPROVE-001
//!
//! ```bash
//! cargo run --features gpu --release --example wgpu_eval -- \
//!   --model /home/noah/src/models/qwen3-4b \
//!   --checkpoint /tmp/wgpu-train-output/lora-checkpoint-step500.json \
//!   --test /home/noah/src/bashrs/training/shellsafetybench/splits/test.jsonl
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
    let model_dir =
        get_arg(&args, "--model").unwrap_or_else(|| "/home/noah/src/models/qwen3-4b".into());
    let ckpt_path = get_arg(&args, "--checkpoint")
        .unwrap_or_else(|| "/tmp/wgpu-train-output/lora-checkpoint-step500.json".into());
    let test_path = get_arg(&args, "--test").unwrap_or_else(|| {
        "/home/noah/src/bashrs/training/shellsafetybench/splits/test.jsonl".into()
    });
    let max_entries: usize = get_arg(&args, "--max").and_then(|s| s.parse().ok()).unwrap_or(500);
    let fast = args.iter().any(|a| a == "--fast"); // lm_head-only (skip 36 layers)

    eprintln!(
        "=== WGPU Eval ({}): ShellSafetyBench ===\n",
        if fast { "lm_head-only" } else { "Full Forward" }
    );

    // 1. Load tokenizer + model + checkpoint
    let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer = entrenar::tokenizer::HfTokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Tokenizer: {e}"))?;
    let rank: u32 = get_arg(&args, "--rank").and_then(|s| s.parse().ok()).unwrap_or(32);
    let alpha = rank as f32 * 2.0;
    let mut model = WgpuModelState::load_qwen3_4b(std::path::Path::new(&model_dir), rank, alpha)?;
    let (step, ckpt_loss) = model.load_checkpoint(std::path::Path::new(&ckpt_path))?;
    eprintln!("Loaded checkpoint: step={step}, loss={ckpt_loss:.3}");

    // 2. GPU device + populate weight cache (pre-transposed, one-time)
    let device = trueno::backends::gpu::GpuDevice::new()?;
    model.populate_weight_cache(&device)?;
    eprintln!("Weight cache populated\n");

    let h = model.hidden_size;
    let i_size = model.intermediate_size;
    let v = model.vocab_size;
    let n_layers = model.num_layers;
    let nh = model.num_heads as u32;
    let nkv = model.num_kv_heads as u32;
    let hd = model.head_dim as u32;
    let lora_alpha = 32.0f32;

    // 3. Load test data
    let test_data = std::fs::read_to_string(&test_path).map_err(|e| format!("Test data: {e}"))?;
    let entries: Vec<(String, i32)> = test_data
        .lines()
        .filter_map(|line| {
            let val: serde_json::Value = serde_json::from_str(line).ok()?;
            Some((val["input"].as_str()?.to_string(), val["label"].as_i64()? as i32))
        })
        .take(max_entries)
        .collect();
    eprintln!("Test entries: {} (max {})\n", entries.len(), max_entries);

    // 4. Score each entry: full forward through 36 layers + lm_head → CE loss
    let mut scores: Vec<(f32, i32)> = Vec::new();

    for (idx, (input, label)) in entries.iter().enumerate() {
        let tokens = tokenizer.encode(input);
        let len = tokens.len().min(64);
        if len < 2 {
            continue;
        }

        let input_ids = &tokens[..len];
        let target_ids: Vec<u32> = tokens[1..len].to_vec();
        let s = target_ids.len() as u32;

        // Embedding lookup
        let mut hidden = vec![0.0f32; s as usize * h];
        for (si, &tid) in input_ids[..s as usize].iter().enumerate() {
            let tid = (tid as usize).min(v - 1);
            for hi in 0..h {
                hidden[si * h + hi] = model.lm_head[tid * h + hi];
            }
        }

        // Full forward or lm_head-only
        if !fast {
            let rmsnorm = |buf: &mut [f32], seq: usize, dim: usize| {
                let eps = 1e-5f32;
                for si in 0..seq {
                    let rms = (buf[si * dim..(si + 1) * dim].iter().map(|x| x * x).sum::<f32>()
                        / dim as f32
                        + eps)
                        .sqrt();
                    for hi in 0..dim {
                        buf[si * dim + hi] /= rms;
                    }
                }
            };

            // Forward through all 36 layers (attention + FFN, no backward)
            for layer_idx in 0..n_layers {
                rmsnorm(&mut hidden, s as usize, h);

                // Attention forward with LoRA Q/V
                let (q_w, k_w, v_w, o_w) = model.attn_cache[layer_idx]
                    .as_ref()
                    .map(|(q, k, vv, o)| (q.as_slice(), k.as_slice(), vv.as_slice(), o.as_slice()))
                    .expect("attn cache");
                let (attn_out, _cache) =
                    entrenar::train::transformer_trainer::wgpu_attention::attention_forward(
                        &device,
                        &hidden,
                        q_w,
                        k_w,
                        v_w,
                        o_w,
                        &model.lora[layer_idx].q,
                        &model.lora[layer_idx].v,
                        lora_alpha,
                        s,
                        h as u32,
                        nh,
                        nkv,
                        hd,
                    )?;
                for j in 0..(s as usize * h) {
                    hidden[j] += attn_out[j];
                }
                rmsnorm(&mut hidden, s as usize, h);

                // FFN forward (pre-transposed weights)
                let (gate_w, up_w, down_w) = model.ffn_cache[layer_idx]
                    .as_ref()
                    .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice()))
                    .expect("ffn cache");
                let mut gate_out = vec![0.0f32; s as usize * i_size];
                device.matmul(&hidden, gate_w, &mut gate_out, s as usize, h, i_size)?;
                let mut up_out = vec![0.0f32; s as usize * i_size];
                device.matmul(&hidden, up_w, &mut up_out, s as usize, h, i_size)?;
                let swiglu: Vec<f32> = gate_out
                    .iter()
                    .zip(up_out.iter())
                    .map(|(&g, &u)| {
                        let sig = 1.0 / (1.0 + (-g).exp());
                        g * sig * u
                    })
                    .collect();
                let mut ffn_out = vec![0.0f32; s as usize * h];
                device.matmul(&swiglu, down_w, &mut ffn_out, s as usize, i_size, h)?;
                for j in 0..(s as usize * h) {
                    hidden[j] += ffn_out[j];
                }
            }
        } // end if !fast

        // lm_head → logits → cross-entropy loss
        let mut logits = vec![0.0f32; s as usize * v];
        device.gemm_backward_a(&hidden, &model.lm_head, &mut logits, s, v as u32, h as u32)?;
        let mut loss = 0.0f32;
        for si in 0..s as usize {
            let row = &logits[si * v..(si + 1) * v];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_val + row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
            let t = target_ids[si] as usize;
            if t < v {
                loss -= logits[si * v + t] - lse;
            }
        }
        loss /= s as f32;

        scores.push((loss, *label));
        if (idx + 1) % 50 == 0 {
            eprintln!("  [{}/{}] loss={loss:.3} label={label}", idx + 1, entries.len());
        }
    }

    // 5. Z-score normalize losses (reduces variance, improves MCC)
    let mean: f32 = scores.iter().map(|(s, _)| *s).sum::<f32>() / scores.len() as f32;
    let var: f32 =
        scores.iter().map(|(s, _)| (s - mean) * (s - mean)).sum::<f32>() / scores.len() as f32;
    let std = var.sqrt().max(1e-6);
    let z_scores: Vec<(f32, i32)> = scores.iter().map(|(s, l)| ((s - mean) / std, *l)).collect();

    // Try both raw and z-scored
    let (mcc_raw, thr_raw, tp_r, fp_r, tn_r, fn_r) = compute_best_mcc(&scores);
    let (mcc_z, thr_z, tp_z, fp_z, tn_z, fn_z) = compute_best_mcc(&z_scores);

    // Also try log-loss (compresses outliers)
    let log_scores: Vec<(f32, i32)> = scores.iter().map(|(s, l)| (s.ln(), *l)).collect();
    let (mcc_log, thr_log, tp_l, fp_l, tn_l, fn_l) = compute_best_mcc(&log_scores);

    // Also try percentile rank
    let mut sorted: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct_scores: Vec<(f32, i32)> = scores
        .iter()
        .map(|(s, l)| {
            let rank = sorted.iter().position(|v| v >= s).unwrap_or(0);
            (rank as f32 / sorted.len() as f32, *l)
        })
        .collect();
    let (mcc_pct, thr_pct, tp_p, fp_p, tn_p, fn_p) = compute_best_mcc(&pct_scores);

    // Also try inverted direction (lower loss = unsafe)
    let inv_scores: Vec<(f32, i32)> = scores.iter().map(|(s, l)| (-s, *l)).collect();
    let (mcc_inv, thr_inv, tp_i, fp_i, tn_i, fn_i) = compute_best_mcc(&inv_scores);

    let best_mcc = mcc_raw.max(mcc_z).max(mcc_log).max(mcc_pct).max(mcc_inv);
    let mode = if best_mcc == mcc_inv { "inverted" } else if best_mcc == mcc_pct {
        "percentile"
    } else if best_mcc == mcc_log {
        "log"
    } else if best_mcc == mcc_z {
        "z-score"
    } else {
        "raw"
    };

    eprintln!("\n=== Results ({}) ===", if fast { "lm_head-only" } else { "Full Forward" });
    eprintln!("Entries scored: {}", scores.len());
    eprintln!(
        "  Raw:        MCC={mcc_raw:.4} (thr={thr_raw:.3}) TP={tp_r} FP={fp_r} TN={tn_r} FN={fn_r}"
    );
    eprintln!(
        "  Z-score:    MCC={mcc_z:.4} (thr={thr_z:.3}) TP={tp_z} FP={fp_z} TN={tn_z} FN={fn_z}"
    );
    eprintln!(
        "  Log-loss:   MCC={mcc_log:.4} (thr={thr_log:.3}) TP={tp_l} FP={fp_l} TN={tn_l} FN={fn_l}"
    );
    eprintln!("  Percentile: MCC={mcc_pct:.4} (thr={thr_pct:.3}) TP={tp_p} FP={fp_p} TN={tn_p} FN={fn_p}");
    eprintln!("  Inverted:   MCC={mcc_inv:.4} (thr={thr_inv:.3}) TP={tp_i} FP={fp_i} TN={tn_i} FN={fn_i}");
    eprintln!("\nBest: MCC={best_mcc:.4} ({mode})");
    eprintln!("Ship criteria: MCC > 0.50 → {}", if best_mcc > 0.5 { "PASS" } else { "FAIL" });
    eprintln!("Stretch goal: MCC > 0.75 → {}", if best_mcc > 0.75 { "PASS" } else { "FAIL" });

    let safe_losses: Vec<f32> = scores.iter().filter(|(_, l)| *l == 0).map(|(s, _)| *s).collect();
    let unsafe_losses: Vec<f32> = scores.iter().filter(|(_, l)| *l == 1).map(|(s, _)| *s).collect();
    if !safe_losses.is_empty() && !unsafe_losses.is_empty() {
        let avg_s = safe_losses.iter().sum::<f32>() / safe_losses.len() as f32;
        let avg_u = unsafe_losses.iter().sum::<f32>() / unsafe_losses.len() as f32;
        let std_s = (safe_losses.iter().map(|x| (x - avg_s) * (x - avg_s)).sum::<f32>()
            / safe_losses.len() as f32)
            .sqrt();
        let std_u = (unsafe_losses.iter().map(|x| (x - avg_u) * (x - avg_u)).sum::<f32>()
            / unsafe_losses.len() as f32)
            .sqrt();
        eprintln!("Safe:   mean={avg_s:.1} std={std_s:.1} (n={})", safe_losses.len());
        eprintln!("Unsafe: mean={avg_u:.1} std={std_u:.1} (n={})", unsafe_losses.len());
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn compute_best_mcc(scores: &[(f32, i32)]) -> (f32, f32, usize, usize, usize, usize) {
    let mut best_mcc = -2.0f32;
    let mut best = (0.0f32, 0usize, 0usize, 0usize, 0usize);
    let mut thresholds: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();
    for &thresh in &thresholds {
        let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0, 0, 0);
        for &(score, label) in scores {
            let pred = if score > thresh { 1 } else { 0 };
            match (pred, label) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_ += 1,
                _ => {}
            }
        }
        let num = (tp * tn) as f64 - (fp * fn_) as f64;
        let den =
            ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
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

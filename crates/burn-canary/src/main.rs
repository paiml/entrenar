//! Cross-framework canary: trueno WGPU vs Burn WGPU

fn main() {
    if let Err(e) = run() { eprintln!("CANARY FAIL: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), String> {
    use burn::tensor::Tensor;
    use burn::tensor::activation;
    use burn::backend::Wgpu;
    type B = Wgpu;
    let dev = Default::default();

    eprintln!("=== Cross-Framework Canary: trueno vs Burn (WGPU) ===\n");
    let mut pass = 0u32;
    let mut fail = 0u32;

    // Test 1: Matmul
    let (m, k, n) = (16, 64, 32);
    let a_data: Vec<f32> = (0..m*k).map(|i| ((i*7+3)%100) as f32/100.0 - 0.5).collect();
    let b_data: Vec<f32> = (0..k*n).map(|i| ((i*13+7)%100) as f32/100.0 - 0.5).collect();
    let a_b = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(a_data.clone(), [m, k]), &dev);
    let b_b = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(b_data.clone(), [k, n]), &dev);
    let c_burn: Vec<f32> = a_b.matmul(b_b).to_data().to_vec().map_err(|e| format!("{e}"))?;

    let gpu = trueno::backends::gpu::GpuDevice::new().map_err(|e| format!("{e}"))?;
    let mut c_tr = vec![0.0f32; m*n];
    gpu.matmul(&a_data, &b_data, &mut c_tr, m, k, n).map_err(|e| format!("{e}"))?;
    let cos = cosim(&c_burn, &c_tr);
    let md = maxd(&c_burn, &c_tr);
    let ok = cos > 0.9999;
    eprintln!("  [{}] Matmul {m}x{k}x{n}: cos={cos:.6}, max_diff={md:.2e}", if ok {"PASS"} else {"FAIL"});
    if ok { pass += 1; } else { fail += 1; }

    // Test 2: SiLU
    let n2 = 256;
    let d: Vec<f32> = (0..n2).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let t = Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(d.clone(), [n2]), &dev);
    let silu_b: Vec<f32> = activation::silu(t).to_data().to_vec().map_err(|e| format!("{e}"))?;
    let silu_cpu: Vec<f32> = d.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let cos2 = cosim(&silu_b, &silu_cpu);
    let ok2 = cos2 > 0.9999;
    eprintln!("  [{}] SiLU n={n2}: cos={cos2:.6}", if ok2 {"PASS"} else {"FAIL"});
    if ok2 { pass += 1; } else { fail += 1; }

    // Test 3: Matmul + SiLU composed
    let (m3, k3, n3) = (8, 32, 16);
    let x: Vec<f32> = (0..m3*k3).map(|i| ((i*11+5)%100) as f32/100.0 - 0.5).collect();
    let w: Vec<f32> = (0..k3*n3).map(|i| ((i*17+3)%100) as f32/100.0 - 0.5).collect();
    let xb = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(x.clone(), [m3, k3]), &dev);
    let wb = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(w.clone(), [k3, n3]), &dev);
    let yb: Vec<f32> = activation::silu(xb.matmul(wb)).to_data().to_vec().map_err(|e| format!("{e}"))?;
    let mut mm = vec![0.0f32; m3*n3];
    gpu.matmul(&x, &w, &mut mm, m3, k3, n3).map_err(|e| format!("{e}"))?;
    let yt: Vec<f32> = mm.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let cos3 = cosim(&yb, &yt);
    let md3 = maxd(&yb, &yt);
    let ok3 = cos3 > 0.999;
    eprintln!("  [{}] Matmul+SiLU {m3}x{k3}x{n3}: cos={cos3:.6}, max_diff={md3:.2e}", if ok3 {"PASS"} else {"FAIL"});
    if ok3 { pass += 1; } else { fail += 1; }

    eprintln!("\n=== Burn Canary: {pass} PASS, {fail} FAIL ===");
    if fail > 0 { return Err(format!("{fail} tests failed")); }
    Ok(())
}

fn cosim(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let na = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { 0.0 } else { d / (na * nb) }
}
fn maxd(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max) }

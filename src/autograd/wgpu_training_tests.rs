use super::*;
/// FALSIFY-WGSL-GEMM-001: Tiled GEMM matches naive matmul
#[test]
fn test_wgpu_matmul_forward() {
    let trainer = match WgpuTrainer::new() {
        Ok(t) => t,
        Err(_) => return, // Skip if no GPU
    };

    // Small matrix: A[4,8] @ B[8,6] = C[4,6]
    let m = 4u32;
    let k = 8u32;
    let n = 6u32;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let a = trainer.upload(&a_data);
    let b = trainer.upload(&b_data);
    let c = trainer.zeros((m * n) as usize);

    trainer.matmul_forward(&a, &b, &c, m, k, n);
    let result = trainer.download(&c);

    // Compute expected on CPU
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            for kk in 0..k as usize {
                expected[i * n as usize + j] +=
                    a_data[i * k as usize + kk] * b_data[kk * n as usize + j];
            }
        }
    }

    for i in 0..(m * n) as usize {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 1e-3,
            "Mismatch at {}: gpu={} cpu={} err={}",
            i, result[i], expected[i], err
        );
    }
}

/// FALSIFY-WGSL-GEMM-002: Backward GEMM correctness
#[test]
fn test_wgpu_matmul_backward() {
    let trainer = match WgpuTrainer::new() {
        Ok(t) => t,
        Err(_) => return,
    };

    let m = 4u32;
    let k = 3u32;
    let n = 5u32;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let gc_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.05).collect();

    let a = trainer.upload(&a_data);
    let b = trainer.upload(&b_data);
    let gc = trainer.upload(&gc_data);
    let ga = trainer.zeros((m * k) as usize);
    let gb = trainer.zeros((k * n) as usize);

    trainer.matmul_backward(&a, &b, &gc, &ga, &gb, m, k, n);
    let ga_result = trainer.download(&ga);
    let gb_result = trainer.download(&gb);

    // CPU reference: grad_a = grad_c @ B^T
    let mut ga_expected = vec![0.0f32; (m * k) as usize];
    for i in 0..m as usize {
        for j in 0..k as usize {
            for nn in 0..n as usize {
                ga_expected[i * k as usize + j] +=
                    gc_data[i * n as usize + nn] * b_data[j * n as usize + nn];
            }
        }
    }

    // CPU reference: grad_b = A^T @ grad_c
    let mut gb_expected = vec![0.0f32; (k * n) as usize];
    for i in 0..k as usize {
        for j in 0..n as usize {
            for mm in 0..m as usize {
                gb_expected[i * n as usize + j] +=
                    a_data[mm * k as usize + i] * gc_data[mm * n as usize + j];
            }
        }
    }

    for i in 0..(m * k) as usize {
        let err = (ga_result[i] - ga_expected[i]).abs();
        assert!(err < 1e-3, "grad_a[{}]: gpu={} cpu={}", i, ga_result[i], ga_expected[i]);
    }
    for i in 0..(k * n) as usize {
        let err = (gb_result[i] - gb_expected[i]).abs();
        assert!(err < 1e-3, "grad_b[{}]: gpu={} cpu={}", i, gb_result[i], gb_expected[i]);
    }
}

/// AdamW updates params in correct direction
#[test]
fn test_wgpu_adamw_step() {
    let mut trainer = match WgpuTrainer::new() {
        Ok(t) => t,
        Err(_) => return,
    };

    let params = trainer.upload(&[1.0, 2.0, 3.0, 4.0]);
    let grads = trainer.upload(&[0.1, 0.2, 0.3, 0.4]);
    let m_state = trainer.zeros(4);
    let v_state = trainer.zeros(4);

    trainer.adamw_step(&params, &grads, &m_state, &v_state, 0.001, 0.9, 0.999, 1e-8, 0.01);

    let result = trainer.download(&params);
    // Params should have decreased (moved toward lower loss)
    assert!(result[0] < 1.0, "param[0] should decrease: {}", result[0]);
    assert!(result[3] < 4.0, "param[3] should decrease: {}", result[3]);
}

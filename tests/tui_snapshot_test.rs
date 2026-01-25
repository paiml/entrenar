//! TUI Snapshot Tests (ENT-140, ENT-144, ENT-145)
//!
//! Uses insta for snapshot testing following probar's TuiSnapshot pattern.
//! Golden snapshots are stored in tests/snapshots/
//!
//! # Running Tests
//!
//! ```bash
//! cargo test --test tui_snapshot_test
//!
//! # Update snapshots when intentionally changing TUI
//! cargo insta review
//! ```

use entrenar::monitor::tui::{
    render_layout_colored, ColorMode, GpuTelemetry, SamplePeek, TrainingSnapshot, TrainingStatus,
};

/// Create a mock training snapshot for testing
fn mock_snapshot(epoch: usize, step: usize, loss: f32) -> TrainingSnapshot {
    TrainingSnapshot {
        timestamp_ms: 1700000000000,
        epoch,
        total_epochs: 15,
        step,
        steps_per_epoch: 16,
        loss,
        loss_history: vec![6.9, 6.5, 6.0, 5.5, 5.0, loss],
        learning_rate: 6e-4,
        gradient_norm: 2.5,
        tokens_per_second: 30.3,
        start_timestamp_ms: 1700000000000 - 60_000,
        gpu: Some(GpuTelemetry {
            device_name: "NVIDIA GeForce RTX 4090".to_string(),
            utilization_percent: 35.0,
            vram_used_gb: 2.1,
            vram_total_gb: 24.0,
            temperature_celsius: 45.0,
            power_watts: 110.0,
            power_limit_watts: 480.0,
            processes: vec![],
        }),
        sample: Some(SamplePeek {
            input_preview: "fn is_prime(n: u64) -> bool { ... }".to_string(),
            target_preview: "#[test] fn test_is_prime() { ... }".to_string(),
            generated_preview: "(training...)".to_string(),
            token_match_percent: 0.0,
        }),
        status: TrainingStatus::Running,
        experiment_id: "test-experiment".to_string(),
        model_name: "Qwen2.5-Coder-0.5B".to_string(),
    }
}

#[test]
fn test_tui_initial_state_snapshot() {
    let snapshot = mock_snapshot(1, 1, 6.9);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_initial_state", rendered);
}

#[test]
fn test_tui_mid_training_snapshot() {
    let snapshot = mock_snapshot(8, 10, 4.2);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_mid_training", rendered);
}

#[test]
fn test_tui_final_state_snapshot() {
    let mut snapshot = mock_snapshot(15, 16, 2.1);
    snapshot.status = TrainingStatus::Completed;
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_final_state", rendered);
}

#[test]
fn test_tui_error_state_snapshot() {
    let mut snapshot = mock_snapshot(5, 8, 15.0);
    snapshot.status = TrainingStatus::Failed("CUDA OOM".to_string());
    snapshot.gradient_norm = 1000.0; // Exploding gradient
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_error_state", rendered);
}

#[test]
fn test_tui_no_gpu_snapshot() {
    let mut snapshot = mock_snapshot(3, 5, 5.5);
    snapshot.gpu = None; // CPU-only training
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_no_gpu", rendered);
}

#[test]
fn test_tui_no_sample_snapshot() {
    let mut snapshot = mock_snapshot(2, 3, 6.2);
    snapshot.sample = None; // Before first sample
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_no_sample", rendered);
}

// Frame sequence test for progress animation (ENT-144)
#[test]
fn test_tui_progress_sequence() {
    let mut frames = Vec::new();

    // Capture frames at different progress points
    for step in [1, 4, 8, 12, 16] {
        let loss = 6.9 - (step as f32 * 0.3);
        let snapshot = mock_snapshot(1, step, loss);
        let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        frames.push(rendered);
    }

    // Verify frame count
    assert_eq!(frames.len(), 5);

    // Verify frames are different (progress is visible)
    for i in 1..frames.len() {
        assert_ne!(
            frames[i - 1],
            frames[i],
            "Frame {} should differ from frame {}",
            i - 1,
            i
        );
    }
}

// Test different color modes
#[test]
fn test_tui_color_modes() {
    let snapshot = mock_snapshot(5, 8, 4.5);

    // Test each color mode
    let mono = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let color16 = render_layout_colored(&snapshot, 80, ColorMode::Color16);
    let color256 = render_layout_colored(&snapshot, 80, ColorMode::Color256);

    // Color modes should produce different output
    assert_ne!(
        mono.len(),
        color256.len(),
        "Color output should have ANSI codes"
    );

    // Mono should have no ANSI escape sequences
    assert!(
        !mono.contains("\x1b["),
        "Mono mode should not contain ANSI codes"
    );

    // Color modes should have ANSI escape sequences
    assert!(
        color16.contains("\x1b[") || color256.contains("\x1b["),
        "Color modes should contain ANSI codes"
    );
}

// Test different terminal widths
#[test]
fn test_tui_width_adaptation() {
    let snapshot = mock_snapshot(5, 8, 4.5);

    let narrow = render_layout_colored(&snapshot, 60, ColorMode::Mono);
    let standard = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let wide = render_layout_colored(&snapshot, 120, ColorMode::Mono);

    // Different widths should produce different layouts
    assert_ne!(
        narrow, standard,
        "Narrow layout should differ from standard"
    );
    assert_ne!(standard, wide, "Standard layout should differ from wide");
}

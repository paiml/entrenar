#![allow(dead_code)]
//! Per-step wall-clock profiler for CUDA training (KAIZEN-047).
//!
//! Collects `Instant`-based timings for each phase of `train_step_single()`.
//! Reports per-step breakdown and running statistics after N steps.
//!
//! # Contract (C-STEPPROF-001)
//!
//! - Zero-overhead when disabled: all methods are no-ops
//! - No GPU synchronization added (relies on existing sync points)
//! - Timings include CPU→GPU async dispatch latency (not pure kernel time)
//! - Report interval configurable at construction
//!
//! # Phases measured
//!
//! 1. `embed`    — CPU embedding lookup
//! 2. `h2d`      — Hidden state upload (H2D transfer + padding)
//! 3. `forward`  — Block forward loop (includes D2D layer_input saves)
//! 4. `norm_lm`  — Final RMSNorm + LM head GEMM + logits D2H
//! 5. `loss`     — CPU softmax + cross-entropy + gradient
//! 6. `grad_h2d` — Grad logits upload (H2D transfer)
//! 7. `lm_bwd`   — LM head backward (GEMM_A + GEMM_B + clip)
//! 8. `norm_bwd` — Final RMSNorm backward + clip
//! 9. `blk_bwd`  — Block backward loop (includes recompute + optimizer)
//! 10. `embed_bwd` — Embedding backward (D2H + clip + scatter-add)
//! 11. `opt`      — CPU optimizer step (embedding + bookkeeping)

use std::time::{Duration, Instant};

/// Phase indices (must match `PHASE_NAMES`).
const EMBED: usize = 0;
const H2D: usize = 1;
const FORWARD: usize = 2;
const NORM_LM: usize = 3;
const LOSS: usize = 4;
const GRAD_H2D: usize = 5;
const LM_BWD: usize = 6;
const NORM_BWD: usize = 7;
const BLK_BWD: usize = 8;
const EMBED_BWD: usize = 9;
const OPT: usize = 10;
const NUM_PHASES: usize = 11;

const PHASE_NAMES: [&str; NUM_PHASES] = [
    "embed",
    "h2d",
    "forward",
    "norm_lm",
    "loss",
    "grad_h2d",
    "lm_bwd",
    "norm_bwd",
    "blk_bwd",
    "embed_bwd",
    "opt",
];

/// Per-step timing accumulator.
///
/// Usage: call `begin(phase)` before each section, `end(phase)` after.
/// Call `finish_step()` to record the step and optionally print a report.
pub struct StepProfiler {
    enabled: bool,
    /// Report every N steps (0 = never auto-report)
    report_interval: usize,
    /// Current step's phase timings
    current: [Duration; NUM_PHASES],
    /// Phase start timestamp (set by `begin`)
    phase_start: Option<Instant>,
    /// Step-level start timestamp
    step_start: Option<Instant>,
    /// Accumulated totals across all steps
    totals: [Duration; NUM_PHASES],
    /// Total wall-clock across all steps
    total_wall: Duration,
    /// Number of completed steps
    step_count: usize,
    /// Per-step wall-clock durations (for percentile analysis)
    step_durations: Vec<Duration>,
}

impl StepProfiler {
    /// Create a new profiler.
    ///
    /// - `enabled`: if false, all methods are no-ops
    /// - `report_interval`: print summary every N steps (0 = manual only)
    pub fn new(enabled: bool, report_interval: usize) -> Self {
        Self {
            enabled,
            report_interval,
            current: [Duration::ZERO; NUM_PHASES],
            phase_start: None,
            step_start: None,
            totals: [Duration::ZERO; NUM_PHASES],
            total_wall: Duration::ZERO,
            step_count: 0,
            step_durations: Vec::new(),
        }
    }

    /// Disabled (no-op) profiler.
    pub fn disabled() -> Self {
        Self::new(false, 0)
    }

    /// Mark the start of a training step.
    #[inline]
    pub fn begin_step(&mut self) {
        if !self.enabled {
            return;
        }
        self.current = [Duration::ZERO; NUM_PHASES];
        self.step_start = Some(Instant::now());
    }

    /// Mark the start of a phase. Must call `end(phase)` to record.
    #[inline]
    pub fn begin(&mut self, _phase: usize) {
        if !self.enabled {
            return;
        }
        self.phase_start = Some(Instant::now());
    }

    /// Record elapsed time for a phase (since last `begin`).
    #[inline]
    pub fn end(&mut self, phase: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.phase_start.take() {
            self.current[phase] += start.elapsed();
        }
    }

    /// Finish the current step. Records totals and optionally prints report.
    pub fn finish_step(&mut self) {
        if !self.enabled {
            return;
        }
        let step_wall = self.step_start.take().map_or(Duration::ZERO, |s| s.elapsed());

        for i in 0..NUM_PHASES {
            self.totals[i] += self.current[i];
        }
        self.total_wall += step_wall;
        self.step_count += 1;
        self.step_durations.push(step_wall);

        if self.report_interval > 0 && self.step_count % self.report_interval == 0 {
            self.print_report();
        }
    }

    /// Print cumulative profiling report to stdout.
    pub fn print_report(&self) {
        if self.step_count == 0 {
            return;
        }

        let total_us = self.total_wall.as_micros() as f64;
        let avg_step_us = total_us / self.step_count as f64;

        println!(
            "\n┌─ Step Profiler ({} steps, avg {:.1} ms/step) ─┐",
            self.step_count,
            avg_step_us / 1000.0
        );
        println!("│ {:>10} │ {:>8} │ {:>6} │ {:>8} │", "phase", "total_ms", "pct", "avg_ms");
        println!("│ {:-<10} │ {:-<8} │ {:-<6} │ {:-<8} │", "", "", "", "");

        let mut accounted = Duration::ZERO;
        for i in 0..NUM_PHASES {
            let t = self.totals[i];
            accounted += t;
            let ms = t.as_micros() as f64 / 1000.0;
            let pct = if total_us > 0.0 { t.as_micros() as f64 / total_us * 100.0 } else { 0.0 };
            let avg = ms / self.step_count as f64;
            println!("│ {:>10} │ {:>8.1} │ {:>5.1}% │ {:>8.2} │", PHASE_NAMES[i], ms, pct, avg);
        }

        let unaccounted = self.total_wall.saturating_sub(accounted);
        let unaccounted_pct =
            if total_us > 0.0 { unaccounted.as_micros() as f64 / total_us * 100.0 } else { 0.0 };
        println!(
            "│ {:>10} │ {:>8.1} │ {:>5.1}% │ {:>8.2} │",
            "other",
            unaccounted.as_micros() as f64 / 1000.0,
            unaccounted_pct,
            unaccounted.as_micros() as f64 / 1000.0 / self.step_count as f64
        );

        println!(
            "│ {:>10} │ {:>8.1} │ {:>5}  │ {:>8.2} │",
            "TOTAL",
            total_us / 1000.0,
            "100%",
            avg_step_us / 1000.0
        );
        println!("└────────────┴──────────┴────────┴──────────┘");

        // Percentiles for step wall-clock
        if self.step_durations.len() >= 10 {
            let mut sorted: Vec<u128> =
                self.step_durations.iter().map(std::time::Duration::as_micros).collect();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p95 = sorted[sorted.len() * 95 / 100];
            let p99 = sorted[sorted.len() * 99 / 100];
            println!(
                "  Step latency: p50={:.1}ms p95={:.1}ms p99={:.1}ms",
                p50 as f64 / 1000.0,
                p95 as f64 / 1000.0,
                p99 as f64 / 1000.0
            );
        }
    }

    /// Whether the profiler is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    // Phase constants for external callers
    pub const EMBED: usize = EMBED;
    pub const H2D: usize = H2D;
    pub const FORWARD: usize = FORWARD;
    pub const NORM_LM: usize = NORM_LM;
    pub const LOSS: usize = LOSS;
    pub const GRAD_H2D: usize = GRAD_H2D;
    pub const LM_BWD: usize = LM_BWD;
    pub const NORM_BWD: usize = NORM_BWD;
    pub const BLK_BWD: usize = BLK_BWD;
    pub const EMBED_BWD: usize = EMBED_BWD;
    pub const OPT: usize = OPT;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_profiler_is_noop() {
        let mut p = StepProfiler::disabled();
        p.begin_step();
        p.begin(StepProfiler::EMBED);
        p.end(StepProfiler::EMBED);
        p.finish_step();
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_enabled_profiler_counts_steps() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        p.begin(StepProfiler::EMBED);
        std::thread::sleep(Duration::from_millis(1));
        p.end(StepProfiler::EMBED);
        p.finish_step();
        assert_eq!(p.step_count(), 1);
        assert!(p.totals[EMBED] >= Duration::from_micros(500));
    }

    #[test]
    fn test_multiple_steps_accumulate() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..3 {
            p.begin_step();
            p.begin(StepProfiler::LOSS);
            std::thread::sleep(Duration::from_millis(1));
            p.end(StepProfiler::LOSS);
            p.finish_step();
        }
        assert_eq!(p.step_count(), 3);
        assert!(p.totals[LOSS] >= Duration::from_millis(3));
    }

    #[test]
    fn test_unaccounted_time_captured() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        // Sleep without recording any phase → should appear as "other"
        std::thread::sleep(Duration::from_millis(2));
        p.finish_step();
        assert!(p.total_wall >= Duration::from_millis(2));
        // All phase totals should be zero
        for i in 0..NUM_PHASES {
            assert_eq!(p.totals[i], Duration::ZERO);
        }
    }
}

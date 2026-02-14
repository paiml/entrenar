//! TUI Rendering - Clean, Simple, Labeled Layout
//!
//! Design: Tufte data-ink ratio, clear labels, no chartjunk

use super::color::{ColorMode, Styled, TrainingPalette};
use super::state::{TrainingSnapshot, TrainingStatus};
use std::time::Duration;

const BLOCK_FULL: char = '█';
const BLOCK_LIGHT: char = '░';
const ARROW_UP: &str = "↑";
const ARROW_DOWN: &str = "↓";
const ARROW_FLAT: &str = "→";
const BRAILLE_BASE: u32 = 0x2800;
const BRAILLE_DOTS: [u32; 8] = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];

// ═══════════════════════════════════════════════════════════════════════════════
// CORE RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
pub fn build_block_bar(percent: f32, width: usize) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f32).max(0.0) as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{}",
        BLOCK_FULL.to_string().repeat(filled),
        BLOCK_LIGHT.to_string().repeat(empty)
    )
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
pub fn build_colored_block_bar(percent: f32, width: usize, color_mode: ColorMode) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f32).max(0.0) as usize;
    let empty = width.saturating_sub(filled);

    let color = pct_color(pct);
    let filled_str = BLOCK_FULL.to_string().repeat(filled);
    let empty_str = BLOCK_LIGHT.to_string().repeat(empty);

    if color_mode == ColorMode::Mono {
        format!("{filled_str}{empty_str}")
    } else {
        format!(
            "{}{}",
            Styled::new(&filled_str, color_mode).fg(color),
            Styled::new(&empty_str, color_mode).fg((60, 60, 60))
        )
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn pct_color(pct: f32) -> (u8, u8, u8) {
    let p = pct.clamp(0.0, 100.0);
    if p >= 90.0 {
        (255, 64, 64)
    } else if p >= 75.0 {
        let t = (p - 75.0) / 15.0;
        (255, (180.0 - t * 116.0).clamp(0.0, 255.0) as u8, 64)
    } else if p >= 50.0 {
        let t = (p - 50.0) / 25.0;
        (255, (220.0 - t * 40.0).clamp(0.0, 255.0) as u8, 64)
    } else if p >= 25.0 {
        let t = (p - 25.0) / 25.0;
        (
            (100.0 + t * 155.0).clamp(0.0, 255.0) as u8,
            220,
            (100.0 - t * 36.0).clamp(0.0, 255.0) as u8,
        )
    } else {
        let t = p / 25.0;
        (
            (64.0 + t * 36.0).clamp(0.0, 255.0) as u8,
            (180.0 + t * 40.0).clamp(0.0, 255.0) as u8,
            (220.0 - t * 120.0).clamp(0.0, 255.0) as u8,
        )
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
pub fn render_sparkline(data: &[f32], width: usize, color_mode: ColorMode) -> String {
    if data.is_empty() {
        return " ".repeat(width);
    }

    let min = data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let max = data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(0.001);

    let mut result = String::new();
    for i in 0..width {
        let idx = (i * data.len()) / width.max(1);
        let idx2 = ((i * 2 + 1) * data.len()) / (width * 2).max(1);

        let v1 = data.get(idx).copied().unwrap_or(min);
        let v2 = data.get(idx2).copied().unwrap_or(v1);

        let h1 = if v1.is_finite() {
            (((v1 - min) / range) * 3.99).max(0.0) as usize
        } else {
            0
        };
        let h2 = if v2.is_finite() {
            (((v2 - min) / range) * 3.99).max(0.0) as usize
        } else {
            0
        };

        let mut code: u32 = 0;
        for y in 0..=h1.min(3) {
            code |= BRAILLE_DOTS[3 - y];
        }
        for y in 0..=h2.min(3) {
            code |= BRAILLE_DOTS[7 - y];
        }

        result.push(char::from_u32(BRAILLE_BASE + code).unwrap_or('⣿'));
    }

    let trend_color = if data.len() > 1 {
        let first = data.first().copied().unwrap_or(0.0);
        let last = data.last().copied().unwrap_or(0.0);
        if last < first * 0.95 {
            TrainingPalette::SUCCESS
        } else if last > first * 1.05 {
            TrainingPalette::ERROR
        } else {
            TrainingPalette::INFO
        }
    } else {
        TrainingPalette::INFO
    };

    if color_mode == ColorMode::Mono {
        result
    } else {
        Styled::new(&result, color_mode).fg(trend_color).to_string()
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn trend_arrow(data: &[f32]) -> &'static str {
    if data.len() < 2 {
        return ARROW_FLAT;
    }
    let recent: Vec<f32> = data.iter().rev().take(5).copied().collect();
    if recent.len() < 2 {
        return ARROW_FLAT;
    }
    let avg_recent: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
    let old_count = data.len().saturating_sub(5).clamp(1, 5);
    let avg_old: f32 = data.iter().rev().skip(5).take(5).copied().sum::<f32>() / old_count as f32;

    if avg_recent < avg_old * 0.95 {
        ARROW_DOWN
    } else if avg_recent > avg_old * 1.05 {
        ARROW_UP
    } else {
        ARROW_FLAT
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMATTING
// ═══════════════════════════════════════════════════════════════════════════════

pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        let gb = bytes / (1024 * 1024 * 1024);
        let frac = (bytes % (1024 * 1024 * 1024)) as f64 / (1024.0 * 1024.0 * 1024.0);
        format!("{:.1}G", gb as f64 + frac)
    } else if bytes >= 1024 * 1024 {
        let mb = bytes / (1024 * 1024);
        format!("{mb}M")
    } else if bytes >= 1024 {
        let kb = bytes / 1024;
        format!("{kb}K")
    } else {
        format!("{bytes}B")
    }
}

pub fn format_lr(lr: f32) -> String {
    if !lr.is_finite() {
        return "???".to_string();
    }
    let lr = lr.max(0.0);
    if lr >= 0.01 {
        format!("{lr:.4}")
    } else if lr >= 0.001 {
        format!("{lr:.5}")
    } else {
        format!("{lr:.6}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EPOCH HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct EpochSummary {
    pub epoch: usize,
    pub avg_loss: f32,
    pub min_loss: f32,
    pub max_loss: f32,
    pub end_loss: f32,
    pub avg_grad: f32,
    pub lr: f32,
    pub tokens_per_sec: f32,
}

#[allow(clippy::cast_precision_loss)]
pub fn compute_epoch_summaries(snapshot: &TrainingSnapshot) -> Vec<EpochSummary> {
    if snapshot.steps_per_epoch == 0 || snapshot.loss_history.is_empty() {
        return Vec::new();
    }

    let steps = snapshot.steps_per_epoch;
    let mut summaries = Vec::new();

    for (epoch_idx, chunk) in snapshot.loss_history.chunks(steps).enumerate() {
        let valid: Vec<f32> = chunk.iter().copied().filter(|v| v.is_finite()).collect();
        if valid.is_empty() {
            continue;
        }

        let avg_loss = valid.iter().sum::<f32>() / valid.len() as f32;
        let min_loss = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max_loss = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let end_loss = *valid.last().unwrap_or(&0.0);

        let lr = if snapshot.lr_history.is_empty() {
            snapshot.learning_rate
        } else {
            let lr_start = epoch_idx * steps;
            let lr_end = (lr_start + steps).min(snapshot.lr_history.len());
            if lr_start < snapshot.lr_history.len() {
                snapshot.lr_history[lr_start..lr_end].iter().sum::<f32>()
                    / (lr_end - lr_start).max(1) as f32
            } else {
                snapshot.learning_rate
            }
        };

        summaries.push(EpochSummary {
            epoch: epoch_idx + 1,
            avg_loss,
            min_loss,
            max_loss,
            end_loss,
            avg_grad: snapshot.gradient_norm.max(0.0),
            lr,
            tokens_per_sec: snapshot.tokens_per_second.max(0.0),
        });
    }
    summaries
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN LAYOUT - Simple, Clean, Labeled
// ═══════════════════════════════════════════════════════════════════════════════

pub fn render_layout(snapshot: &TrainingSnapshot, width: usize) -> String {
    render_layout_colored(snapshot, width, ColorMode::detect())
}

#[allow(clippy::cast_precision_loss)]
pub fn render_layout_colored(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    let w = width.max(80);

    render_header(&mut lines, snapshot, w, color_mode);
    render_progress(&mut lines, snapshot, w, color_mode);
    render_metrics(&mut lines, snapshot, color_mode);
    render_loss_sparkline(&mut lines, snapshot, w, color_mode);
    render_gpu_section(&mut lines, snapshot, w, color_mode);
    render_epoch_table(&mut lines, snapshot, w, color_mode);
    render_config_footer(&mut lines, snapshot, w, color_mode);

    lines.join("\n")
}

fn render_header(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    let (status_icon, status_text, status_color) = match &snapshot.status {
        TrainingStatus::Initializing => ("◐", "Init", TrainingPalette::INFO),
        TrainingStatus::Running => ("●", "Running", TrainingPalette::SUCCESS),
        TrainingStatus::Paused => ("◐", "Paused", TrainingPalette::WARNING),
        TrainingStatus::Completed => ("○", "Done", TrainingPalette::PRIMARY),
        TrainingStatus::Failed(_) => ("●", "FAIL", TrainingPalette::ERROR),
    };

    let elapsed = format_duration(snapshot.elapsed());
    let tps = snapshot.tokens_per_second.max(0.0);

    let model_display = if snapshot.model_name.is_empty() {
        "N/A".to_string()
    } else if snapshot.model_name.len() > 30 {
        format!("{}...", &snapshot.model_name[..27])
    } else {
        snapshot.model_name.clone()
    };

    let header = format!(
        "ENTRENAR  {}  {} {}  {:.0} tok/s",
        Styled::new(&format!("{status_icon} {status_text}"), color_mode).fg(status_color),
        Styled::new(&elapsed, color_mode).fg((150, 150, 150)),
        Styled::new(&model_display, color_mode).fg((180, 180, 220)),
        tps
    );
    lines.push(format!("═{:═<w$}═", "", w = w - 2));
    lines.push(header);
    lines.push(format!("─{:─<w$}─", "", w = w - 2));
}

#[allow(clippy::cast_precision_loss)]
fn render_progress(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    _w: usize,
    color_mode: ColorMode,
) {
    let epoch = snapshot.epoch.min(snapshot.total_epochs);
    let epoch_pct = if snapshot.total_epochs > 0 {
        (epoch as f32 / snapshot.total_epochs as f32 * 100.0).min(100.0)
    } else {
        0.0
    };

    let step = snapshot.step.min(snapshot.steps_per_epoch);
    let step_pct = if snapshot.steps_per_epoch > 0 {
        (step as f32 / snapshot.steps_per_epoch as f32 * 100.0).min(100.0)
    } else {
        0.0
    };

    let bar_w = 20;
    let epoch_bar = build_colored_block_bar(epoch_pct, bar_w, color_mode);
    let step_bar = build_colored_block_bar(step_pct, bar_w, color_mode);

    lines.push(format!(
        "Epoch {:>2}/{:<2} {} {:>3.0}%    Step {:>2}/{:<2} {} {:>3.0}%",
        epoch,
        snapshot.total_epochs,
        epoch_bar,
        epoch_pct,
        step,
        snapshot.steps_per_epoch,
        step_bar,
        step_pct
    ));
}

fn render_metrics(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    color_mode: ColorMode,
) {
    let loss_str = if snapshot.loss.is_finite() {
        format!("{:.4}", snapshot.loss)
    } else {
        "???".to_string()
    };
    let loss_color = if snapshot.loss.is_finite() {
        pct_color((snapshot.loss * 10.0).min(100.0))
    } else {
        (255, 64, 64)
    };

    let best = snapshot
        .loss_history
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let best_str = if best.is_finite() {
        format!("{best:.4}")
    } else {
        "---".to_string()
    };

    let grad = snapshot.gradient_norm.max(0.0);
    let eta = snapshot
        .estimated_remaining()
        .map_or("--:--:--".to_string(), format_duration);

    lines.push(format!(
        "Loss {} {}  Best {}  LR {}  Grad {:.2}  ETA {}",
        Styled::new(&loss_str, color_mode).fg(loss_color),
        trend_arrow(&snapshot.loss_history),
        Styled::new(&best_str, color_mode).fg((100, 200, 100)),
        format_lr(snapshot.learning_rate),
        grad,
        eta
    ));
}

fn render_loss_sparkline(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    if snapshot.loss_history.is_empty() {
        return;
    }

    let spark_w = w.saturating_sub(20);
    let sparkline = render_sparkline(&snapshot.loss_history, spark_w, color_mode);

    let valid: Vec<f32> = snapshot
        .loss_history
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let (min_l, max_l) = if valid.is_empty() {
        (0.0, 0.0)
    } else {
        (
            valid.iter().copied().fold(f32::INFINITY, f32::min),
            valid.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    };

    lines.push(format!("─{:─<w$}─", "", w = w - 2));
    lines.push(format!(
        "Loss History: {sparkline} [{min_l:.2} - {max_l:.2}]"
    ));
}

fn render_gpu_section(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("─{:─<w$}─", "", w = w - 2));
    if let Some(gpu) = &snapshot.gpu {
        let util_bar = build_colored_block_bar(gpu.utilization_percent, 15, color_mode);
        let vram_pct = gpu.vram_percent().min(100.0);
        let vram_bar = build_colored_block_bar(vram_pct, 15, color_mode);

        let temp_color = if gpu.temperature_celsius > 80.0 {
            TrainingPalette::ERROR
        } else if gpu.temperature_celsius > 70.0 {
            TrainingPalette::WARNING
        } else {
            TrainingPalette::SUCCESS
        };

        lines.push(format!(
            "GPU: {}  Util {} {:>3.0}%  Temp {}  Power {:.0}W",
            gpu.device_name.chars().take(20).collect::<String>(),
            util_bar,
            gpu.utilization_percent,
            Styled::new(&format!("{:.0}°C", gpu.temperature_celsius), color_mode).fg(temp_color),
            gpu.power_watts
        ));

        lines.push(format!(
            "VRAM: {} {:>3.0}%  {:.1}G / {:.0}G",
            vram_bar,
            vram_pct,
            gpu.vram_used_gb.min(gpu.vram_total_gb),
            gpu.vram_total_gb
        ));
    } else {
        lines.push("GPU: N/A".to_string());
    }
}

fn render_epoch_table(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("─{:─<w$}─", "", w = w - 2));
    lines.push(format!(
        "{:>5}  {:>8}  {:>8}  {:>8}  {:>10}  {:>5}",
        Styled::new("Epoch", color_mode).fg((150, 150, 150)),
        Styled::new("Loss", color_mode).fg((150, 150, 150)),
        Styled::new("Min", color_mode).fg((150, 150, 150)),
        Styled::new("Max", color_mode).fg((150, 150, 150)),
        Styled::new("LR", color_mode).fg((150, 150, 150)),
        Styled::new("Trend", color_mode).fg((150, 150, 150)),
    ));

    let summaries = compute_epoch_summaries(snapshot);
    if summaries.is_empty() {
        lines.push("  (waiting for epoch data...)".to_string());
        return;
    }

    let max_rows = 6;
    let start_idx = summaries.len().saturating_sub(max_rows);

    for (i, summary) in summaries.iter().skip(start_idx).enumerate() {
        let trend = epoch_trend_arrow(i, start_idx, &summaries, color_mode);
        let loss_color = pct_color((summary.avg_loss * 8.0).min(100.0));

        lines.push(format!(
            "{:>5}  {}  {:>8.4}  {:>8.4}  {:>10}  {:>5}",
            summary.epoch,
            Styled::new(&format!("{:>8.4}", summary.avg_loss), color_mode).fg(loss_color),
            summary.min_loss,
            summary.max_loss,
            format_lr(summary.lr),
            trend
        ));
    }

    if start_idx > 0 {
        lines.push(format!(
            "  ... {} earlier epochs",
            Styled::new(&format!("{start_idx}"), color_mode).fg((100, 100, 100))
        ));
    }
}

fn epoch_trend_arrow(
    i: usize,
    start_idx: usize,
    summaries: &[EpochSummary],
    color_mode: ColorMode,
) -> String {
    if i == 0 && start_idx == 0 {
        return " ".to_string();
    }
    let prev_idx = if i > 0 {
        start_idx + i - 1
    } else {
        start_idx.saturating_sub(1)
    };
    if let Some(prev) = summaries.get(prev_idx) {
        let current = &summaries[start_idx + i];
        let change = (current.avg_loss - prev.avg_loss) / prev.avg_loss.abs().max(0.001);
        if change < -0.02 {
            Styled::new("↓", color_mode).fg((100, 255, 100)).to_string()
        } else if change > 0.02 {
            Styled::new("↑", color_mode).fg((255, 100, 100)).to_string()
        } else {
            Styled::new("→", color_mode).fg((150, 150, 150)).to_string()
        }
    } else {
        " ".to_string()
    }
}

fn render_config_footer(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("─{:─<w$}─", "", w = w - 2));

    let opt = if snapshot.optimizer_name.is_empty() {
        "N/A"
    } else {
        &snapshot.optimizer_name
    };
    let batch = if snapshot.batch_size > 0 {
        format!("{}", snapshot.batch_size)
    } else {
        "N/A".to_string()
    };

    lines.push(format!(
        "Config: {}  Batch: {}  Checkpoint: {}",
        Styled::new(opt, color_mode).fg((150, 255, 150)),
        batch,
        if snapshot.checkpoint_path.is_empty() {
            "N/A"
        } else {
            &snapshot.checkpoint_path
        }
    ));

    lines.push(format!("═{:═<w$}═", "", w = w - 2));
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════

pub fn render_config_panel(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines = Vec::new();

    let model_name = if snapshot.model_name.is_empty() {
        "N/A"
    } else {
        &snapshot.model_name
    };
    let model_display: String = model_name.chars().take(width - 8).collect();
    lines.push(
        Styled::new(&model_display, color_mode)
            .fg((180, 180, 255))
            .to_string(),
    );

    let opt = if snapshot.optimizer_name.is_empty() {
        "N/A"
    } else {
        &snapshot.optimizer_name
    };
    let batch = if snapshot.batch_size > 0 {
        format!("batch:{}", snapshot.batch_size)
    } else {
        "N/A".to_string()
    };
    lines.push(format!("{opt}  {batch}"));

    lines.join("\n")
}

pub fn render_history_table(
    snapshot: &TrainingSnapshot,
    width: usize,
    max_rows: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines = Vec::new();

    let header = format!(
        "{:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>5}",
        "Epoch", "Loss", "Min", "Max", "LR", "Tok/s", "Trend"
    );
    lines.push(
        Styled::new(&header, color_mode)
            .fg((150, 150, 150))
            .to_string(),
    );
    lines.push("─".repeat(width.min(70)));

    let summaries = compute_epoch_summaries(snapshot);
    if summaries.is_empty() {
        lines.push("(waiting for epoch data...)".to_string());
        return lines.join("\n");
    }

    let start_idx = summaries.len().saturating_sub(max_rows);
    for (i, summary) in summaries.iter().skip(start_idx).enumerate() {
        let trend = if i > 0 || start_idx > 0 {
            let prev_idx = if i > 0 {
                start_idx + i - 1
            } else {
                start_idx.saturating_sub(1)
            };
            if let Some(prev) = summaries.get(prev_idx) {
                let change = (summary.avg_loss - prev.avg_loss) / prev.avg_loss.abs().max(0.001);
                if change < -0.02 {
                    ("↓", (100, 255, 100))
                } else if change > 0.02 {
                    ("↑", (255, 100, 100))
                } else {
                    ("→", (150, 150, 150))
                }
            } else {
                ("", (150, 150, 150))
            }
        } else {
            ("", (150, 150, 150))
        };

        let row = format!(
            "{:>5} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.1} {}",
            summary.epoch,
            summary.avg_loss,
            summary.min_loss,
            summary.max_loss,
            format_lr(summary.lr),
            summary.tokens_per_sec,
            Styled::new(trend.0, color_mode).fg(trend.1)
        );
        lines.push(row);
    }

    if start_idx > 0 {
        lines.push(format!("  ↑ {start_idx} more epochs above"));
    }

    lines.join("\n")
}

pub fn render_gauge(value: f32, max: f32, width: usize, label: &str) -> String {
    let percent = if max > 0.0 { value / max * 100.0 } else { 0.0 };
    let bar = build_block_bar(percent, width.saturating_sub(label.len() + 8));
    format!("{label}{bar} {percent:>5.1}%")
}

pub struct BrailleChart {
    width: usize,
    height: usize,
    data: Vec<f32>,
    color_mode: ColorMode,
}

impl BrailleChart {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: Vec::new(),
            color_mode: ColorMode::detect(),
        }
    }

    pub fn color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = data;
        self
    }

    #[allow(dead_code)]
    pub fn bounds(self, _min: f32, _max: f32) -> Self {
        self
    }

    pub fn log_scale(self, _enabled: bool) -> Self {
        self
    }

    pub fn render(&self) -> String {
        if self.data.is_empty() {
            return " ".repeat(self.width).repeat(self.height);
        }
        let mut lines = Vec::new();
        for row in 0..self.height {
            let start = (row * self.data.len()) / self.height;
            let end = ((row + 1) * self.data.len()) / self.height;
            let slice = if end > start {
                &self.data[start..end]
            } else if start < self.data.len() {
                &self.data[start..=start]
            } else {
                &[]
            };
            lines.push(render_sparkline(slice, self.width, self.color_mode));
        }
        lines.join("\n")
    }
}

pub fn render_braille_chart(data: &[f32], width: usize, height: usize, _log_scale: bool) -> String {
    BrailleChart::new(width, height)
        .data(data.to_vec())
        .render()
}

use super::state::SamplePeek;

pub fn render_sample_panel(
    _sample: Option<&SamplePeek>,
    _width: usize,
    _color_mode: ColorMode,
) -> String {
    String::new()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_bar() {
        let bar = build_block_bar(50.0, 10);
        assert_eq!(bar.chars().count(), 10);
        assert!(bar.contains(BLOCK_FULL));
        assert!(bar.contains(BLOCK_LIGHT));
    }

    #[test]
    fn test_pct_color_gradient() {
        let mut prev = pct_color(0.0);
        for i in 1..=100 {
            let curr = pct_color(i as f32);
            let dr = (curr.0 as i32 - prev.0 as i32).abs();
            let dg = (curr.1 as i32 - prev.1 as i32).abs();
            let db = (curr.2 as i32 - prev.2 as i32).abs();
            assert!(dr < 50 && dg < 50 && db < 50, "Color jump at {}%", i);
            prev = curr;
        }
    }

    #[test]
    fn test_sparkline() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let spark = render_sparkline(&data, 5, ColorMode::Mono);
        assert!(!spark.is_empty());
    }

    #[test]
    fn test_trend_arrow() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(trend_arrow(&increasing), ARROW_UP);

        let decreasing = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(trend_arrow(&decreasing), ARROW_DOWN);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512M");
        assert_eq!(format_bytes(1024), "1K");
    }

    #[test]
    fn test_format_lr() {
        assert_eq!(format_lr(0.01), "0.0100");
        assert_eq!(format_lr(0.001), "0.00100");
        assert_eq!(format_lr(0.0001), "0.000100");
    }

    #[test]
    fn test_epoch_summaries() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 4,
            loss_history: vec![10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
            ..Default::default()
        };

        let summaries = compute_epoch_summaries(&snapshot);
        assert_eq!(summaries.len(), 3);
        assert!((summaries[0].avg_loss - 9.25).abs() < 0.01);
        assert!((summaries[0].min_loss - 8.5).abs() < 0.01);
        assert!((summaries[0].max_loss - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_history_table_render() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 4,
            loss_history: vec![10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
            tokens_per_second: 100.0,
            learning_rate: 0.0001,
            gradient_norm: 2.5,
            ..Default::default()
        };

        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        assert!(table.contains("Epoch"));
        assert!(table.contains("Loss"));
    }

    #[test]
    fn test_history_table_empty() {
        let snapshot = TrainingSnapshot::default();
        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        assert!(table.contains("waiting for epoch data"));
    }

    #[test]
    fn test_layout_renders() {
        let snapshot = TrainingSnapshot {
            epoch: 5,
            total_epochs: 10,
            step: 8,
            steps_per_epoch: 16,
            loss: 2.5,
            loss_history: vec![5.0, 4.5, 4.0, 3.5, 3.0, 2.8, 2.6, 2.5],
            learning_rate: 0.0001,
            gradient_norm: 1.5,
            tokens_per_second: 100.0,
            model_name: "TestModel".to_string(),
            optimizer_name: "AdamW".to_string(),
            batch_size: 4,
            ..Default::default()
        };

        let layout = render_layout(&snapshot, 80);
        assert!(layout.contains("ENTRENAR"));
        assert!(layout.contains("Epoch"));
        assert!(layout.contains("Loss"));
        assert!(layout.contains("Step"));
    }

    #[test]
    fn test_render_header_initializing() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Initializing,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Init"));
    }

    #[test]
    fn test_render_header_running() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Running,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Running"));
    }

    #[test]
    fn test_render_header_paused() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Paused,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Paused"));
    }

    #[test]
    fn test_render_header_completed() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Completed,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Done"));
    }

    #[test]
    fn test_render_header_failed() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Failed("out of memory".to_string()),
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("FAIL"));
    }
}

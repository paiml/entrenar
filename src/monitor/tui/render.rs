//! TUI Rendering - Tufte/Zen/Material Design Principles
//!
//! Design philosophy:
//! - Tufte: Maximize data-ink ratio, eliminate chartjunk
//! - Zen: Simplicity, asymmetric balance, tranquility
//! - Material: Clear hierarchy, meaningful density
//!
//! Layout:
//! ```text
//! ╭─────────────────────────────────────────────────────────────────────────────╮
//! │ ENTRENAR                              ● Running 00:04:12         67 tok/s   │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │ Epoch 7/18 ████████████░░░░░░░ 39%    Loss 6.91 ↓ best:4.79   LR 0.00047   │
//! │ Step 13/16 ██████████████████░ 81%    Grad 11.3               ETA 00:02:15  │
//! ├──────────────────────────────────┬──────────────────────────────────────────┤
//! │ LOSS ⣿⣿⣿⣷⣶⣴⣤⣄⣀⡀ 4.79→6.91      │ GPU RTX 4090  ████████████░░░░ 62% 42°C │
//! │ min 4.79  max 19.0  avg 10.2     │ VRAM ██░░░░░░░░░░░░░░ 4.1G/24G 17%      │
//! ├──────────────────────────────────┴──────────────────────────────────────────┤
//! │  E  Loss    LR      E  Loss    LR    │ Qwen2.5-Coder-0.5B  AdamW  batch:1   │
//! │  1  6.91  0.00020   5 13.38  0.00047 │ ./experiments/finetune-real          │
//! │  2  6.91  0.00018   6 12.20  0.00053 │ finetune_real                        │
//! │  3  6.91  0.00015   7 11.69  0.00047 │                                      │
//! ╰─────────────────────────────────────────────────────────────────────────────╯
//! ```

use super::color::{ColorMode, Styled, TrainingPalette};
use super::state::{TrainingSnapshot, TrainingStatus};
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const BLOCK_FULL: char = '█';
const BLOCK_LIGHT: char = '░';

const ARROW_UP: &str = "↑";
const ARROW_DOWN: &str = "↓";
const ARROW_FLAT: &str = "→";

const STATUS_RUNNING: &str = "●";
const STATUS_DONE: &str = "○";
const STATUS_WARN: &str = "◐";

const BRAILLE_BASE: u32 = 0x2800;
const BRAILLE_DOTS: [u32; 8] = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];

const BOX_TL: &str = "╭";
const BOX_TR: &str = "╮";
const BOX_BL: &str = "╰";
const BOX_BR: &str = "╯";
const BOX_H: &str = "─";
const BOX_V: &str = "│";
const BOX_T: &str = "┬";
const BOX_B: &str = "┴";
const BOX_L: &str = "├";
const BOX_R: &str = "┤";
#[allow(dead_code)]
const BOX_X: &str = "┼";

// ═══════════════════════════════════════════════════════════════════════════════
// CORE RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Build block bar: ████████░░░░
pub fn build_block_bar(percent: f32, width: usize) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f32) as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{}",
        BLOCK_FULL.to_string().repeat(filled),
        BLOCK_LIGHT.to_string().repeat(empty)
    )
}

/// Build colored block bar with gradient
pub fn build_colored_block_bar(percent: f32, width: usize, color_mode: ColorMode) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f32) as usize;
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

/// Gradient: cyan→green→yellow→orange→red
fn pct_color(pct: f32) -> (u8, u8, u8) {
    let p = pct.clamp(0.0, 100.0);
    if p >= 90.0 {
        (255, 64, 64)
    } else if p >= 75.0 {
        let t = (p - 75.0) / 15.0;
        (255, (180.0 - t * 116.0) as u8, 64)
    } else if p >= 50.0 {
        let t = (p - 50.0) / 25.0;
        (255, (220.0 - t * 40.0) as u8, 64)
    } else if p >= 25.0 {
        let t = (p - 25.0) / 25.0;
        ((100.0 + t * 155.0) as u8, 220, (100.0 - t * 36.0) as u8)
    } else {
        let t = p / 25.0;
        (
            (64.0 + t * 36.0) as u8,
            (180.0 + t * 40.0) as u8,
            (220.0 - t * 120.0) as u8,
        )
    }
}

/// Render sparkline using braille
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
            (((v1 - min) / range) * 3.99) as usize
        } else {
            0
        };
        let h2 = if v2.is_finite() {
            (((v2 - min) / range) * 3.99) as usize
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

/// Trend arrow from data
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
        format!("{:.1}G", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.0}M", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0}K", bytes as f64 / 1024.0)
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

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}

fn strip_ansi_width(s: &str) -> usize {
    let mut width = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else {
            width += 1;
        }
    }
    width
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
// MAIN LAYOUT (Tufte/Zen/Material)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn render_layout(snapshot: &TrainingSnapshot, width: usize) -> String {
    render_layout_colored(snapshot, width, ColorMode::detect())
}

pub fn render_layout_colored(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut out = String::new();
    let w = width.max(60);
    let half = w / 2;

    // ─────────────────────────────────────────────────────────────────────────
    // HEADER: Title + Status + Throughput (single dense line)
    // ─────────────────────────────────────────────────────────────────────────
    out.push_str(&format!("{BOX_TL}{}{BOX_TR}\n", BOX_H.repeat(w - 2)));

    let (status_icon, status_text, status_color) = match &snapshot.status {
        TrainingStatus::Initializing => (STATUS_WARN, "Init", TrainingPalette::INFO),
        TrainingStatus::Running => (STATUS_RUNNING, "Running", TrainingPalette::SUCCESS),
        TrainingStatus::Paused => (STATUS_WARN, "Paused", TrainingPalette::WARNING),
        TrainingStatus::Completed => (STATUS_DONE, "Done", TrainingPalette::PRIMARY),
        TrainingStatus::Failed(_) => (STATUS_RUNNING, "FAIL", TrainingPalette::ERROR),
    };

    let elapsed = format_duration(snapshot.elapsed());
    let tps = snapshot.tokens_per_second.max(0.0);
    let tps_str = if tps > 0.0 {
        format!("{tps:.0} tok/s")
    } else {
        String::new()
    };

    let title = Styled::new("ENTRENAR", color_mode).fg(TrainingPalette::PRIMARY);
    let status_text_full = format!("{status_icon} {status_text} {elapsed}");
    let status = Styled::new(&status_text_full, color_mode).fg(status_color);
    let throughput = Styled::new(&tps_str, color_mode).fg(TrainingPalette::INFO);

    let header_content = format!("{title}");
    let status_content = format!("{status}");
    let tps_content = format!("{throughput}");

    let header_vis = 8; // "ENTRENAR"
    let status_vis = strip_ansi_width(&format!("{status_icon} {status_text} {elapsed}"));
    let tps_vis = tps_str.len();
    let gap = w.saturating_sub(header_vis + status_vis + tps_vis + 6);

    out.push_str(&format!(
        "{BOX_V} {header_content}{}{status_content}   {tps_content} {BOX_V}\n",
        " ".repeat(gap)
    ));

    // ─────────────────────────────────────────────────────────────────────────
    // PROGRESS ROW: Epoch/Step bars + Key metrics (highest information density)
    // ─────────────────────────────────────────────────────────────────────────
    out.push_str(&format!("{BOX_L}{}{BOX_R}\n", BOX_H.repeat(w - 2)));

    // Epoch progress
    let epoch = snapshot.epoch.min(snapshot.total_epochs);
    let epoch_pct = if snapshot.total_epochs > 0 {
        (epoch as f32 / snapshot.total_epochs as f32 * 100.0).min(100.0)
    } else {
        0.0
    };
    let epoch_bar_w = 18;
    let epoch_bar = build_colored_block_bar(epoch_pct, epoch_bar_w, color_mode);

    // Loss with trend
    let loss_str = if snapshot.loss.is_finite() {
        format!("{:.2}", snapshot.loss)
    } else {
        "???".to_string()
    };
    let loss_color = if snapshot.loss.is_finite() {
        pct_color((snapshot.loss * 10.0).min(100.0))
    } else {
        (255, 64, 64)
    };
    let loss_trend = trend_arrow(&snapshot.loss_history);
    let best = snapshot
        .loss_history
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let best_str = if best.is_finite() {
        format!("{best:.2}")
    } else {
        "---".to_string()
    };

    // LR
    let lr_str = format_lr(snapshot.learning_rate);

    let epoch_label = format!("Epoch {}/{}", epoch, snapshot.total_epochs);
    let loss_text = format!("Loss {loss_str}");
    let best_text = format!("best:{best_str}");
    let lr_text = format!("LR {lr_str}");
    let loss_label = Styled::new(&loss_text, color_mode).fg(loss_color);
    let best_label = Styled::new(&best_text, color_mode).fg((120, 200, 120));
    let lr_label = Styled::new(&lr_text, color_mode).fg(TrainingPalette::INFO);

    let line1_left = format!("{epoch_label} {epoch_bar} {epoch_pct:.0}%");
    let line1_right = format!("{loss_label} {loss_trend} {best_label}   {lr_label}");
    let line1_left_vis = epoch_label.len() + 1 + epoch_bar_w + 5;
    let line1_right_vis = 6 + loss_str.len() + 2 + 5 + best_str.len() + 6 + lr_str.len();
    let line1_gap = w.saturating_sub(line1_left_vis + line1_right_vis + 4);

    out.push_str(&format!(
        "{BOX_V} {line1_left}{}{line1_right} {BOX_V}\n",
        " ".repeat(line1_gap)
    ));

    // Step progress
    let step = snapshot.step.min(snapshot.steps_per_epoch);
    let step_pct = if snapshot.steps_per_epoch > 0 {
        (step as f32 / snapshot.steps_per_epoch as f32 * 100.0).min(100.0)
    } else {
        0.0
    };
    let step_bar = build_colored_block_bar(step_pct, epoch_bar_w, color_mode);

    // Grad + ETA
    let grad = snapshot.gradient_norm.max(0.0);
    let grad_color = if grad > 10.0 {
        TrainingPalette::ERROR
    } else if grad > 5.0 {
        TrainingPalette::WARNING
    } else {
        TrainingPalette::SUCCESS
    };
    let eta = snapshot
        .estimated_remaining()
        .map_or("--:--:--".to_string(), format_duration);

    let step_label = format!("Step {}/{}", step, snapshot.steps_per_epoch);
    let grad_text = format!("Grad {grad:.1}");
    let eta_text = format!("ETA {eta}");
    let grad_label = Styled::new(&grad_text, color_mode).fg(grad_color);
    let eta_label = Styled::new(&eta_text, color_mode).fg((150, 150, 150));

    // Pad step label to match epoch label width
    let step_label_padded = format!("{step_label:width$}", width = epoch_label.len());

    let line2_left = format!("{step_label_padded} {step_bar} {step_pct:.0}%");
    let line2_right = format!("{grad_label}               {eta_label}");
    let line2_left_vis = step_label_padded.len() + 1 + epoch_bar_w + 5;
    let line2_right_vis = 6 + format!("{grad:.1}").len() + 15 + 4 + eta.len();
    let line2_gap = w.saturating_sub(line2_left_vis + line2_right_vis + 4);

    out.push_str(&format!(
        "{BOX_V} {line2_left}{}{line2_right} {BOX_V}\n",
        " ".repeat(line2_gap)
    ));

    // ─────────────────────────────────────────────────────────────────────────
    // MIDDLE ROW: Loss sparkline | GPU stats
    // ─────────────────────────────────────────────────────────────────────────
    out.push_str(&format!(
        "{BOX_L}{}{BOX_T}{}{BOX_R}\n",
        BOX_H.repeat(half - 1),
        BOX_H.repeat(w - half - 2)
    ));

    // Left: Loss panel
    let loss_title = Styled::new("LOSS", color_mode).fg(TrainingPalette::PRIMARY);
    let spark_w = half.saturating_sub(25);
    let sparkline = if snapshot.loss_history.is_empty() {
        " ".repeat(spark_w)
    } else {
        render_sparkline(&snapshot.loss_history, spark_w, color_mode)
    };

    let valid_hist: Vec<f32> = snapshot
        .loss_history
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let (hist_min, hist_max) = if valid_hist.is_empty() {
        (0.0, 0.0)
    } else {
        (
            valid_hist.iter().copied().fold(f32::INFINITY, f32::min),
            valid_hist.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    };

    let loss_range = if hist_min.is_finite() && hist_max.is_finite() {
        format!("{hist_min:.2}→{hist_max:.2}")
    } else {
        "---".to_string()
    };

    let left1 = format!("{loss_title} {sparkline} {loss_range}");

    // Right: GPU panel
    let gpu_line1 = if let Some(gpu) = &snapshot.gpu {
        let name: String = gpu.device_name.chars().take(12).collect();
        let util_bar_w = 14;
        let util_bar = build_colored_block_bar(gpu.utilization_percent, util_bar_w, color_mode);
        let temp_color = if gpu.temperature_celsius > 80.0 {
            TrainingPalette::ERROR
        } else if gpu.temperature_celsius > 70.0 {
            TrainingPalette::WARNING
        } else {
            TrainingPalette::SUCCESS
        };
        let temp_text = format!("{:.0}°C", gpu.temperature_celsius);
        let temp = Styled::new(&temp_text, color_mode).fg(temp_color);
        let gpu_label = Styled::new("GPU", color_mode).fg(TrainingPalette::SUCCESS);
        let name_label = Styled::new(&name, color_mode).fg((180, 180, 220));
        format!(
            "{gpu_label} {name_label} {util_bar}{:.0}% {temp}",
            gpu.utilization_percent
        )
    } else {
        Styled::new("GPU N/A", color_mode)
            .fg((100, 100, 100))
            .to_string()
    };

    let left1_vis = 5 + spark_w + 1 + loss_range.len();
    let right1_vis = if snapshot.gpu.is_some() {
        4 + 12 + 14 + 5 + 5
    } else {
        6
    };
    let l1_pad = (half - 4).saturating_sub(left1_vis);
    let r1_pad = (w - half - 4).saturating_sub(right1_vis);

    out.push_str(&format!(
        "{BOX_V} {left1}{} {BOX_V} {gpu_line1}{} {BOX_V}\n",
        " ".repeat(l1_pad),
        " ".repeat(r1_pad)
    ));

    // Second line: stats | VRAM
    let avg = if valid_hist.is_empty() {
        0.0
    } else {
        valid_hist.iter().sum::<f32>() / valid_hist.len() as f32
    };
    let stats_str = if hist_min.is_finite() {
        format!("min {hist_min:.2}  max {hist_max:.2}  avg {avg:.2}")
    } else {
        "(waiting for data...)".to_string()
    };
    let left2 = Styled::new(&stats_str, color_mode)
        .fg((140, 140, 140))
        .to_string();

    let gpu_line2 = if let Some(gpu) = &snapshot.gpu {
        let vram_pct = gpu.vram_percent().min(100.0);
        let vram_bar_w = 14;
        let vram_bar = build_colored_block_bar(vram_pct, vram_bar_w, color_mode);
        let vram_used = gpu.vram_used_gb.min(gpu.vram_total_gb);
        format!(
            "VRAM {vram_bar} {vram_used:.1}G/{:.0}G {vram_pct:.0}%",
            gpu.vram_total_gb
        )
    } else {
        String::new()
    };

    let left2_vis = strip_ansi_width(&left2);
    let right2_vis = if snapshot.gpu.is_some() {
        5 + 14 + 15
    } else {
        0
    };
    let l2_pad = (half - 4).saturating_sub(left2_vis);
    let r2_pad = (w - half - 4).saturating_sub(right2_vis);

    out.push_str(&format!(
        "{BOX_V} {left2}{} {BOX_V} {gpu_line2}{} {BOX_V}\n",
        " ".repeat(l2_pad),
        " ".repeat(r2_pad)
    ));

    // ─────────────────────────────────────────────────────────────────────────
    // BOTTOM ROW: Epoch history table (multi-column) | Config
    // ─────────────────────────────────────────────────────────────────────────
    let hist_w = (w * 2) / 3;
    let conf_w = w - hist_w;

    out.push_str(&format!(
        "{BOX_L}{}{BOX_T}{}{BOX_R}\n",
        BOX_H.repeat(hist_w - 1),
        BOX_H.repeat(conf_w - 2)
    ));

    let summaries = compute_epoch_summaries(snapshot);
    let max_rows = 4;

    // Prepare config lines
    let model_name = if snapshot.model_name.is_empty() {
        "N/A"
    } else {
        &snapshot.model_name
    };
    let model_display: String = model_name.chars().take(conf_w - 6).collect();
    let opt = if snapshot.optimizer_name.is_empty() {
        "N/A"
    } else {
        &snapshot.optimizer_name
    };
    let batch = if snapshot.batch_size > 0 {
        format!("batch:{}", snapshot.batch_size)
    } else {
        String::new()
    };
    let path_display = if snapshot.checkpoint_path.is_empty() {
        format!("./experiments/{}/", snapshot.experiment_id)
    } else {
        truncate_path(&snapshot.checkpoint_path, conf_w - 6)
    };
    let exe_display = if snapshot.executable_path.is_empty() {
        "N/A".to_string()
    } else {
        let exe_name = snapshot
            .executable_path
            .split('/')
            .next_back()
            .unwrap_or(&snapshot.executable_path);
        exe_name.chars().take(conf_w - 6).collect()
    };

    let conf_lines = vec![
        Styled::new(&model_display, color_mode)
            .fg((180, 180, 255))
            .to_string(),
        format!(
            "{}  {}",
            Styled::new(opt, color_mode).fg((150, 255, 150)),
            Styled::new(&batch, color_mode).fg((150, 150, 150))
        ),
        Styled::new(&path_display, color_mode)
            .fg((150, 200, 150))
            .to_string(),
        Styled::new(&exe_display, color_mode)
            .fg((150, 150, 180))
            .to_string(),
    ];

    // Render epoch history in two columns
    if summaries.is_empty() {
        // No data yet
        for (i, conf_line) in conf_lines.iter().take(max_rows).enumerate() {
            let left = if i == 0 {
                Styled::new("(waiting for epoch data...)", color_mode)
                    .fg((100, 100, 100))
                    .to_string()
            } else {
                String::new()
            };
            let left_vis = strip_ansi_width(&left);
            let conf_vis = strip_ansi_width(conf_line);
            let l_pad = (hist_w - 4).saturating_sub(left_vis);
            let r_pad = (conf_w - 4).saturating_sub(conf_vis);
            out.push_str(&format!(
                "{BOX_V} {left}{} {BOX_V} {conf_line}{} {BOX_V}\n",
                " ".repeat(l_pad),
                " ".repeat(r_pad)
            ));
        }
    } else {
        // Split epochs into two columns
        let col_w = (hist_w - 8) / 2;
        let start_idx = summaries.len().saturating_sub(max_rows * 2);
        let visible: Vec<&EpochSummary> = summaries.iter().skip(start_idx).collect();

        for row in 0..max_rows {
            let idx1 = row;
            let idx2 = row + max_rows;

            let fmt_epoch = |s: &EpochSummary| -> String {
                let loss_color = pct_color((s.avg_loss * 8.0).min(100.0));
                format!(
                    "{:>2} {} {}",
                    Styled::new(&format!("{}", s.epoch), color_mode).fg((180, 180, 220)),
                    Styled::new(&format!("{:>6.2}", s.avg_loss), color_mode).fg(loss_color),
                    Styled::new(&format_lr(s.lr).to_string(), color_mode).fg((100, 180, 220)),
                )
            };

            let col1 = visible.get(idx1).map(|s| fmt_epoch(s)).unwrap_or_default();
            let col2 = visible.get(idx2).map(|s| fmt_epoch(s)).unwrap_or_default();

            let col1_vis = if visible.get(idx1).is_some() {
                3 + 7 + 8
            } else {
                0
            };
            let col2_vis = if visible.get(idx2).is_some() {
                3 + 7 + 8
            } else {
                0
            };

            let col1_pad = col_w.saturating_sub(col1_vis);
            let col2_pad = col_w.saturating_sub(col2_vis);

            let conf_line = conf_lines.get(row).cloned().unwrap_or_default();
            let conf_vis = strip_ansi_width(&conf_line);
            let r_pad = (conf_w - 4).saturating_sub(conf_vis);

            out.push_str(&format!(
                "{BOX_V} {col1}{} {col2}{} {BOX_V} {conf_line}{} {BOX_V}\n",
                " ".repeat(col1_pad),
                " ".repeat(col2_pad),
                " ".repeat(r_pad)
            ));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FOOTER
    // ─────────────────────────────────────────────────────────────────────────
    out.push_str(&format!(
        "{BOX_BL}{}{BOX_B}{}{BOX_BR}\n",
        BOX_H.repeat(hist_w - 1),
        BOX_H.repeat(conf_w - 2)
    ));

    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY (for probar tests)
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

    let path_display = if snapshot.checkpoint_path.is_empty() {
        "N/A".to_string()
    } else {
        truncate_path(&snapshot.checkpoint_path, width - 4)
    };
    lines.push(path_display);

    let exe_display = if snapshot.executable_path.is_empty() {
        "N/A".to_string()
    } else {
        truncate_path(&snapshot.executable_path, width - 4)
    };
    lines.push(exe_display);

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
    lines.push(BOX_H.repeat(width.min(70)));

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

// Legacy gauge
pub fn render_gauge(value: f32, max: f32, width: usize, label: &str) -> String {
    let percent = if max > 0.0 { value / max * 100.0 } else { 0.0 };
    let bar = build_block_bar(percent, width.saturating_sub(label.len() + 8));
    format!("{label}{bar} {percent:>5.1}%")
}

// Legacy braille chart
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

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLE PREVIEW (retained for backwards compat, but minimal)
// ═══════════════════════════════════════════════════════════════════════════════

use super::state::SamplePeek;

pub fn render_sample_panel(
    _sample: Option<&SamplePeek>,
    _width: usize,
    _color_mode: ColorMode,
) -> String {
    // Intentionally minimal - sample preview during training is noise
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
    fn test_strip_ansi_width() {
        assert_eq!(strip_ansi_width("hello"), 5);
        assert_eq!(strip_ansi_width("\x1b[31mred\x1b[0m"), 3);
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
        assert!(layout.contains("LOSS"));
        assert!(layout.contains("Epoch"));
        assert!(layout.contains("Step"));
    }
}

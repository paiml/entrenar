//! TUI Rendering Components - 10/10 ptop-style (SPEC-FT-001 Section 10.1)
//!
//! btop/ptop-inspired visualization with:
//! - Block bars (█░) with gradient colors
//! - Braille sparklines for time series
//! - Process display with full path, CPU%, RSS, GPU memory
//! - Trend arrows (↑↓→) for metrics
//! - Unicode status symbols (●◐◔○⚡)

use super::color::{ColorMode, Styled, TrainingPalette};
use super::state::{GpuTelemetry, SamplePeek, TrainingSnapshot, TrainingStatus};
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// UNICODE BLOCK CHARACTERS & SYMBOLS
// ═══════════════════════════════════════════════════════════════════════════════

const BLOCK_FULL: char = '█';
const BLOCK_LIGHT: char = '░';

// Trend arrows
const ARROW_UP: &str = "↑";
const ARROW_DOWN: &str = "↓";
const ARROW_FLAT: &str = "→";

// Status indicators
const STATUS_CRITICAL: &str = "●";
const STATUS_WARNING: &str = "◐";
const STATUS_OK: &str = "◔";
const STATUS_GOOD: &str = "○";
const BOLT: &str = "⚡";

// Braille base for sparklines
const BRAILLE_BASE: u32 = 0x2800;
const BRAILLE_DOTS: [u32; 8] = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];

// Box drawing
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
const BOX_X: &str = "┼";

// ═══════════════════════════════════════════════════════════════════════════════
// BLOCK BAR RENDERING (btop-style)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a block-style load bar: ████████░░░░
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

/// Build a colored block bar with btop-style gradient
pub fn build_colored_block_bar(percent: f32, width: usize, color_mode: ColorMode) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f32) as usize;
    let empty = width.saturating_sub(filled);

    let color = percent_to_color(pct);
    let filled_str = BLOCK_FULL.to_string().repeat(filled);
    let empty_str = BLOCK_LIGHT.to_string().repeat(empty);

    if color_mode == ColorMode::Mono {
        format!("{filled_str}{empty_str}")
    } else {
        format!(
            "{}{}",
            Styled::new(&filled_str, color_mode).fg(color),
            Styled::new(&empty_str, color_mode).fg((80, 80, 80))
        )
    }
}

/// btop-style color gradient: cyan→green→yellow→orange→red
fn percent_to_color(pct: f32) -> (u8, u8, u8) {
    let p = pct.clamp(0.0, 100.0);

    if p >= 90.0 {
        // Critical: bright red
        (255, 64, 64)
    } else if p >= 75.0 {
        // High: orange to red
        let t = (p - 75.0) / 15.0;
        (255, (180.0 - t * 116.0) as u8, 64)
    } else if p >= 50.0 {
        // Medium: yellow to orange
        let t = (p - 50.0) / 25.0;
        (255, (220.0 - t * 40.0) as u8, 64)
    } else if p >= 25.0 {
        // Low-medium: green to yellow
        let t = (p - 25.0) / 25.0;
        ((100.0 + t * 155.0) as u8, 220, (100.0 - t * 36.0) as u8)
    } else {
        // Low: cyan to green
        let t = p / 25.0;
        (
            (64.0 + t * 36.0) as u8,
            (180.0 + t * 40.0) as u8,
            (220.0 - t * 120.0) as u8,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPARKLINE RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Render a compact sparkline using braille characters
pub fn render_sparkline(data: &[f32], width: usize, color_mode: ColorMode) -> String {
    if data.is_empty() {
        return " ".repeat(width);
    }

    // Find min/max for normalization
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(0.001);

    // Sample data to fit width (each braille char = 2 columns)
    let chars_needed = width;
    let mut result = String::new();

    for i in 0..chars_needed {
        let idx = (i * data.len()) / chars_needed.max(1);
        let idx2 = ((i * 2 + 1) * data.len()) / (chars_needed * 2).max(1);

        let v1 = data.get(idx).copied().unwrap_or(min);
        let v2 = data.get(idx2).copied().unwrap_or(v1);

        // Normalize to 0-3 (4 vertical positions in braille)
        let h1 = (((v1 - min) / range) * 3.99) as usize;
        let h2 = (((v2 - min) / range) * 3.99) as usize;

        // Build braille char
        let mut code: u32 = 0;
        // Left column dots (positions 0,1,2,3 = dots 1,2,3,7)
        for y in 0..=h1.min(3) {
            code |= BRAILLE_DOTS[3 - y]; // invert for visual
        }
        // Right column dots (positions 4,5,6,7 = dots 4,5,6,8)
        for y in 0..=h2.min(3) {
            code |= BRAILLE_DOTS[7 - y];
        }

        let ch = char::from_u32(BRAILLE_BASE + code).unwrap_or('⣿');
        result.push(ch);
    }

    // Color based on trend (last vs first)
    let trend_color = if data.len() > 1 {
        let first = data.first().copied().unwrap_or(0.0);
        let last = data.last().copied().unwrap_or(0.0);
        if last < first * 0.95 {
            TrainingPalette::SUCCESS // decreasing loss = good
        } else if last > first * 1.05 {
            TrainingPalette::ERROR // increasing = bad
        } else {
            TrainingPalette::INFO // stable
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

/// Compute trend arrow based on recent values
pub fn trend_arrow(data: &[f32]) -> &'static str {
    if data.len() < 2 {
        return ARROW_FLAT;
    }
    let recent = data.iter().rev().take(5).copied().collect::<Vec<_>>();
    if recent.len() < 2 {
        return ARROW_FLAT;
    }
    let avg_recent: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
    let avg_old: f32 = data.iter().rev().skip(5).take(5).copied().sum::<f32>()
        / 5.0_f32.min(data.len().saturating_sub(5) as f32).max(1.0);

    if avg_recent < avg_old * 0.95 {
        ARROW_DOWN
    } else if avg_recent > avg_old * 1.05 {
        ARROW_UP
    } else {
        ARROW_FLAT
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMATTING HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Format duration as HH:MM:SS
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{hours:02}:{mins:02}:{secs:02}")
}

/// Format bytes as human readable (G/M/K)
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

/// Format learning rate (avoid scientific notation)
pub fn format_lr(lr: f32) -> String {
    if lr >= 0.01 {
        format!("{lr:.4}")
    } else if lr >= 0.001 {
        format!("{lr:.5}")
    } else {
        format!("{lr:.6}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN LAYOUT RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Render the full TUI layout (10/10 style)
pub fn render_layout(snapshot: &TrainingSnapshot, width: usize) -> String {
    render_layout_colored(snapshot, width, ColorMode::detect())
}

/// Render the full TUI layout with specified color mode
///
/// Layout:
/// ```text
/// ╭──────────────────────────────────────────────────────────────────────────────╮
/// │ ⚡ ENTRENAR v0.5.6                              [● Running: 00:04:12] 67 tok/s│
/// ├─────────────────────────────────┬────────────────────────────────────────────┤
/// │ 📉 LOSS CURVE                   │ 🖥️  GPU: RTX 4090                          │
/// │ ⣿⣿⣿⣷⣶⣴⣤⣄⣀⡀ 6.90→2.31 ↓     │ ████████████░░░░░░ 67% 64°C 285W          │
/// │ min:2.31 max:6.90 avg:4.12      │ VRAM ██████░░░░░░░░░░ 3.4G/24G 14%         │
/// │ ⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀ grad_norm        │ ──────────────────────────────────────────│
/// │                                 │ 📊 PROCESS                                 │
/// │                                 │ /mnt/.../finetune_real                     │
/// │                                 │ CPU 89% ████████░░ RSS 3.4G GPU 2.8G       │
/// ├─────────────────────────────────┼────────────────────────────────────────────┤
/// │ 📝 SAMPLE PREVIEW               │ 📈 TRAINING METRICS                        │
/// │ In: fn is_prime(n: u64)...      │ Epoch ████████████████░░░░ 15/18 83%       │
/// │ Tgt: #[test] fn test_is_prime() │ Step  ████████████░░░░░░░░ 12/16 75%       │
/// │ Gen: #[test] fn test_is_prime() │ LR 0.00058 ↓  Grad 2.31 ↓                  │
/// │ Match: ████████████░░░░ 78%     │ Loss 2.31 ↓ best:2.15 ETA 00:02:15         │
/// ╰─────────────────────────────────┴────────────────────────────────────────────╯
/// ```
pub fn render_layout_colored(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut output = String::new();
    let half_width = width / 2;

    // ─────────────────────────────────────────────────────────────────────────
    // HEADER
    // ─────────────────────────────────────────────────────────────────────────
    output.push_str(&render_header(snapshot, width, color_mode));

    // ─────────────────────────────────────────────────────────────────────────
    // TOP ROW: Loss Curve | GPU + Process
    // ─────────────────────────────────────────────────────────────────────────
    output.push_str(&format!(
        "{BOX_L}{}{BOX_T}{}{BOX_R}\n",
        BOX_H.repeat(half_width - 1),
        BOX_H.repeat(width - half_width - 2)
    ));

    let loss_panel = render_loss_panel(snapshot, half_width - 4, color_mode);
    let gpu_panel = render_gpu_panel(snapshot.gpu.as_ref(), width - half_width - 4, color_mode);

    let loss_lines: Vec<&str> = loss_panel.lines().collect();
    let gpu_lines: Vec<&str> = gpu_panel.lines().collect();
    let max_lines = loss_lines.len().max(gpu_lines.len()).max(7);

    for i in 0..max_lines {
        let left = loss_lines.get(i).copied().unwrap_or("");
        let right = gpu_lines.get(i).copied().unwrap_or("");
        let left_visual = strip_ansi_width(left);
        let right_visual = strip_ansi_width(right);
        let left_pad = (half_width - 4).saturating_sub(left_visual);
        let right_pad = (width - half_width - 4).saturating_sub(right_visual);

        output.push_str(&format!(
            "{BOX_V} {}{} {BOX_V} {}{} {BOX_V}\n",
            left,
            " ".repeat(left_pad),
            right,
            " ".repeat(right_pad)
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BOTTOM ROW: Sample Preview | Training Metrics
    // ─────────────────────────────────────────────────────────────────────────
    output.push_str(&format!(
        "{BOX_L}{}{BOX_X}{}{BOX_R}\n",
        BOX_H.repeat(half_width - 1),
        BOX_H.repeat(width - half_width - 2)
    ));

    let sample_panel = render_sample_panel(snapshot.sample.as_ref(), half_width - 4, color_mode);
    let metrics_panel = render_metrics_panel(snapshot, width - half_width - 4, color_mode);

    let sample_lines: Vec<&str> = sample_panel.lines().collect();
    let metrics_lines: Vec<&str> = metrics_panel.lines().collect();
    let max_lines2 = sample_lines.len().max(metrics_lines.len()).max(5);

    for i in 0..max_lines2 {
        let left = sample_lines.get(i).copied().unwrap_or("");
        let right = metrics_lines.get(i).copied().unwrap_or("");
        let left_visual = strip_ansi_width(left);
        let right_visual = strip_ansi_width(right);
        let left_pad = (half_width - 4).saturating_sub(left_visual);
        let right_pad = (width - half_width - 4).saturating_sub(right_visual);

        output.push_str(&format!(
            "{BOX_V} {}{} {BOX_V} {}{} {BOX_V}\n",
            left,
            " ".repeat(left_pad),
            right,
            " ".repeat(right_pad)
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FOOTER
    // ─────────────────────────────────────────────────────────────────────────
    output.push_str(&format!(
        "{BOX_BL}{}{BOX_B}{}{BOX_BR}\n",
        BOX_H.repeat(half_width - 1),
        BOX_H.repeat(width - half_width - 2)
    ));

    output
}

// ═══════════════════════════════════════════════════════════════════════════════
// PANEL RENDERERS
// ═══════════════════════════════════════════════════════════════════════════════

fn render_header(snapshot: &TrainingSnapshot, width: usize, color_mode: ColorMode) -> String {
    let mut output = String::new();

    // Status indicator and color
    let (status_icon, status_text, status_color) = match &snapshot.status {
        TrainingStatus::Initializing => (STATUS_OK, "Init", TrainingPalette::INFO),
        TrainingStatus::Running => (STATUS_CRITICAL, "Running", TrainingPalette::SUCCESS),
        TrainingStatus::Paused => (STATUS_WARNING, "Paused", TrainingPalette::WARNING),
        TrainingStatus::Completed => (STATUS_GOOD, "Done", TrainingPalette::PRIMARY),
        TrainingStatus::Failed(_) => (STATUS_CRITICAL, "FAIL", TrainingPalette::ERROR),
    };

    let elapsed = format_duration(snapshot.elapsed());
    let tps = if snapshot.tokens_per_second > 0.0 {
        format!("{:.0} tok/s", snapshot.tokens_per_second)
    } else {
        String::new()
    };

    let status_str = format!("[{status_icon} {status_text}: {elapsed}]");
    let colored_status = Styled::new(&status_str, color_mode)
        .fg(status_color)
        .to_string();

    let title = format!("{BOLT} ENTRENAR v0.5.6");
    let colored_title = Styled::new(&title, color_mode)
        .fg(TrainingPalette::PRIMARY)
        .to_string();

    // Calculate spacing
    let title_len = title.len();
    let status_len = status_str.len();
    let tps_len = tps.len();
    let content_len = title_len + status_len + tps_len + 4;
    let padding = width.saturating_sub(content_len + 4);

    output.push_str(&format!("{BOX_TL}{}{BOX_TR}\n", BOX_H.repeat(width - 2)));
    let colored_tps = Styled::new(&tps, color_mode).fg(TrainingPalette::INFO);
    output.push_str(&format!(
        "{BOX_V} {colored_title}{}{colored_status} {colored_tps} {BOX_V}\n",
        " ".repeat(padding),
    ));

    output
}

fn render_loss_panel(snapshot: &TrainingSnapshot, width: usize, color_mode: ColorMode) -> String {
    let mut lines = Vec::new();

    // Panel title
    let title = "📉 LOSS CURVE";
    lines.push(
        Styled::new(title, color_mode)
            .fg(TrainingPalette::PRIMARY)
            .to_string(),
    );

    if snapshot.loss_history.is_empty() {
        lines.push("(waiting for data...)".to_string());
        return lines.join("\n");
    }

    // Sparkline with stats
    let spark_width = width.saturating_sub(15);
    let sparkline = render_sparkline(&snapshot.loss_history, spark_width, color_mode);
    let trend = trend_arrow(&snapshot.loss_history);
    let current_loss = snapshot.loss;

    let loss_str = format!("{current_loss:.2}");
    let colored_loss = Styled::new(&loss_str, color_mode)
        .fg(percent_to_color((current_loss * 10.0).min(100.0)))
        .to_string();

    lines.push(format!("{sparkline} {colored_loss} {trend}"));

    // Min/max/avg stats
    let min = snapshot
        .loss_history
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max = snapshot
        .loss_history
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg: f32 = snapshot.loss_history.iter().sum::<f32>() / snapshot.loss_history.len() as f32;

    let stats = format!("min:{min:.2} max:{max:.2} avg:{avg:.2}");
    lines.push(
        Styled::new(&stats, color_mode)
            .fg((150, 150, 150))
            .to_string(),
    );

    // Gradient norm sparkline if we have history
    lines.push(String::new());
    let grad_title = "📊 GRADIENT NORM";
    lines.push(
        Styled::new(grad_title, color_mode)
            .fg(TrainingPalette::INFO)
            .to_string(),
    );

    let grad_bar_width = width.saturating_sub(12);
    let grad_pct = (snapshot.gradient_norm / 10.0 * 100.0).clamp(0.0, 100.0);
    let grad_bar = build_colored_block_bar(grad_pct, grad_bar_width, color_mode);
    let grad_arrow = if snapshot.gradient_norm > 5.0 {
        ARROW_UP
    } else if snapshot.gradient_norm < 1.0 {
        ARROW_DOWN
    } else {
        ARROW_FLAT
    };
    lines.push(format!(
        "{grad_bar} {:.2} {grad_arrow}",
        snapshot.gradient_norm
    ));

    lines.join("\n")
}

fn render_gpu_panel(gpu: Option<&GpuTelemetry>, width: usize, color_mode: ColorMode) -> String {
    let mut lines = Vec::new();

    if let Some(gpu) = gpu {
        // GPU title with name
        let title = format!("🖥️  GPU: {}", &gpu.device_name);
        let title_truncated: String = title.chars().take(width).collect();
        lines.push(
            Styled::new(&title_truncated, color_mode)
                .fg(TrainingPalette::SUCCESS)
                .to_string(),
        );

        // GPU utilization bar
        let util_bar_width = width.saturating_sub(20);
        let util_bar = build_colored_block_bar(gpu.utilization_percent, util_bar_width, color_mode);
        let temp_color = if gpu.temperature_celsius > 80.0 {
            TrainingPalette::ERROR
        } else if gpu.temperature_celsius > 70.0 {
            TrainingPalette::WARNING
        } else {
            TrainingPalette::SUCCESS
        };
        let temp_str = format!("{:.0}°C", gpu.temperature_celsius);
        let colored_temp = Styled::new(&temp_str, color_mode)
            .fg(temp_color)
            .to_string();
        lines.push(format!(
            "{util_bar} {:.0}% {colored_temp} {:.0}W",
            gpu.utilization_percent, gpu.power_watts
        ));

        // VRAM bar
        let vram_pct = gpu.vram_percent();
        let vram_bar_width = width.saturating_sub(25);
        let vram_bar = build_colored_block_bar(vram_pct, vram_bar_width, color_mode);
        lines.push(format!(
            "VRAM {vram_bar} {:.1}G/{:.0}G {:.0}%",
            gpu.vram_used_gb, gpu.vram_total_gb, vram_pct
        ));

        // Separator
        lines.push(BOX_H.repeat(width));

        // Process section
        let proc_title = "📊 PROCESS";
        lines.push(
            Styled::new(proc_title, color_mode)
                .fg(TrainingPalette::INFO)
                .to_string(),
        );

        if let Some(proc) = gpu.processes.first() {
            // Executable path (truncated from left to show end)
            let max_path = width.saturating_sub(2);
            let exe_display = if proc.exe_path.len() > max_path {
                format!(
                    "...{}",
                    &proc.exe_path[proc.exe_path.len() - max_path + 3..]
                )
            } else {
                proc.exe_path.clone()
            };
            lines.push(
                Styled::new(&exe_display, color_mode)
                    .fg((200, 200, 255))
                    .to_string(),
            );

            // Process stats with bars
            let cpu_bar_width = 10;
            let cpu_bar = build_colored_block_bar(proc.cpu_percent, cpu_bar_width, color_mode);
            let rss_str = format_bytes(proc.rss_mb * 1024 * 1024);
            let gpu_mem_str = format_bytes(proc.gpu_memory_mb * 1024 * 1024);

            lines.push(format!(
                "CPU {:.0}% {cpu_bar} RSS {} GPU {}",
                proc.cpu_percent, rss_str, gpu_mem_str
            ));
        } else {
            lines.push("(no process detected)".to_string());
            lines.push(String::new());
        }
    } else {
        lines.push(
            Styled::new("🖥️  GPU: N/A", color_mode)
                .fg((100, 100, 100))
                .to_string(),
        );
        lines.push("(GPU monitoring unavailable)".to_string());
        lines.push(String::new());
        lines.push(BOX_H.repeat(width));
        lines.push("📊 PROCESS".to_string());
        lines.push("(no GPU process)".to_string());
        lines.push(String::new());
    }

    lines.join("\n")
}

fn render_sample_panel(sample: Option<&SamplePeek>, width: usize, color_mode: ColorMode) -> String {
    let mut lines = Vec::new();

    let title = "📝 SAMPLE PREVIEW";
    lines.push(
        Styled::new(title, color_mode)
            .fg(TrainingPalette::PRIMARY)
            .to_string(),
    );

    let truncate = |s: &str, max_len: usize| -> String {
        if s.len() > max_len {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        } else {
            s.to_string()
        }
    };

    if let Some(sample) = sample {
        let max_preview = width.saturating_sub(5);

        let in_label = Styled::new("In:", color_mode)
            .fg((100, 200, 255))
            .to_string();
        lines.push(format!(
            "{in_label} {}",
            truncate(&sample.input_preview, max_preview)
        ));

        let tgt_label = Styled::new("Tgt:", color_mode)
            .fg((100, 255, 150))
            .to_string();
        lines.push(format!(
            "{tgt_label} {}",
            truncate(&sample.target_preview, max_preview)
        ));

        let gen_label = Styled::new("Gen:", color_mode)
            .fg((255, 200, 100))
            .to_string();
        lines.push(format!(
            "{gen_label} {}",
            truncate(&sample.generated_preview, max_preview)
        ));

        // Match percentage bar
        let match_bar_width = width.saturating_sub(15);
        let match_bar =
            build_colored_block_bar(sample.token_match_percent, match_bar_width, color_mode);
        lines.push(format!(
            "Match: {match_bar} {:.0}%",
            sample.token_match_percent
        ));
    } else {
        lines.push(
            Styled::new("(waiting for sample...)", color_mode)
                .fg((100, 100, 100))
                .to_string(),
        );
        lines.push(String::new());
        lines.push(String::new());
        lines.push(String::new());
    }

    lines.join("\n")
}

fn render_metrics_panel(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines = Vec::new();

    let title = "📈 TRAINING METRICS";
    lines.push(
        Styled::new(title, color_mode)
            .fg(TrainingPalette::PRIMARY)
            .to_string(),
    );

    // Epoch progress (clamp to 100% to handle edge cases)
    let epoch_pct = if snapshot.total_epochs > 0 {
        ((snapshot.epoch as f32 / snapshot.total_epochs as f32) * 100.0).min(100.0)
    } else {
        0.0
    };
    let epoch_bar_width = width.saturating_sub(25);
    let epoch_bar = build_colored_block_bar(epoch_pct, epoch_bar_width, color_mode);
    lines.push(format!(
        "Epoch {epoch_bar} {}/{} {:.0}%",
        snapshot.epoch, snapshot.total_epochs, epoch_pct
    ));

    // Step progress (clamp to 100% to handle edge cases)
    let step_pct = if snapshot.steps_per_epoch > 0 {
        ((snapshot.step as f32 / snapshot.steps_per_epoch as f32) * 100.0).min(100.0)
    } else {
        0.0
    };
    let step_bar_width = width.saturating_sub(25);
    let step_bar = build_colored_block_bar(step_pct, step_bar_width, color_mode);
    // Display step clamped to steps_per_epoch for sanity
    let display_step = snapshot.step.min(snapshot.steps_per_epoch);
    lines.push(format!(
        "Step  {step_bar} {}/{} {:.0}%",
        display_step, snapshot.steps_per_epoch, step_pct
    ));

    // LR and Gradient
    let lr_str = format_lr(snapshot.learning_rate);
    let lr_trend = trend_arrow(&snapshot.loss_history); // approximate LR trend from loss
    let grad_trend = if snapshot.gradient_norm > 5.0 {
        ARROW_UP
    } else {
        ARROW_DOWN
    };

    let lr_colored = Styled::new(&format!("LR {lr_str}"), color_mode)
        .fg(TrainingPalette::INFO)
        .to_string();
    let grad_colored = Styled::new(&format!("Grad {:.2}", snapshot.gradient_norm), color_mode)
        .fg(if snapshot.gradient_norm > 10.0 {
            TrainingPalette::ERROR
        } else if snapshot.gradient_norm > 5.0 {
            TrainingPalette::WARNING
        } else {
            TrainingPalette::SUCCESS
        })
        .to_string();

    lines.push(format!(
        "{lr_colored} {lr_trend}  {grad_colored} {grad_trend}"
    ));

    // Loss and ETA
    let loss_colored = Styled::new(&format!("Loss {:.3}", snapshot.loss), color_mode)
        .fg(percent_to_color((snapshot.loss * 10.0).min(100.0)))
        .to_string();

    let best_loss = snapshot
        .loss_history
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let eta = snapshot
        .estimated_remaining()
        .map_or_else(|| "--:--:--".to_string(), format_duration);

    lines.push(format!(
        "{loss_colored} {ARROW_DOWN} best:{best_loss:.2} ETA {eta}"
    ));

    lines.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate visual width of a string (excluding ANSI escape sequences)
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

// Legacy compatibility
pub fn render_gauge(value: f32, max: f32, width: usize, label: &str) -> String {
    let percent = if max > 0.0 { value / max * 100.0 } else { 0.0 };
    let bar = build_block_bar(percent, width.saturating_sub(label.len() + 8));
    format!("{label}{bar} {percent:>5.1}%")
}

/// Braille chart (legacy compatibility)
pub struct BrailleChart {
    width: usize,
    height: usize,
    data: Vec<f32>,
    log_scale: bool,
    color_mode: ColorMode,
}

impl BrailleChart {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: Vec::new(),
            log_scale: false,
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

    pub fn log_scale(mut self, enabled: bool) -> Self {
        self.log_scale = enabled;
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
    fn test_percent_to_color_gradient() {
        // Test gradient continuity
        let mut prev = percent_to_color(0.0);
        for i in 1..=100 {
            let curr = percent_to_color(i as f32);
            let dr = (curr.0 as i32 - prev.0 as i32).abs();
            let dg = (curr.1 as i32 - prev.1 as i32).abs();
            let db = (curr.2 as i32 - prev.2 as i32).abs();
            assert!(
                dr < 50 && dg < 50 && db < 50,
                "Color jump too large at {}%",
                i
            );
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
}

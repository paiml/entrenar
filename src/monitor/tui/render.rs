//! TUI Rendering Components (SPEC-FT-001 Section 10.1)
//!
//! Braille-encoded charts and ptop-style layout for training visualization.
//! Uses UTF-8 braille characters for high-resolution loss curves.

use super::state::{GpuTelemetry, SamplePeek, TrainingSnapshot, TrainingStatus};
use std::time::Duration;

/// Braille character mapping for 2x4 dot patterns
///
/// Each braille character encodes a 2x4 grid of dots.
/// Unicode range: U+2800 to U+28FF (256 characters)
const BRAILLE_BASE: u32 = 0x2800;

/// Braille dot positions (row-major):
/// ```text
/// 1 4
/// 2 5
/// 3 6
/// 7 8
/// ```
const BRAILLE_DOTS: [u32; 8] = [
    0x01, // dot 1 (top-left)
    0x02, // dot 2
    0x04, // dot 3
    0x40, // dot 7 (bottom-left)
    0x08, // dot 4 (top-right)
    0x10, // dot 5
    0x20, // dot 6
    0x80, // dot 8 (bottom-right)
];

/// Braille chart renderer
pub struct BrailleChart {
    /// Chart width in characters
    width: usize,
    /// Chart height in characters
    height: usize,
    /// Data points
    data: Vec<f32>,
    /// Minimum value (auto-scale if None)
    min_val: Option<f32>,
    /// Maximum value (auto-scale if None)
    max_val: Option<f32>,
    /// Use log scale
    log_scale: bool,
}

impl BrailleChart {
    /// Create a new braille chart
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: Vec::new(),
            min_val: None,
            max_val: None,
            log_scale: false,
        }
    }

    /// Set data points
    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = data;
        self
    }

    /// Set Y-axis bounds
    pub fn bounds(mut self, min: f32, max: f32) -> Self {
        self.min_val = Some(min);
        self.max_val = Some(max);
        self
    }

    /// Enable log scale for Y-axis
    pub fn log_scale(mut self, enabled: bool) -> Self {
        self.log_scale = enabled;
        self
    }

    /// Render the chart to a string
    pub fn render(&self) -> String {
        if self.data.is_empty() {
            return self.render_empty();
        }

        // Calculate bounds
        let (min, max) = self.calculate_bounds();
        if (max - min).abs() < f32::EPSILON {
            return self.render_flat_line();
        }

        // Each braille character is 2 dots wide, 4 dots tall
        let dots_wide = self.width * 2;
        let dots_tall = self.height * 4;

        // Create dot grid
        let mut grid = vec![vec![false; dots_wide]; dots_tall];

        // Plot data points
        let step = if self.data.len() > dots_wide {
            self.data.len() as f32 / dots_wide as f32
        } else {
            1.0
        };

        for x in 0..dots_wide.min(self.data.len()) {
            let idx = (x as f32 * step) as usize;
            let idx = idx.min(self.data.len() - 1);
            let val = self.data[idx];

            // Normalize value to dot position
            let normalized = if self.log_scale && val > 0.0 {
                let log_min = if min > 0.0 { min.ln() } else { 0.0 };
                let log_max = if max > 0.0 { max.ln() } else { 1.0 };
                let log_val = val.ln();
                (log_val - log_min) / (log_max - log_min)
            } else {
                (val - min) / (max - min)
            };

            let y = ((1.0 - normalized) * (dots_tall - 1) as f32) as usize;
            let y = y.min(dots_tall - 1);

            grid[y][x] = true;

            // Fill below for area effect (optional)
            // for y_fill in y..dots_tall {
            //     grid[y_fill][x] = true;
            // }
        }

        // Convert grid to braille characters
        let mut output = String::new();
        for row in 0..self.height {
            for col in 0..self.width {
                let mut code: u32 = 0;
                for dy in 0..4 {
                    for dx in 0..2 {
                        let gy = row * 4 + dy;
                        let gx = col * 2 + dx;
                        if gx < dots_wide && gy < dots_tall && grid[gy][gx] {
                            code |= BRAILLE_DOTS[dy * 2 + dx];
                        }
                    }
                }
                let ch = char::from_u32(BRAILLE_BASE + code).unwrap_or(' ');
                output.push(ch);
            }
            output.push('\n');
        }

        output
    }

    fn calculate_bounds(&self) -> (f32, f32) {
        let min = self
            .min_val
            .unwrap_or_else(|| self.data.iter().copied().fold(f32::INFINITY, f32::min));
        let max = self
            .max_val
            .unwrap_or_else(|| self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        (min, max)
    }

    fn render_empty(&self) -> String {
        let empty_char = char::from_u32(BRAILLE_BASE).unwrap_or(' ');
        let line = empty_char.to_string().repeat(self.width);
        let mut output = String::new();
        for _ in 0..self.height {
            output.push_str(&line);
            output.push('\n');
        }
        output
    }

    fn render_flat_line(&self) -> String {
        // Middle row of dots
        let middle_row = self.height / 2;
        let mut output = String::new();
        for row in 0..self.height {
            for _ in 0..self.width {
                let ch = if row == middle_row {
                    char::from_u32(BRAILLE_BASE | 0x36).unwrap_or('─') // dots 2,3,5,6
                } else {
                    char::from_u32(BRAILLE_BASE).unwrap_or(' ')
                };
                output.push(ch);
            }
            output.push('\n');
        }
        output
    }
}

/// Render a Braille chart from data
pub fn render_braille_chart(data: &[f32], width: usize, height: usize, log_scale: bool) -> String {
    BrailleChart::new(width, height)
        .data(data.to_vec())
        .log_scale(log_scale)
        .render()
}

/// Render a horizontal gauge/progress bar
///
/// Format: `[=========>          ] 45%`
pub fn render_gauge(value: f32, max: f32, width: usize, label: &str) -> String {
    let percent = if max > 0.0 { value / max * 100.0 } else { 0.0 };
    let percent = percent.clamp(0.0, 100.0);

    let bar_width = width.saturating_sub(label.len() + 8); // space for label + percentage
    let filled = ((percent / 100.0) * bar_width as f32) as usize;
    let empty = bar_width.saturating_sub(filled);

    let bar = format!(
        "{}[{}{}] {:>5.1}%",
        label,
        "=".repeat(filled.saturating_sub(1)) + if filled > 0 { ">" } else { "" },
        " ".repeat(empty),
        percent
    );

    bar
}

/// Format duration as HH:MM:SS
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{hours:02}:{mins:02}:{secs:02}")
}

/// Render the full TUI layout (SPEC-FT-001 Section 10.1)
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────────────┐
/// │  Entrenar Fine-Tuner v1.6.0                      [Running: 00:04:12]   │
/// ├──────────────────────────────────┬─────────────────────────────────────┤
/// │  Loss Curve (Log Scale)          │  Hardware Telemetry                 │
/// │                                  │                                     │
/// │  │        ⣀                      │  GPU: RTX 4090 [==========] 42%     │
/// │  │          ⣀                    │  VRAM: 3.4GB   [==        ] 14%     │
/// │  │           ⣀                   │  Temp: 64°C                         │
/// │  │            ⣀⣀                 │                                     │
/// │  └──────────────────────         │  Throughput: 1420 tok/s             │
/// │                                  │  Est. Remaining: 00:12:45           │
/// ├──────────────────────────────────┼─────────────────────────────────────┤
/// │  Latest Sample                   │  Training State                     │
/// │                                  │                                     │
/// │  Input:  fn is_even(n: u32)...   │  Epoch: 2/15                        │
/// │  Target: assert!(is_even(2))...  │  Step:  450/3000                    │
/// │  Gen:    assert!(is_even(2))...  │  LR:    5.8e-4                      │
/// │                                  │  Grad Norm: 3.2                     │
/// └──────────────────────────────────┴─────────────────────────────────────┘
/// ```
pub fn render_layout(snapshot: &TrainingSnapshot, width: usize) -> String {
    let mut output = String::new();
    let half_width = width / 2;

    // Status text
    let status_str = match &snapshot.status {
        TrainingStatus::Initializing => "Initializing",
        TrainingStatus::Running => "Running",
        TrainingStatus::Paused => "Paused",
        TrainingStatus::Completed => "Completed",
        TrainingStatus::Failed(_) => "Failed",
    };

    let elapsed = format_duration(snapshot.elapsed());

    // Header
    output.push_str(&format!("┌{}┐\n", "─".repeat(width - 2)));
    output.push_str(&format!(
        "│  Entrenar Fine-Tuner v0.5.6 {:>width$}│\n",
        format!("[{}: {}]  ", status_str, elapsed),
        width = width - 32
    ));
    output.push_str(&format!(
        "├{}┬{}┤\n",
        "─".repeat(half_width - 1),
        "─".repeat(width - half_width - 2)
    ));

    // Loss Curve Panel
    output.push_str(&format!(
        "│  Loss Curve (Log Scale){}│  Hardware Telemetry{}│\n",
        " ".repeat(half_width - 26),
        " ".repeat(width - half_width - 23)
    ));

    // Render braille chart (6 rows)
    let chart_width = half_width - 6;
    let chart = BrailleChart::new(chart_width, 6)
        .data(snapshot.loss_history.clone())
        .log_scale(true)
        .render();

    let chart_lines: Vec<&str> = chart.lines().collect();

    // GPU telemetry lines
    let gpu_lines = render_gpu_telemetry(snapshot.gpu.as_ref(), width - half_width - 4);

    // Combine chart and GPU telemetry
    for i in 0..6 {
        let chart_line = chart_lines.get(i).unwrap_or(&"");
        let gpu_line = gpu_lines.get(i).map_or("", std::string::String::as_str);
        let gpu_width = width - half_width - 5;

        output.push_str(&format!(
            "│  {}{}│  {:gpu_width$}│\n",
            chart_line,
            " ".repeat(half_width - chart_line.chars().count() - 4),
            gpu_line
        ));
    }

    // Middle divider
    output.push_str(&format!(
        "├{}┼{}┤\n",
        "─".repeat(half_width - 1),
        "─".repeat(width - half_width - 2)
    ));

    // Sample Peek Panel
    output.push_str(&format!(
        "│  Latest Sample{}│  Training State{}│\n",
        " ".repeat(half_width - 18),
        " ".repeat(width - half_width - 19)
    ));

    let sample_lines = render_sample_peek(snapshot.sample.as_ref(), half_width - 4);
    let state_lines = render_training_state(snapshot, width - half_width - 4);

    for i in 0..4 {
        let sample_line = sample_lines.get(i).map_or("", std::string::String::as_str);
        let state_line = state_lines.get(i).map_or("", std::string::String::as_str);
        let sample_width = half_width - 4;
        let state_width = width - half_width - 5;

        #[allow(clippy::uninlined_format_args)]
        output.push_str(&format!(
            "│  {:sample_width$}│  {:state_width$}│\n",
            sample_line, state_line
        ));
    }

    // Footer
    output.push_str(&format!(
        "└{}┴{}┘\n",
        "─".repeat(half_width - 1),
        "─".repeat(width - half_width - 2)
    ));

    output
}

fn render_gpu_telemetry(gpu: Option<&GpuTelemetry>, width: usize) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(gpu) = gpu {
        lines.push(format!(
            "GPU: {} {}",
            gpu.device_name,
            render_mini_gauge(gpu.utilization_percent, 100.0, 12)
        ));
        lines.push(format!(
            "VRAM: {:.1}GB/{}GB {}",
            gpu.vram_used_gb,
            gpu.vram_total_gb,
            render_mini_gauge(gpu.vram_percent(), 100.0, 10)
        ));
        lines.push(format!("Temp: {:.0}°C", gpu.temperature_celsius));
        lines.push(String::new());
    } else {
        lines.push("GPU: N/A".to_string());
        lines.push("VRAM: N/A".to_string());
        lines.push("Temp: N/A".to_string());
        lines.push(String::new());
    }

    // Pad or truncate to width
    for line in &mut lines {
        if line.len() > width {
            line.truncate(width);
        }
    }

    lines
}

fn render_mini_gauge(value: f32, max: f32, width: usize) -> String {
    let percent = if max > 0.0 { value / max } else { 0.0 };
    let percent = percent.clamp(0.0, 1.0);
    let filled = (percent * width as f32) as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "[{}{}] {:>3.0}%",
        "=".repeat(filled),
        " ".repeat(empty),
        percent * 100.0
    )
}

fn render_sample_peek(sample: Option<&SamplePeek>, width: usize) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(sample) = sample {
        let truncate = |s: &str, max_len: usize| -> String {
            if s.len() > max_len {
                format!("{}...", &s[..max_len.saturating_sub(3)])
            } else {
                s.to_string()
            }
        };

        let max_preview = width.saturating_sub(10);
        lines.push(format!(
            "Input:  {}",
            truncate(&sample.input_preview, max_preview)
        ));
        lines.push(format!(
            "Target: {}",
            truncate(&sample.target_preview, max_preview)
        ));
        lines.push(format!(
            "Gen:    {}",
            truncate(&sample.generated_preview, max_preview)
        ));
        lines.push(format!("Match:  {:.1}%", sample.token_match_percent));
    } else {
        lines.push("Input:  (waiting...)".to_string());
        lines.push("Target: (waiting...)".to_string());
        lines.push("Gen:    (waiting...)".to_string());
        lines.push(String::new());
    }

    lines
}

fn render_training_state(snapshot: &TrainingSnapshot, _width: usize) -> Vec<String> {
    let mut lines = Vec::new();

    lines.push(format!(
        "Epoch: {}/{}",
        snapshot.epoch, snapshot.total_epochs
    ));
    lines.push(format!(
        "Step:  {}/{}",
        snapshot.step, snapshot.steps_per_epoch
    ));
    lines.push(format!("LR:    {:.2e}", snapshot.learning_rate));
    lines.push(format!("Grad:  {:.3}", snapshot.gradient_norm));

    // Add throughput and ETA
    if snapshot.tokens_per_second > 0.0 {
        lines.push(format!("Tok/s: {:.0}", snapshot.tokens_per_second));
    }

    if let Some(remaining) = snapshot.estimated_remaining() {
        lines.push(format!("ETA:   {}", format_duration(remaining)));
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braille_chart_empty() {
        let chart = BrailleChart::new(10, 5).render();
        assert!(!chart.is_empty());
        assert!(chart.lines().count() == 5);
    }

    #[test]
    fn test_braille_chart_with_data() {
        let data = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        let chart = BrailleChart::new(10, 4).data(data).render();
        assert!(!chart.is_empty());
        // Should contain braille characters
        assert!(chart.chars().any(|c| c as u32 >= BRAILLE_BASE));
    }

    #[test]
    fn test_render_gauge() {
        let gauge = render_gauge(45.0, 100.0, 30, "GPU: ");
        assert!(gauge.contains("GPU:"));
        assert!(gauge.contains("45.0%"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
        assert_eq!(format_duration(Duration::from_secs(7200)), "02:00:00");
    }

    #[test]
    fn test_render_layout() {
        let snapshot = TrainingSnapshot {
            epoch: 2,
            total_epochs: 10,
            step: 50,
            steps_per_epoch: 100,
            loss: 0.42,
            loss_history: vec![1.0, 0.8, 0.6, 0.5, 0.42],
            learning_rate: 0.0002,
            gradient_norm: 1.5,
            tokens_per_second: 1200.0,
            status: TrainingStatus::Running,
            ..Default::default()
        };

        let layout = render_layout(&snapshot, 80);
        assert!(layout.contains("Entrenar"));
        assert!(layout.contains("Loss Curve"));
        assert!(layout.contains("Training State"));
    }
}

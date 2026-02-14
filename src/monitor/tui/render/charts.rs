//! Chart widgets: gauge, braille chart, sample panel, config panel, history table.

use super::super::color::{ColorMode, Styled};
use super::super::state::{SamplePeek, TrainingSnapshot};
use super::bars::{build_block_bar, render_sparkline};
use super::epoch::{compute_epoch_summaries, EpochSummary};
use super::format::format_lr;

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

pub fn render_sample_panel(
    _sample: Option<&SamplePeek>,
    _width: usize,
    _color_mode: ColorMode,
) -> String {
    String::new()
}

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
    lines.push("\u{2500}".repeat(width.min(70)));

    let summaries = compute_epoch_summaries(snapshot);
    if summaries.is_empty() {
        lines.push("(waiting for epoch data...)".to_string());
        return lines.join("\n");
    }

    let start_idx = summaries.len().saturating_sub(max_rows);
    for (i, summary) in summaries.iter().skip(start_idx).enumerate() {
        let trend = history_trend(i, start_idx, summary, &summaries, color_mode);

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
        lines.push(format!("  \u{2191} {start_idx} more epochs above"));
    }

    lines.join("\n")
}

fn history_trend<'a>(
    i: usize,
    start_idx: usize,
    summary: &EpochSummary,
    summaries: &[EpochSummary],
    _color_mode: ColorMode,
) -> (&'a str, (u8, u8, u8)) {
    if i > 0 || start_idx > 0 {
        let prev_idx = if i > 0 {
            start_idx + i - 1
        } else {
            start_idx.saturating_sub(1)
        };
        if let Some(prev) = summaries.get(prev_idx) {
            let change = (summary.avg_loss - prev.avg_loss) / prev.avg_loss.abs().max(0.001);
            if change < -0.02 {
                ("\u{2193}", (100, 255, 100))
            } else if change > 0.02 {
                ("\u{2191}", (255, 100, 100))
            } else {
                ("\u{2192}", (150, 150, 150))
            }
        } else {
            ("", (150, 150, 150))
        }
    } else {
        ("", (150, 150, 150))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

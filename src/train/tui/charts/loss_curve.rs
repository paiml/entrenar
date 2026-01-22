//! Loss curve display wrapper for trueno-viz.

use trueno_viz::output::{TerminalEncoder, TerminalMode as TruenoTerminalMode};
use trueno_viz::plots::{LossCurve, MetricSeries};
use trueno_viz::prelude::Rgba;

use crate::train::tui::capability::TerminalMode;

/// Summary of a metric series: (name, min_value, last_smoothed, best_epoch).
pub type SeriesSummaryTuple = (String, Option<f32>, Option<f32>, Option<usize>);

/// Wrapper for trueno-viz LossCurve with terminal output support.
///
/// Provides streaming loss curve visualization with:
/// - Train and validation loss tracking
/// - Exponential moving average smoothing
/// - Best value markers
/// - ASCII/Unicode/ANSI terminal rendering modes
///
/// # Example
///
/// ```no_run
/// use entrenar::train::tui::LossCurveDisplay;
///
/// let mut display = LossCurveDisplay::new(80, 20);
/// display.push_train_loss(1.0);
/// display.push_val_loss(1.2);
/// println!("{}", display.render_terminal());
/// ```
pub struct LossCurveDisplay {
    loss_curve: LossCurve,
    width: u32,
    height: u32,
    pub(crate) terminal_mode: TerminalMode,
}

impl LossCurveDisplay {
    /// Create a new loss curve display.
    pub fn new(width: u32, height: u32) -> Self {
        let loss_curve = LossCurve::new()
            .add_series(MetricSeries::new("Train", Rgba::rgb(66, 133, 244)))
            .add_series(MetricSeries::new("Val", Rgba::rgb(255, 128, 0)))
            .dimensions(width, height)
            .margin(2)
            .best_markers(true)
            .lower_is_better(true)
            .build()
            .expect("LossCurve build should succeed");
        Self {
            loss_curve,
            width,
            height,
            terminal_mode: TerminalMode::Unicode,
        }
    }

    /// Set terminal rendering mode.
    pub fn terminal_mode(mut self, mode: TerminalMode) -> Self {
        self.terminal_mode = mode;
        self
    }

    /// Set smoothing factor (0.0 = none, 0.99 = heavy).
    pub fn smoothing(mut self, factor: f32) -> Self {
        // Re-create with smoothing applied
        self.loss_curve = LossCurve::new()
            .add_series(MetricSeries::new("Train", Rgba::rgb(66, 133, 244)).smoothing(factor))
            .add_series(MetricSeries::new("Val", Rgba::rgb(255, 128, 0)).smoothing(factor))
            .dimensions(self.width, self.height)
            .margin(2)
            .best_markers(true)
            .lower_is_better(true)
            .build()
            .expect("LossCurve build should succeed");
        self
    }

    /// Push a training loss value.
    pub fn push_train_loss(&mut self, value: f32) {
        self.loss_curve.push(0, value);
    }

    /// Push a validation loss value.
    pub fn push_val_loss(&mut self, value: f32) {
        self.loss_curve.push(1, value);
    }

    /// Push both train and val loss at once.
    pub fn push_losses(&mut self, train: f32, val: f32) {
        self.loss_curve.push_all(&[train, val]);
    }

    /// Get the number of epochs recorded.
    pub fn epochs(&self) -> usize {
        self.loss_curve.max_epochs()
    }

    /// Get summary of all series.
    pub fn summary(&self) -> Vec<SeriesSummaryTuple> {
        self.loss_curve
            .summary()
            .into_iter()
            .map(|s| (s.name, s.min, s.last_smoothed, s.best_epoch))
            .collect()
    }

    /// Render to terminal string.
    pub fn render_terminal(&self) -> String {
        if self.loss_curve.max_epochs() < 2 {
            return String::from("(waiting for data...)");
        }

        let fb = match self.loss_curve.to_framebuffer() {
            Ok(fb) => fb,
            Err(_) => return String::from("(render error)"),
        };

        let trueno_mode = match self.terminal_mode {
            TerminalMode::Ascii => TruenoTerminalMode::Ascii,
            TerminalMode::Unicode => TruenoTerminalMode::UnicodeHalfBlock,
            TerminalMode::Ansi => TruenoTerminalMode::AnsiTrueColor,
        };

        let encoder = TerminalEncoder::new()
            .mode(trueno_mode)
            .width(self.width)
            .height(self.height / 2); // Terminal chars are ~2:1 aspect

        encoder.render(&fb)
    }

    /// Print to stdout.
    pub fn print(&self) {
        println!("{}", self.render_terminal());
    }
}

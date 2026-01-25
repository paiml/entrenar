//! Terminal Color Support (ENT-122)
//!
//! Provides ANSI color output with automatic terminal capability detection.
//! Based on presentar's color system with semantic colors for training metrics.

use std::fmt;

/// Terminal color capability mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// True color (24-bit RGB)
    TrueColor,
    /// 256 color palette
    Color256,
    /// 16 color palette
    Color16,
    /// No color (monochrome)
    #[default]
    Mono,
}

impl ColorMode {
    /// Detect terminal color capability from environment
    pub fn detect() -> Self {
        Self::detect_with_env(
            std::env::var("COLORTERM").ok().as_deref(),
            std::env::var("TERM").ok().as_deref(),
            std::env::var("NO_COLOR").ok().as_deref(),
        )
    }

    /// Detect with explicit environment values (for testing)
    pub fn detect_with_env(
        colorterm: Option<&str>,
        term: Option<&str>,
        no_color: Option<&str>,
    ) -> Self {
        // NO_COLOR takes precedence
        if no_color.is_some() {
            return Self::Mono;
        }

        // Check COLORTERM for truecolor support
        if let Some(ct) = colorterm {
            if ct.contains("truecolor") || ct.contains("24bit") {
                return Self::TrueColor;
            }
        }

        // Check TERM for capability hints
        if let Some(term) = term {
            if term.contains("256color") || term.contains("kitty") || term.contains("alacritty") {
                return Self::Color256;
            }
            if term.contains("xterm") || term.contains("screen") || term.contains("tmux") {
                return Self::Color16;
            }
            if term == "dumb" || term.is_empty() {
                return Self::Mono;
            }
        }

        // Default to 16 colors for unknown terminals
        Self::Color16
    }
}

/// RGB color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to ANSI 256-color index (approximate)
    pub fn to_256(self) -> u8 {
        // Use the 216 color cube (indices 16-231)
        // Each channel has 6 levels: 0, 95, 135, 175, 215, 255
        let r6 = (u16::from(self.r) * 5 / 255) as u8;
        let g6 = (u16::from(self.g) * 5 / 255) as u8;
        let b6 = (u16::from(self.b) * 5 / 255) as u8;
        16 + 36 * r6 + 6 * g6 + b6
    }

    /// Convert to ANSI 16-color index (approximate)
    pub fn to_16(self) -> u8 {
        // Use max channel for brightness detection (saturated colors should be bright)
        let max_channel = self.r.max(self.g).max(self.b);
        let is_bright = max_channel > 180;

        // Determine dominant color
        let r_dom = self.r >= self.g && self.r >= self.b;
        let g_dom = self.g >= self.r && self.g >= self.b;
        let b_dom = self.b >= self.r && self.b >= self.g;

        // Mix detection
        let r_present = self.r > 85;
        let g_present = self.g > 85;
        let b_present = self.b > 85;

        let base = match (r_present, g_present, b_present) {
            (true, true, true) => 7,   // white
            (true, true, false) => 3,  // yellow
            (true, false, true) => 5,  // magenta
            (false, true, true) => 6,  // cyan
            (true, false, false) => 1, // red
            (false, true, false) => 2, // green
            (false, false, true) => 4, // blue
            (false, false, false) => {
                // Near black - check if any color is dominant
                if r_dom && self.r > 40 {
                    1
                } else if g_dom && self.g > 40 {
                    2
                } else if b_dom && self.b > 40 {
                    4
                } else {
                    0
                }
            }
        };

        if is_bright {
            base + 8
        } else {
            base
        }
    }
}

/// Styled text with foreground color
pub struct Styled<'a> {
    text: &'a str,
    fg: Option<Rgb>,
    bold: bool,
    mode: ColorMode,
}

impl<'a> Styled<'a> {
    pub fn new(text: &'a str, mode: ColorMode) -> Self {
        Self {
            text,
            fg: None,
            bold: false,
            mode,
        }
    }

    pub fn fg(mut self, color: Rgb) -> Self {
        self.fg = Some(color);
        self
    }

    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }
}

impl fmt::Display for Styled<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode == ColorMode::Mono {
            return write!(f, "{}", self.text);
        }

        let mut has_style = false;

        // Bold
        if self.bold {
            write!(f, "\x1b[1m")?;
            has_style = true;
        }

        // Foreground color
        if let Some(rgb) = self.fg {
            match self.mode {
                ColorMode::TrueColor => {
                    write!(f, "\x1b[38;2;{};{};{}m", rgb.r, rgb.g, rgb.b)?;
                }
                ColorMode::Color256 => {
                    write!(f, "\x1b[38;5;{}m", rgb.to_256())?;
                }
                ColorMode::Color16 => {
                    let code = rgb.to_16();
                    if code >= 8 {
                        write!(f, "\x1b[9{}m", code - 8)?;
                    } else {
                        write!(f, "\x1b[3{code}m")?;
                    }
                }
                ColorMode::Mono => {}
            }
            has_style = true;
        }

        write!(f, "{}", self.text)?;

        if has_style {
            write!(f, "\x1b[0m")?;
        }

        Ok(())
    }
}

/// Semantic color palette for training metrics
#[derive(Debug, Clone)]
pub struct TrainingPalette {
    pub mode: ColorMode,
}

impl Default for TrainingPalette {
    fn default() -> Self {
        Self {
            mode: ColorMode::detect(),
        }
    }
}

impl TrainingPalette {
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }

    /// Style text with this palette's color mode
    pub fn style<'a>(&self, text: &'a str) -> Styled<'a> {
        Styled::new(text, self.mode)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Semantic Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Success/good state (green)
    pub const SUCCESS: Rgb = Rgb::new(80, 200, 120);

    /// Warning state (yellow/orange)
    pub const WARNING: Rgb = Rgb::new(255, 193, 7);

    /// Error/danger state (red)
    pub const ERROR: Rgb = Rgb::new(244, 67, 54);

    /// Info/neutral (blue)
    pub const INFO: Rgb = Rgb::new(33, 150, 243);

    /// Muted/secondary text (gray)
    pub const MUTED: Rgb = Rgb::new(158, 158, 158);

    /// Primary accent (cyan)
    pub const PRIMARY: Rgb = Rgb::new(0, 188, 212);

    // ─────────────────────────────────────────────────────────────────────────
    // GPU Metrics Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for GPU utilization based on percentage
    pub fn gpu_util_color(percent: f32) -> Rgb {
        match percent as u32 {
            0..=30 => Self::MUTED,    // Low (gray - underutilized)
            31..=70 => Self::SUCCESS, // Good (green)
            71..=90 => Self::INFO,    // High (blue)
            _ => Self::PRIMARY,       // Very high (cyan)
        }
    }

    /// Color for VRAM usage based on percentage
    pub fn vram_color(percent: f32) -> Rgb {
        match percent as u32 {
            0..=50 => Self::SUCCESS,  // OK (green)
            51..=75 => Self::INFO,    // Moderate (blue)
            76..=90 => Self::WARNING, // High (yellow)
            _ => Self::ERROR,         // Critical (red)
        }
    }

    /// Color for temperature in Celsius
    pub fn temp_color(celsius: f32) -> Rgb {
        match celsius as u32 {
            0..=50 => Self::SUCCESS,  // Cool (green)
            51..=70 => Self::INFO,    // Normal (blue)
            71..=80 => Self::WARNING, // Warm (yellow)
            _ => Self::ERROR,         // Hot (red)
        }
    }

    /// Color for power usage based on percentage of limit
    pub fn power_color(percent: f32) -> Rgb {
        match percent as u32 {
            0..=60 => Self::SUCCESS,  // Low (green)
            61..=80 => Self::INFO,    // Moderate (blue)
            81..=95 => Self::WARNING, // High (yellow)
            _ => Self::ERROR,         // At limit (red)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Training Metrics Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for gradient norm (explosion warning)
    pub fn grad_norm_color(norm: f32) -> Rgb {
        match norm {
            n if n <= 1.0 => Self::SUCCESS,  // Healthy
            n if n <= 5.0 => Self::INFO,     // Normal
            n if n <= 10.0 => Self::WARNING, // High
            _ => Self::ERROR,                // Explosion risk
        }
    }

    /// Color for loss value (lower is better)
    /// Returns a gradient from red (high loss) to green (low loss)
    pub fn loss_color(loss: f32, min_loss: f32, max_loss: f32) -> Rgb {
        if max_loss <= min_loss {
            return Self::INFO;
        }

        let normalized = ((loss - min_loss) / (max_loss - min_loss)).clamp(0.0, 1.0);

        // Gradient from green (0.0) to yellow (0.5) to red (1.0)
        let (r, g, b) = if normalized < 0.5 {
            // Green to yellow
            let t = normalized * 2.0;
            (
                (80.0 + t * 175.0) as u8,  // 80 -> 255
                (200.0 - t * 7.0) as u8,   // 200 -> 193
                (120.0 - t * 113.0) as u8, // 120 -> 7
            )
        } else {
            // Yellow to red
            let t = (normalized - 0.5) * 2.0;
            (
                (255.0 - t * 11.0) as u8,  // 255 -> 244
                (193.0 - t * 126.0) as u8, // 193 -> 67
                (7.0 + t * 47.0) as u8,    // 7 -> 54
            )
        };

        Rgb::new(r, g, b)
    }

    /// Color for training status
    pub fn status_color(status: &str) -> Rgb {
        match status.to_lowercase().as_str() {
            "running" => Self::SUCCESS,
            "completed" => Self::PRIMARY,
            "paused" => Self::WARNING,
            "failed" => Self::ERROR,
            "initializing" => Self::INFO,
            _ => Self::MUTED,
        }
    }

    /// Color for loss trend indicator
    pub fn loss_trend_color(trend: &str) -> Rgb {
        match trend {
            "decreasing" => Self::SUCCESS, // Good - loss is going down
            "stable" => Self::INFO,        // Neutral - plateauing
            "increasing" => Self::ERROR,   // Bad - loss is going up
            _ => Self::MUTED,              // Unknown
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Progress Bar Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for progress bar fill based on completion percentage
    pub fn progress_color(percent: f32) -> Rgb {
        match percent as u32 {
            0..=25 => Self::INFO,     // Starting (blue)
            26..=50 => Self::INFO,    // Quarter way
            51..=75 => Self::INFO,    // Half way
            76..=99 => Self::SUCCESS, // Almost done (green)
            _ => Self::PRIMARY,       // Complete (cyan)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Colored Progress Bar
// ─────────────────────────────────────────────────────────────────────────────

/// Render a colored progress bar
pub fn colored_bar(value: f32, max: f32, width: usize, color: Rgb, mode: ColorMode) -> String {
    let percent = if max > 0.0 { value / max } else { 0.0 };
    let percent = percent.clamp(0.0, 1.0);
    let filled = (percent * width as f32) as usize;
    let empty = width.saturating_sub(filled);

    let filled_str: String = std::iter::repeat_n('█', filled).collect();
    let empty_str: String = std::iter::repeat_n('░', empty).collect();

    if mode == ColorMode::Mono {
        format!("{filled_str}{empty_str}")
    } else {
        format!(
            "{}{}",
            Styled::new(&filled_str, mode).fg(color),
            Styled::new(&empty_str, mode).fg(TrainingPalette::MUTED)
        )
    }
}

/// Render a colored value with semantic coloring
pub fn colored_value<T: fmt::Display>(value: T, color: Rgb, mode: ColorMode) -> String {
    let text = value.to_string();
    Styled::new(&text, mode).fg(color).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_mode_detection() {
        // NO_COLOR takes precedence
        assert_eq!(
            ColorMode::detect_with_env(Some("truecolor"), Some("xterm-256color"), Some("1")),
            ColorMode::Mono
        );

        // COLORTERM truecolor
        assert_eq!(
            ColorMode::detect_with_env(Some("truecolor"), None, None),
            ColorMode::TrueColor
        );

        // TERM 256color
        assert_eq!(
            ColorMode::detect_with_env(None, Some("xterm-256color"), None),
            ColorMode::Color256
        );

        // TERM xterm
        assert_eq!(
            ColorMode::detect_with_env(None, Some("xterm"), None),
            ColorMode::Color16
        );

        // TERM dumb
        assert_eq!(
            ColorMode::detect_with_env(None, Some("dumb"), None),
            ColorMode::Mono
        );
    }

    #[test]
    fn test_rgb_to_256() {
        // Black
        assert_eq!(Rgb::new(0, 0, 0).to_256(), 16);
        // White
        assert_eq!(Rgb::new(255, 255, 255).to_256(), 231);
        // Red
        assert_eq!(Rgb::new(255, 0, 0).to_256(), 196);
        // Green
        assert_eq!(Rgb::new(0, 255, 0).to_256(), 46);
        // Blue
        assert_eq!(Rgb::new(0, 0, 255).to_256(), 21);
    }

    #[test]
    fn test_rgb_to_16() {
        // Bright red
        assert_eq!(Rgb::new(255, 50, 50).to_16(), 9); // bright red
                                                      // Bright green
        assert_eq!(Rgb::new(50, 255, 50).to_16(), 10); // bright green
                                                       // Dark blue
        assert_eq!(Rgb::new(0, 0, 100).to_16(), 4); // blue
    }

    #[test]
    fn test_styled_display_truecolor() {
        let styled = Styled::new("test", ColorMode::TrueColor).fg(Rgb::new(255, 0, 0));
        let output = styled.to_string();
        assert!(output.contains("\x1b[38;2;255;0;0m"));
        assert!(output.contains("test"));
        assert!(output.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_styled_display_mono() {
        let styled = Styled::new("test", ColorMode::Mono).fg(Rgb::new(255, 0, 0));
        let output = styled.to_string();
        assert_eq!(output, "test");
    }

    #[test]
    fn test_gpu_util_color() {
        assert_eq!(
            TrainingPalette::gpu_util_color(20.0),
            TrainingPalette::MUTED
        );
        assert_eq!(
            TrainingPalette::gpu_util_color(50.0),
            TrainingPalette::SUCCESS
        );
        assert_eq!(TrainingPalette::gpu_util_color(80.0), TrainingPalette::INFO);
        assert_eq!(
            TrainingPalette::gpu_util_color(95.0),
            TrainingPalette::PRIMARY
        );
    }

    #[test]
    fn test_temp_color() {
        assert_eq!(TrainingPalette::temp_color(40.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::temp_color(65.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::temp_color(75.0), TrainingPalette::WARNING);
        assert_eq!(TrainingPalette::temp_color(85.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_grad_norm_color() {
        assert_eq!(
            TrainingPalette::grad_norm_color(0.5),
            TrainingPalette::SUCCESS
        );
        assert_eq!(TrainingPalette::grad_norm_color(3.0), TrainingPalette::INFO);
        assert_eq!(
            TrainingPalette::grad_norm_color(8.0),
            TrainingPalette::WARNING
        );
        assert_eq!(
            TrainingPalette::grad_norm_color(20.0),
            TrainingPalette::ERROR
        );
    }

    #[test]
    fn test_loss_color_gradient() {
        let min = 0.0;
        let max = 1.0;

        // Low loss should be greenish
        let low = TrainingPalette::loss_color(0.1, min, max);
        assert!(low.g > low.r); // More green than red

        // High loss should be reddish
        let high = TrainingPalette::loss_color(0.9, min, max);
        assert!(high.r > high.g); // More red than green
    }

    #[test]
    fn test_colored_bar() {
        let bar = colored_bar(50.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        assert!(bar.contains('█'));
        assert!(bar.contains('░'));
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 5);
    }
}

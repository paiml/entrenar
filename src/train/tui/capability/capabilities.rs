//! Terminal capability detection.

use super::TerminalMode;

/// Detected terminal capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    /// Terminal width in columns
    pub width: u16,
    /// Terminal height in rows
    pub height: u16,
    /// Supports Unicode characters
    pub unicode: bool,
    /// Supports ANSI color codes
    pub ansi_color: bool,
    /// Supports 24-bit true color
    pub true_color: bool,
    /// Is interactive TTY
    pub is_tty: bool,
}

impl Default for TerminalCapabilities {
    fn default() -> Self {
        Self {
            width: 80,
            height: 24,
            unicode: true,
            ansi_color: true,
            true_color: false,
            is_tty: true,
        }
    }
}

impl TerminalCapabilities {
    /// Detect terminal capabilities from environment.
    pub fn detect() -> Self {
        use std::env;
        use std::io::{stdout, IsTerminal};

        let is_tty = stdout().is_terminal();

        // Get size from environment or default
        let (width, height) = Self::get_size();

        // Check for Unicode support (most modern terminals)
        let lang = env::var("LANG").unwrap_or_default();
        let unicode = lang.contains("UTF") || lang.contains("utf");

        // Check for ANSI color support
        let term = env::var("TERM").unwrap_or_default();
        let ansi_color = !term.is_empty() && term != "dumb";

        // Check for true color support
        let colorterm = env::var("COLORTERM").unwrap_or_default();
        let true_color = colorterm == "truecolor" || colorterm == "24bit";

        Self {
            width,
            height,
            unicode,
            ansi_color,
            true_color,
            is_tty,
        }
    }

    /// Get terminal size.
    pub(crate) fn get_size() -> (u16, u16) {
        use std::env;

        // 1. Check environment variables (CI/headless)
        if let (Ok(cols), Ok(rows)) = (env::var("COLUMNS"), env::var("LINES")) {
            if let (Ok(c), Ok(r)) = (cols.parse(), rows.parse()) {
                return (c, r);
            }
        }

        // 2. Try ioctl on Unix
        #[cfg(unix)]
        {
            use std::io::{stdout, IsTerminal};
            if stdout().is_terminal() {
                // Use libc directly for TIOCGWINSZ
                #[repr(C)]
                struct WinSize {
                    ws_row: u16,
                    ws_col: u16,
                    ws_xpixel: u16,
                    ws_ypixel: u16,
                }
                extern "C" {
                    fn ioctl(fd: i32, request: u64, ...) -> i32;
                }
                const TIOCGWINSZ: u64 = 0x5413; // Linux
                let mut ws = WinSize {
                    ws_row: 0,
                    ws_col: 0,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };
                // SAFETY: ioctl with TIOCGWINSZ is safe for reading terminal size
                #[allow(unsafe_code)]
                if unsafe { ioctl(1, TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 {
                    return (ws.ws_col, ws.ws_row);
                }
            }
        }

        // 3. Fallback
        (80, 24)
    }

    /// Get recommended terminal mode based on capabilities.
    pub fn recommended_mode(&self) -> TerminalMode {
        if !self.is_tty {
            TerminalMode::Ascii
        } else if self.true_color {
            TerminalMode::Ansi
        } else if self.unicode {
            TerminalMode::Unicode
        } else {
            TerminalMode::Ascii
        }
    }
}

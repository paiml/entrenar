# ENTRENAR-TUI-SPEC-001: Probar-Compliant TUI Monitor

**Status:** Implemented
**Version:** 1.0.0
**Date:** 2026-01-25

## Overview

The entrenar TUI monitor implements a btop/ptop-style training visualization with probar compliance for testing and falsification.

## Probar Integration

### Features Used

| Feature | Module | Purpose |
|---------|--------|---------|
| TuiFrame | `tui` | Frame capture and assertions |
| TuiSnapshot | `tui` | Hash-based snapshot comparison |
| UxCoverageTracker | `ux_coverage` | Panel interaction coverage |
| InteractionType | `ux_coverage` | Element hover/click tracking |
| ElementId | `ux_coverage` | Panel identification |

### Falsification Protocol (F001-F010)

Per PROBAR-SPEC-015, we implement 10-point falsification for TUI robustness:

| ID | Test | Assertion |
|----|------|-----------|
| F001 | Epoch overflow | epoch clamped to total_epochs |
| F002 | Step overflow | step clamped to steps_per_epoch |
| F003 | NaN loss | Shows "???" with error color |
| F004 | Inf loss | Shows "???" with error color |
| F005 | Negative LR | Clamped to 0.0 |
| F006 | VRAM overflow | Clamped to 100% |
| F007 | Extreme temp | Shows warning indicator |
| F008 | Empty processes | Descriptive message |
| F009 | Missing GPU | Shows N/A placeholder |
| F010 | Missing sample | Graceful degradation |

## Quantitative Metrics

### Required Thresholds

| Metric | Minimum | Current |
|--------|---------|---------|
| Pixel coverage | 40% | 51.2% |
| Unicode richness | 10% | 17.0% |
| UX coverage | 70% | 80.0% |
| Panel score | 4/5 | 5/5 |
| Render time | <1ms | 0.046ms |
| Large history (10K) | <100ms | 1.01ms |

### Panel Coverage

All 5 panels must be present and functional:

1. **Epoch Progress** - Shows current/total epochs with bar
2. **Step Progress** - Shows current/total steps with bar
3. **Loss Display** - Current loss with sparkline and trend
4. **GPU Panel** - Utilization, VRAM, temperature, processes
5. **Sample Preview** - Input/target/generated previews

## Architecture

### Panel Verification (panel.rs)

```rust
pub trait Panel {
    fn name(&self) -> &'static str;
    fn can_render(&self) -> bool;  // Jidoka gate
    fn verify(&self) -> PanelVerification;
    fn budget_ms(&self) -> u32 { 16 }  // 60fps default
}
```

Implemented panels:
- `LossCurvePanel`
- `GpuPanel`
- `ProcessPanel`
- `SamplePanel`
- `MetricsPanel`

### Color System (color.rs)

btop-style gradient coloring:
- 0-25%: Cyan → Green
- 25-50%: Green → Yellow
- 50-75%: Yellow → Orange
- 75-90%: Orange → Red
- 90-100%: Critical Red

### Rendering (render.rs)

Unicode characters:
- Block bars: `█░`
- Braille sparklines: `⣿⣷⣶⣴⣤⣄⣀⡀`
- Trend arrows: `↑↓→`
- Status indicators: `●◐◔○⚡`
- Box drawing: `╭╮╰╯─│┬┴├┤┼`

## Data Sanitization

All numeric values are sanitized before display:

```rust
// Loss
if !loss.is_finite() { "???" } else { format!("{:.3}", loss) }

// Learning rate
lr.max(0.0)  // Clamp negative

// VRAM
vram_used.min(vram_total)  // Clamp overflow

// Progress
epoch.min(total_epochs)
step.min(steps_per_epoch)
```

## Testing

### Running Tests

```bash
# Full probar compliance test
cargo test --test probar_tui_compliance -- --nocapture

# Unit tests only
cargo test --lib -- tui::panel
```

### Test Output

```
════════════════════════════════════════════════════════════
              PROBAR TUI COMPLIANCE METRICS
════════════════════════════════════════════════════════════

PIXEL COVERAGE:
  Content density: 51.2%
  Unicode richness: 17.0%
  Total characters: 3150

FRAME DIMENSIONS:
  Width: 188 chars
  Height: 17 lines
  Total area: 3196 cells

PANEL COVERAGE:
  Epoch progress: ✓
  Step progress:  ✓
  Loss display:   ✓
  GPU panel:      ✓
  Sample preview: ✓

OVERALL PANEL SCORE: 5/5 (100%)
════════════════════════════════════════════════════════════
```

## Files

- `src/monitor/tui/mod.rs` - Module exports
- `src/monitor/tui/render.rs` - TUI rendering
- `src/monitor/tui/panel.rs` - Panel verification
- `src/monitor/tui/color.rs` - Color system
- `src/monitor/tui/state.rs` - State types
- `tests/probar_tui_compliance.rs` - Probar tests

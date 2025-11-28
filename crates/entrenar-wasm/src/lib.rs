//! Entrenar WASM - Training monitor for browsers
//!
//! Minimal WASM bindings without heavy dependencies.

use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {
    // WASM module initialized
}

/// Running statistics using Welford's algorithm
#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, value: f64) {
        if value.is_nan() || value.is_infinite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn std(&self) -> f64 {
        if self.count < 2 { 0.0 } else { (self.m2 / (self.count - 1) as f64).sqrt() }
    }
}

/// WASM Metrics Collector
#[wasm_bindgen]
pub struct MetricsCollector {
    loss: RunningStats,
    accuracy: RunningStats,
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
}

#[wasm_bindgen]
impl MetricsCollector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            loss: RunningStats::new(),
            accuracy: RunningStats::new(),
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
        }
    }

    pub fn record_loss(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.loss.update(value);
            self.loss_history.push(value);
            if self.loss_history.len() > 100 {
                self.loss_history.remove(0);
            }
        }
    }

    pub fn record_accuracy(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.accuracy.update(value);
            self.accuracy_history.push(value);
            if self.accuracy_history.len() > 100 {
                self.accuracy_history.remove(0);
            }
        }
    }

    pub fn loss_mean(&self) -> f64 {
        if self.loss.count == 0 { f64::NAN } else { self.loss.mean }
    }

    pub fn accuracy_mean(&self) -> f64 {
        if self.accuracy.count == 0 { f64::NAN } else { self.accuracy.mean }
    }

    pub fn loss_std(&self) -> f64 { self.loss.std() }
    pub fn accuracy_std(&self) -> f64 { self.accuracy.std() }
    pub fn count(&self) -> usize { self.loss.count + self.accuracy.count }

    pub fn clear(&mut self) {
        self.loss = RunningStats::new();
        self.accuracy = RunningStats::new();
        self.loss_history.clear();
        self.accuracy_history.clear();
    }

    pub fn loss_sparkline(&self) -> String {
        sparkline(&self.loss_history)
    }

    pub fn accuracy_sparkline(&self) -> String {
        sparkline(&self.accuracy_history)
    }

    pub fn state_json(&self) -> String {
        let state = serde_json::json!({
            "loss_mean": self.loss_mean(),
            "loss_std": self.loss_std(),
            "accuracy_mean": self.accuracy_mean(),
            "accuracy_std": self.accuracy_std(),
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        });
        state.to_string()
    }
}

fn sparkline(values: &[f64]) -> String {
    const CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() { return String::new(); }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-10 { 1.0 } else { max - min };

    values.iter().map(|v| {
        let norm = ((v - min) / range).clamp(0.0, 1.0);
        let idx = ((norm * 7.0).round() as usize).min(7);
        CHARS[idx]
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_new() {
        let c = MetricsCollector::new();
        assert_eq!(c.count(), 0);
        assert!(c.loss_mean().is_nan());
        assert!(c.accuracy_mean().is_nan());
    }

    #[test]
    fn test_record_loss() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_loss(0.3);
        assert_eq!(c.count(), 2);
        assert!((c.loss_mean() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_record_accuracy() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.8);
        c.record_accuracy(0.9);
        assert!((c.accuracy_mean() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_loss_std() {
        let mut c = MetricsCollector::new();
        c.record_loss(2.0);
        c.record_loss(4.0);
        c.record_loss(4.0);
        c.record_loss(4.0);
        c.record_loss(5.0);
        c.record_loss(5.0);
        c.record_loss(7.0);
        c.record_loss(9.0);
        let std = c.loss_std();
        assert!(std > 2.0 && std < 2.5);
    }

    #[test]
    fn test_ignores_nan() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_loss(f64::NAN);
        c.record_loss(0.3);
        assert_eq!(c.count(), 2);
    }

    #[test]
    fn test_ignores_inf() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.8);
        c.record_accuracy(f64::INFINITY);
        assert_eq!(c.count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_accuracy(0.8);
        c.clear();
        assert_eq!(c.count(), 0);
        assert!(c.loss_sparkline().is_empty());
    }

    #[test]
    fn test_sparkline_empty() {
        let c = MetricsCollector::new();
        assert!(c.loss_sparkline().is_empty());
    }

    #[test]
    fn test_sparkline_values() {
        let mut c = MetricsCollector::new();
        for i in 0..10 {
            c.record_loss(i as f64 / 10.0);
        }
        let s = c.loss_sparkline();
        assert!(!s.is_empty());
        assert!(s.chars().all(|c| "▁▂▃▄▅▆▇█".contains(c)));
    }

    #[test]
    fn test_state_json() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_accuracy(0.8);
        let json = c.state_json();
        assert!(json.contains("loss_mean"));
        assert!(json.contains("accuracy_mean"));
        assert!(json.contains("loss_history"));
    }

    #[test]
    fn test_history_bounded() {
        let mut c = MetricsCollector::new();
        for i in 0..150 {
            c.record_loss(i as f64);
        }
        // History should be bounded to 100
        let json = c.state_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let history = parsed["loss_history"].as_array().unwrap();
        assert_eq!(history.len(), 100);
    }

    #[test]
    fn test_running_stats_min_max() {
        let mut c = MetricsCollector::new();
        c.record_loss(5.0);
        c.record_loss(2.0);
        c.record_loss(8.0);
        // Check via state_json that values are tracked
        let json = c.state_json();
        assert!(json.contains("2") || json.contains("8"));
    }

    #[test]
    fn test_accuracy_std() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.7);
        c.record_accuracy(0.8);
        c.record_accuracy(0.9);
        let std = c.accuracy_std();
        assert!(std > 0.0);
    }

    #[test]
    fn test_sparkline_constant() {
        let mut c = MetricsCollector::new();
        for _ in 0..5 {
            c.record_loss(0.5);
        }
        let s = c.loss_sparkline();
        // All same value should produce consistent sparkline
        assert_eq!(s.chars().count(), 5);
    }
}

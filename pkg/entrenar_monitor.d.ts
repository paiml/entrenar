/**
 * Entrenar Monitor WASM TypeScript Definitions
 *
 * Real-time training visualization for the browser.
 *
 * @example
 * ```typescript
 * import init, { WasmMetricsCollector, WasmDashboardOptions } from 'entrenar-monitor';
 *
 * await init();
 *
 * const collector = new WasmMetricsCollector();
 * collector.record_loss(0.5);
 * collector.record_accuracy(0.85);
 *
 * const stats = JSON.parse(collector.summary_json());
 * console.log(`Loss: ${stats.loss.mean}, Accuracy: ${stats.accuracy.mean}`);
 * ```
 */

/**
 * Initialize the WASM module.
 * Must be called before using any other functions.
 */
export default function init(): Promise<void>;

/**
 * Statistics for a single metric.
 */
export interface MetricStats {
  /** Number of recorded values */
  count: number;
  /** Mean value */
  mean: number;
  /** Standard deviation */
  std: number;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Whether any NaN values were recorded */
  has_nan: boolean;
  /** Whether any Inf values were recorded */
  has_inf: boolean;
}

/**
 * Summary of all metrics.
 */
export interface MetricsSummary {
  loss?: MetricStats;
  accuracy?: MetricStats;
  learning_rate?: MetricStats;
  gradient_norm?: MetricStats;
  [key: string]: MetricStats | undefined;
}

/**
 * WASM-compatible metrics collector for training.
 *
 * @example
 * ```typescript
 * const collector = new WasmMetricsCollector();
 *
 * // Record metrics during training
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   collector.record_loss(1.0 / (epoch + 1));
 *   collector.record_accuracy(0.5 + 0.005 * epoch);
 * }
 *
 * // Get summary statistics
 * const summary: MetricsSummary = JSON.parse(collector.summary_json());
 * console.log(`Final loss: ${summary.loss?.mean}`);
 * ```
 */
export class WasmMetricsCollector {
  /**
   * Create a new metrics collector.
   */
  constructor();

  /**
   * Record a loss value.
   * @param value - The loss value to record
   */
  record_loss(value: number): void;

  /**
   * Record an accuracy value.
   * @param value - The accuracy value to record (0.0 to 1.0)
   */
  record_accuracy(value: number): void;

  /**
   * Record a learning rate value.
   * @param value - The learning rate value to record
   */
  record_learning_rate(value: number): void;

  /**
   * Record a gradient norm value.
   * @param value - The gradient L2 norm to record
   */
  record_gradient_norm(value: number): void;

  /**
   * Record a custom metric.
   * @param name - The metric name
   * @param value - The metric value
   */
  record_custom(name: string, value: number): void;

  /**
   * Get the total number of recorded metrics.
   */
  count(): number;

  /**
   * Check if the collector is empty.
   */
  is_empty(): boolean;

  /**
   * Clear all recorded metrics.
   */
  clear(): void;

  /**
   * Get summary statistics as a JSON string.
   * Parse with JSON.parse() to get MetricsSummary.
   */
  summary_json(): string;

  /**
   * Get the mean loss value.
   * Returns NaN if no loss has been recorded.
   */
  loss_mean(): number;

  /**
   * Get the mean accuracy value.
   * Returns NaN if no accuracy has been recorded.
   */
  accuracy_mean(): number;
}

/**
 * Dashboard rendering options.
 *
 * @example
 * ```typescript
 * const opts = new WasmDashboardOptions()
 *   .width(1024)
 *   .height(768)
 *   .background_color('#1a1a2e')
 *   .loss_color('#ff6b6b')
 *   .accuracy_color('#4ecdc4');
 * ```
 */
export class WasmDashboardOptions {
  /**
   * Create default dashboard options.
   * Default: 800x400, dark theme, sparklines enabled.
   */
  constructor();

  /**
   * Set width in pixels.
   * @param width - Width in pixels (default: 800)
   */
  width(width: number): WasmDashboardOptions;

  /**
   * Set height in pixels.
   * @param height - Height in pixels (default: 400)
   */
  height(height: number): WasmDashboardOptions;

  /**
   * Set background color.
   * @param color - Hex color (e.g., '#1a1a2e')
   */
  background_color(color: string): WasmDashboardOptions;

  /**
   * Set loss curve color.
   * @param color - Hex color (e.g., '#ff6b6b')
   */
  loss_color(color: string): WasmDashboardOptions;

  /**
   * Set accuracy curve color.
   * @param color - Hex color (e.g., '#4ecdc4')
   */
  accuracy_color(color: string): WasmDashboardOptions;

  /**
   * Enable or disable sparklines.
   * @param show - Whether to show sparklines (default: true)
   */
  show_sparklines(show: boolean): WasmDashboardOptions;
}

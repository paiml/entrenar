// Demo JavaScript for Entrenar Monitor WASM
// This simulates training and displays metrics

let collector = null;
let dashboard = null;
let interval = null;
let epoch = 0;

async function init() {
  const badge = document.getElementById('status-badge');

  try {
    // Import WASM module (built with wasm-pack)
    const wasm = await import('./pkg/entrenar_wasm.js');
    await wasm.default();

    collector = new wasm.MetricsCollector();
    dashboard = collector;  // Same object handles both

    badge.textContent = 'WASM Ready';
    badge.className = 'badge ready';

    setupControls();
  } catch (e) {
    badge.textContent = 'WASM Error';
    console.error('Failed to load WASM:', e);
  }
}

function setupControls() {
  document.getElementById('btn-start').onclick = startTraining;
  document.getElementById('btn-stop').onclick = stopTraining;
  document.getElementById('btn-clear').onclick = clearMetrics;
}

function startTraining() {
  if (interval) return;

  interval = setInterval(() => {
    epoch++;

    // Simulate training metrics
    const loss = 1.0 / (epoch * 0.1 + 1) + Math.random() * 0.1;
    const accuracy = Math.min(0.99, 0.5 + epoch * 0.005 + Math.random() * 0.02);

    collector.record_loss(loss);
    collector.record_accuracy(accuracy);

    updateDisplay();
  }, 100);
}

function stopTraining() {
  if (interval) {
    clearInterval(interval);
    interval = null;
  }
}

function clearMetrics() {
  stopTraining();
  epoch = 0;
  collector.clear();
  updateDisplay();
}

function updateDisplay() {
  const lossMean = collector.loss_mean();
  const accMean = collector.accuracy_mean();

  document.getElementById('loss-value').textContent =
    isNaN(lossMean) ? '-' : lossMean.toFixed(4);
  document.getElementById('acc-value').textContent =
    isNaN(accMean) ? '-' : (accMean * 100).toFixed(1) + '%';

  document.getElementById('loss-sparkline').textContent = collector.loss_sparkline();
  document.getElementById('acc-sparkline').textContent = collector.accuracy_sparkline();

  renderCanvas();
}

function renderCanvas() {
  const canvas = document.getElementById('chart-canvas');
  const ctx = canvas.getContext('2d');
  const state = JSON.parse(collector.state_json());

  // Clear
  ctx.fillStyle = '#0f0f1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (state.loss_history.length < 2) return;

  // Draw loss line
  drawLine(ctx, state.loss_history, state.loss_color, canvas);

  // Draw accuracy line
  drawLine(ctx, state.accuracy_history, state.accuracy_color, canvas);
}

function drawLine(ctx, values, color, canvas) {
  if (values.length < 2) return;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * canvas.width;
    const y = canvas.height - ((v - min) / range) * (canvas.height - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

init();

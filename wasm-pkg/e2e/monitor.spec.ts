import { test, expect } from '@playwright/test';

test.describe('Entrenar Monitor WASM Demo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WASM to initialize
    await page.waitForFunction(() => {
      const badge = document.getElementById('status-badge');
      return badge && badge.textContent?.includes('Ready');
    }, { timeout: 15000 });
  });

  test('loads and shows WASM ready badge', async ({ page }) => {
    const badge = page.locator('#status-badge');
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText('WASM Ready');
    await expect(badge).toHaveClass(/ready/);
  });

  test('page has correct title', async ({ page }) => {
    await expect(page).toHaveTitle('Entrenar Monitor Dashboard');
  });

  test('control buttons exist', async ({ page }) => {
    await expect(page.locator('#btn-start')).toBeVisible();
    await expect(page.locator('#btn-stop')).toBeVisible();
    await expect(page.locator('#btn-clear')).toBeVisible();
  });

  test('start training updates metrics', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(500);

    const lossValue = await page.locator('#loss-value').textContent();
    expect(lossValue).not.toBe('-');
    expect(parseFloat(lossValue || '0')).toBeGreaterThan(0);
  });

  test('start training updates sparklines', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(500);

    const sparkline = await page.locator('#loss-sparkline').textContent();
    expect(sparkline?.length).toBeGreaterThan(0);
  });

  test('stop training halts updates', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(300);
    await page.locator('#btn-stop').click();

    const valueBefore = await page.locator('#loss-value').textContent();
    await page.waitForTimeout(300);
    const valueAfter = await page.locator('#loss-value').textContent();

    expect(valueBefore).toBe(valueAfter);
  });

  test('clear resets metrics', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(300);
    await page.locator('#btn-clear').click();

    const lossValue = await page.locator('#loss-value').textContent();
    expect(lossValue).toBe('-');
  });

  test('canvas exists for chart', async ({ page }) => {
    const canvas = page.locator('#chart-canvas');
    await expect(canvas).toBeVisible();
  });

  test('accuracy displays as percentage', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(500);

    const accValue = await page.locator('#acc-value').textContent();
    expect(accValue).toContain('%');
  });

  test('screenshot: initial state', async ({ page }) => {
    await expect(page).toHaveScreenshot('monitor-initial.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.10,
    });
  });

  test('screenshot: during training', async ({ page }) => {
    await page.locator('#btn-start').click();
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('monitor-training.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.15,
    });
  });
});

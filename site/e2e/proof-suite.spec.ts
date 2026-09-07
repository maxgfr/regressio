import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("runs every exported API in a real browser", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#api-count")).toHaveText("36 / 36 APIs", { timeout: 30_000 });
  await expect(page.locator("#failed-count")).toHaveText("0 failed");
  await expect(page.locator("#engine-status")).toHaveText("WASM active");
  await expect(page.locator(".trace--ols")).toHaveAttribute("d", /M/);

  for (const family of ["Regression", "Classification", "Diagnostics", "Preprocessing", "Intervals", "Matrix"]) {
    const tab = page.getByRole("tab", { name: family });
    await expect(tab).toHaveCount(1);
    await tab.click();
    await expect(tab).toHaveCSS("animation-name", "control-punch");
    await expect(page.locator("#all-checks tr")).not.toHaveCount(0);
    await expect(page.locator("#all-checks .status--fail")).toHaveCount(0);
  }
});

test("has no automatically detectable WCAG violations", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#api-count")).toHaveText("36 / 36 APIs", { timeout: 30_000 });
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("keeps every scenario working when WASM is unavailable", async ({ page }) => {
  await page.route(/\.wasm(?:\?|$)/, (route) => route.abort());
  await page.goto("./");
  await expect(page.locator("#failed-count")).toHaveText(/\d+ failed/, { timeout: 30_000 });
  const failedChecks: string[] = [];
  for (const family of [
    "Regression",
    "Classification",
    "Diagnostics",
    "Preprocessing",
    "Intervals",
    "Matrix",
  ]) {
    await page.getByRole("tab", { name: family }).click();
    failedChecks.push(...(await page.locator("#all-checks tr:has(.status--fail)").allTextContents()));
  }
  expect(failedChecks).toEqual([]);
  await expect(page.locator("#api-count")).toHaveText("36 / 36 APIs");
  await expect(page.locator("#failed-count")).toHaveText("0 failed");
  await expect(page.locator("#engine-status")).toHaveText("TypeScript fallback");
});

test("opens an API family from its shareable URL", async ({ page }) => {
  await page.goto("./#lab/matrix");
  await expect(page.getByRole("tab", { name: "Matrix" })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#checks-caption")).toHaveText("Matrix API checks");
});

test("replays the proof animation and keeps family selection visible", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#api-count")).toHaveText("36 / 36 APIs", { timeout: 30_000 });

  await page.getByRole("button", { name: "Run suite" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-suite", "running");
  await expect(page.locator(".plot-sweep")).toHaveCSS("animation-name", "plot-sweep");
  await expect(page.getByRole("button", { name: "Run suite" })).toBeDisabled();
  await expect(page.locator("html")).toHaveAttribute("data-suite", "passed", { timeout: 5_000 });

  await page.getByRole("tab", { name: "Classification" }).click();
  await expect(page.getByRole("tab", { name: "Classification" })).toHaveCSS(
    "animation-name",
    "control-punch",
  );
  await expect(page.locator("#lab-panel")).toHaveCSS("animation-name", "lab-enter");
  await expect(page.locator("#all-checks tr").first()).toHaveCSS("animation-name", "row-print");
  await expect(page.locator(".code-proof pre")).toHaveCSS("animation-name", "code-print");
  await expect(page.locator("#checks-caption")).toHaveText("Classification API checks");

  const copyButton = page.locator("#copy-code");
  await copyButton.click();
  await expect(copyButton).toHaveClass(/is-(copied|error)/);
  await expect
    .poll(() => copyButton.evaluate((element) => getComputedStyle(element).animationName))
    .toMatch(/copy-(confirm|error)/);
});

test("keeps the completed plot legible with reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./");
  await expect(page.locator("html")).toHaveAttribute("data-suite", "passed", { timeout: 30_000 });

  const point = page.locator(".plot-points circle").first();
  await expect(point).toHaveCSS("animation-name", "none");
  await expect(point).toHaveCSS("opacity", "1");

  const diagnosticsTab = page.getByRole("tab", { name: "Diagnostics" });
  await diagnosticsTab.click();
  await expect(diagnosticsTab).toHaveCSS("animation-name", "none");
  await expect(page.locator("#lab-panel")).toHaveCSS("animation-name", "none");
  await expect(page.locator("#all-checks tr").first()).toHaveCSS("animation-name", "none");
  await expect(page.locator(".code-proof pre")).toHaveCSS("animation-name", "none");

  const copyButton = page.locator("#copy-code");
  await copyButton.click();
  await expect(copyButton).toHaveCSS("animation-name", "none");
});

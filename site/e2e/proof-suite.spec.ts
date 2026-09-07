import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("runs every exported API in a real browser", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#api-count")).toHaveText("36 / 36 APIs", { timeout: 30_000 });
  await expect(page.locator("#failed-count")).toHaveText("0 failed");
  await expect(page.locator("#engine-status")).toHaveText("WASM active");
  await expect(page.locator(".trace--ols")).toHaveAttribute("d", /M/);

  for (const family of ["Regression", "Classification", "Diagnostics", "Preprocessing", "Intervals", "Matrix"]) {
    await page.getByRole("tab", { name: family }).click();
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

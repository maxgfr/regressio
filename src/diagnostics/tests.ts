import { chi2TestPValue } from "../core/distributions";
import { LinearRegression } from "../models/linear-regression";
import type { DataMatrix, DataVector, TestResult } from "../types";

/**
 * Durbin-Watson test for autocorrelation in residuals.
 * DW ∈ [0,4], ~2 = no autocorrelation, <2 = positive, >2 = negative.
 */
export function durbinWatson(residuals: DataVector): TestResult {
  const n = residuals.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    den += residuals[i]! * residuals[i]!;
    if (i > 0) {
      const diff = residuals[i]! - residuals[i - 1]!;
      num += diff * diff;
    }
  }
  const statistic = den > 0 ? num / den : 2;

  // Approximate p-value via normal approximation for large samples
  // Under H0 (no autocorrelation), DW ≈ 2
  // This is a rough approximation; exact tables exist but are complex
  const pValue = Math.exp(-Math.abs(statistic - 2) * Math.sqrt(n) * 0.5);

  return { statistic, pValue };
}

/**
 * Breusch-Pagan test for heteroscedasticity.
 * Regresses squared residuals on X, uses Chi² test statistic = n * R².
 */
export function breuschPagan(X: DataMatrix, residuals: DataVector): TestResult {
  const n = residuals.length;
  const squaredResiduals = residuals.map((r) => r * r);

  // Regress squared residuals on X
  const auxModel = new LinearRegression();
  auxModel.fit(X, squaredResiduals);
  const stats = auxModel.statistics();

  const statistic = n * stats.rSquared;
  const df = X[0]!.length;
  const pValue = chi2TestPValue(statistic, df);

  return { statistic, pValue };
}

/**
 * Shapiro-Wilk test for normality (simplified version for n ≤ 5000).
 * Tests H0: data comes from a normal distribution.
 */
export function shapiroWilk(data: DataVector): TestResult {
  const n = data.length;
  if (n < 3) throw new Error("Shapiro-Wilk requires at least 3 observations");

  // Sort data
  const sorted = [...data].sort((a, b) => a - b);

  // Compute mean
  let mean = 0;
  for (const x of sorted) mean += x;
  mean /= n;

  // Compute SS (total sum of squares)
  let ss = 0;
  for (const x of sorted) ss += (x - mean) ** 2;

  if (ss < 1e-15) {
    // All values identical
    return { statistic: 1, pValue: 1 };
  }

  // Compute expected normal order statistics and coefficients
  // Simplified: use approximation based on inverse normal CDF
  const { normalInverseCDF } =
    require("../core/distributions") as typeof import("../core/distributions");
  const m: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = (i + 1 - 0.375) / (n + 0.25);
    m.push(normalInverseCDF(p));
  }

  // Normalize m
  let mSqSum = 0;
  for (const mi of m) mSqSum += mi * mi;
  const mNorm = Math.sqrt(mSqSum);

  const a = m.map((mi) => mi / mNorm);

  // W statistic: W = (Σ a_i * x_(i))² / SS
  let aTimesX = 0;
  for (let i = 0; i < n; i++) {
    aTimesX += a[i]! * sorted[i]!;
  }
  const statistic = (aTimesX * aTimesX) / ss;

  // Approximate p-value via transformation to normal
  // Using Royston's approximation (simplified)
  const lnW = Math.log(1 - statistic);
  const mu = -1.2725 + 1.0521 * Math.log(n);
  const sigma = 1.0308 - 0.26758 * Math.log(n);
  const z = (lnW - mu) / sigma;
  const { normalCDF: normCDF } =
    require("../core/distributions") as typeof import("../core/distributions");
  const pValue = 1 - normCDF(z);

  return { statistic, pValue: Math.max(0, Math.min(1, pValue)) };
}

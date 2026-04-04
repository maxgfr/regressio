import type { DataInput, DataVector, RobustOptions } from "../types";
import { BaseRegression } from "./base";
import { LinearRegression } from "./linear-regression";
import { WeightedRegression } from "./weighted-regression";

export class RobustRegression extends BaseRegression {
  private _method: "huber" | "tukey";
  private _tuningConstant: number;
  private _maxIterations: number;
  private _tolerance: number;

  constructor(options: RobustOptions = {}) {
    super(options);
    this._method = options.method ?? "huber";
    this._tuningConstant = options.tuningConstant ?? (this._method === "huber" ? 1.345 : 4.685);
    this._maxIterations = options.maxIterations ?? 50;
    this._tolerance = options.tolerance ?? 1e-4;
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    this._X = Xmat;
    this._y = y;

    // Step 1: Initial OLS fit
    const ols = new LinearRegression({ fitIntercept: this._fitIntercept });
    ols.fit(Xmat, y);
    this._coefficients = ols.coefficients;
    this._intercept = ols.intercept;

    // IRLS loop
    for (let iter = 0; iter < this._maxIterations; iter++) {
      const prevCoeffs = [...this._coefficients];
      const prevIntercept = this._intercept;

      // Compute residuals
      const yHat = this.predict(Xmat);
      const residuals = y.map((yi, i) => yi - yHat[i]!);

      // Estimate scale: MAD (median absolute deviation) / 0.6745
      const absResiduals = residuals.map(Math.abs).sort((a, b) => a - b);
      const mad = median(absResiduals);
      const scale = Math.max(mad / 0.6745, 1e-10);

      // Compute weights
      const weights = residuals.map((r) => {
        const u = r / scale;
        return this._method === "huber"
          ? huberWeight(u, this._tuningConstant)
          : tukeyWeight(u, this._tuningConstant);
      });

      // Refit with WLS
      const wls = new WeightedRegression({ fitIntercept: this._fitIntercept });
      wls.fit(Xmat, y, weights);
      this._coefficients = wls.coefficients;
      this._intercept = wls.intercept;

      // Check convergence
      let maxChange = Math.abs(this._intercept - prevIntercept);
      for (let j = 0; j < this._coefficients.length; j++) {
        maxChange = Math.max(maxChange, Math.abs(this._coefficients[j]! - prevCoeffs[j]!));
      }
      if (maxChange < this._tolerance) break;
    }

    this._yHat = this.predict(Xmat);
    this._fitted = true;
    return this;
  }

  predict(X: DataInput): DataVector {
    const Xmat = this.normalizeInput(X);
    return Xmat.map((row) => {
      let sum = this._intercept;
      for (let j = 0; j < this._coefficients.length; j++) {
        sum += row[j]! * this._coefficients[j]!;
      }
      return sum;
    });
  }
}

function huberWeight(u: number, k: number): number {
  return Math.abs(u) <= k ? 1 : k / Math.abs(u);
}

function tukeyWeight(u: number, k: number): number {
  if (Math.abs(u) > k) return 0;
  const t = 1 - (u / k) ** 2;
  return t * t;
}

function median(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

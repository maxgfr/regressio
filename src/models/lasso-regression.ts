import { engineCoordinateDescent } from "../core/engine";
import type { DataInput, DataMatrix, DataVector, LassoOptions } from "../types";
import { BaseRegression } from "./base";

export class LassoRegression extends BaseRegression {
  protected _alpha: number;
  protected _maxIterations: number;
  protected _tolerance: number;

  constructor(options: LassoOptions = {}) {
    super(options);
    this._alpha = options.alpha ?? 1.0;
    this._maxIterations = options.maxIterations ?? 1000;
    this._tolerance = options.tolerance ?? 1e-4;
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    this._X = Xmat;
    this._y = y;

    const n = Xmat.length;
    const p = Xmat[0]!.length;

    // WASM fast path — full coordinate descent in Rust
    const xFlat = new Float64Array(n * p);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) xFlat[i * p + j] = Xmat[i]![j]!;

    const wasmResult = engineCoordinateDescent(
      xFlat,
      new Float64Array(y),
      this._alpha,
      this.getL1Ratio(),
      this._maxIterations,
      this._tolerance,
      n,
      p,
      this._fitIntercept,
    );
    if (wasmResult) {
      this._intercept = wasmResult[0]!;
      this._coefficients = Array.from(wasmResult.slice(1));
      this._yHat = this.predict(Xmat);
      this._fitted = true;
      return this;
    }

    // TypeScript fallback
    const { Xstd, xMeans, xStds, yMean } = this.standardize(Xmat, y);
    const yCentered = y.map((yi) => yi - yMean);

    const beta = new Float64Array(p);
    const residual = new Float64Array(yCentered);

    const colNormsSq = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += Xstd[i]![j]! * Xstd[i]![j]!;
      }
      colNormsSq[j] = sum;
    }

    for (let iter = 0; iter < this._maxIterations; iter++) {
      let maxChange = 0;

      for (let j = 0; j < p; j++) {
        const oldBeta = beta[j]!;

        let rho = 0;
        for (let i = 0; i < n; i++) {
          rho += Xstd[i]![j]! * (residual[i]! + oldBeta * Xstd[i]![j]!);
        }

        beta[j] = this.coordinateUpdate(rho, colNormsSq[j]!, n, j);

        const change = beta[j]! - oldBeta;
        if (change !== 0) {
          for (let i = 0; i < n; i++) {
            residual[i] -= change * Xstd[i]![j]!;
          }
        }

        maxChange = Math.max(maxChange, Math.abs(change));
      }

      if (maxChange < this._tolerance) break;
    }

    this._coefficients = Array.from(beta).map((bj, j) => {
      const std = xStds[j]!;
      return std > 1e-15 ? bj / std : 0;
    });
    this._intercept = this._fitIntercept
      ? yMean - this._coefficients.reduce((sum, bj, j) => sum + bj * xMeans[j]!, 0)
      : 0;

    this._yHat = this.predict(Xmat);
    this._fitted = true;
    return this;
  }

  protected getL1Ratio(): number {
    return 1.0;
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

  protected coordinateUpdate(rho: number, colNormSq: number, n: number, _j: number): number {
    return this.softThreshold(rho, n * this._alpha) / colNormSq;
  }

  protected softThreshold(rho: number, lambda: number): number {
    if (rho < -lambda) return rho + lambda;
    if (rho > lambda) return rho - lambda;
    return 0;
  }

  protected standardize(
    X: DataMatrix,
    y: DataVector,
  ): { Xstd: DataMatrix; xMeans: number[]; xStds: number[]; yMean: number } {
    const n = X.length;
    const p = X[0]!.length;
    const xMeans: number[] = [];
    const xStds: number[] = [];

    // Compute means
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += X[i]![j]!;
      xMeans.push(sum / n);
    }

    // Compute stds
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const diff = X[i]![j]! - xMeans[j]!;
        sum += diff * diff;
      }
      xStds.push(Math.sqrt(sum / n));
    }

    // Standardize
    const Xstd: DataMatrix = X.map((row) =>
      row.map((val, j) => {
        const std = xStds[j]!;
        return std > 1e-15 ? (val - xMeans[j]!) / std : 0;
      }),
    );

    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += y[i]!;
    yMean /= n;

    return { Xstd, xMeans, xStds, yMean };
  }
}

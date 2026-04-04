import { backSubstitution, qrDecomposition } from "../core/decompositions";
import { fTestPValue, tInverseCDF, tTestPValue } from "../core/distributions";
import { Matrix } from "../core/matrix";
import type {
  BaseModelOptions,
  DataInput,
  DataMatrix,
  DataVector,
  RegressionStatistics,
} from "../types";

export abstract class BaseRegression {
  protected _coefficients: number[] = [];
  protected _intercept = 0;
  protected _fitted = false;
  protected _fitIntercept: boolean;

  protected _X: DataMatrix = [];
  protected _y: DataVector = [];
  protected _yHat: DataVector = [];

  constructor(options: BaseModelOptions = {}) {
    this._fitIntercept = options.fitIntercept ?? true;
  }

  get coefficients(): number[] {
    this.assertFitted();
    return this._coefficients;
  }

  get intercept(): number {
    this.assertFitted();
    return this._intercept;
  }

  abstract fit(X: DataInput, y: DataVector): this;
  abstract predict(X: DataInput): DataVector;

  residuals(): number[] {
    this.assertFitted();
    return this._y.map((yi, i) => yi - this._yHat[i]!);
  }

  statistics(): RegressionStatistics {
    this.assertFitted();

    const n = this._y.length;
    const p = this._coefficients.length; // number of predictors (excluding intercept)
    const k = p + (this._fitIntercept ? 1 : 0); // total parameters
    const dfResidual = n - k;

    if (dfResidual <= 0) {
      throw new Error("Not enough observations for statistical inference");
    }

    // Mean of y
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += this._y[i]!;
    yMean /= n;

    // RSS and TSS
    let rss = 0;
    let tss = 0;
    for (let i = 0; i < n; i++) {
      const residual = this._y[i]! - this._yHat[i]!;
      rss += residual * residual;
      const diff = this._y[i]! - yMean;
      tss += diff * diff;
    }

    const rSquared = 1 - rss / tss;
    const adjustedRSquared = 1 - ((1 - rSquared) * (n - 1)) / dfResidual;
    const mse = rss / dfResidual;
    const residualStandardError = Math.sqrt(mse);

    // Standard errors: SE = sqrt(diag(MSE * (X^T X)^{-1}))
    // Use QR to compute (X^T X)^{-1} = R^{-1} * R^{-T}
    const Xdesign = this._fitIntercept ? this.addInterceptColumn(this._X) : this._X;
    const A = Matrix.fromArray(Xdesign);
    const { R } = qrDecomposition(A);
    const Rsq = R.submatrix(0, k, 0, k);

    // Compute R^{-1} by solving R * Rinv = I column by column
    const Rinv = Matrix.zeros(k, k);
    for (let j = 0; j < k; j++) {
      const ej = Matrix.zeros(k, 1);
      ej.set(j, 0, 1);
      const col = backSubstitution(Rsq, ej);
      for (let i = 0; i < k; i++) {
        Rinv.set(i, j, col.get(i, 0));
      }
    }

    // (X^T X)^{-1} = R^{-1} * R^{-T}
    const XtXinv = Rinv.multiply(Rinv.transpose());

    // Extract standard errors, t-stats, p-values, confidence intervals
    const standardErrors: number[] = [];
    const tStatistics: number[] = [];
    const pValues: number[] = [];
    const confidenceIntervals: [number, number][] = [];
    const tCrit = tInverseCDF(0.975, dfResidual);

    const allCoeffs = this._fitIntercept
      ? [this._intercept, ...this._coefficients]
      : [...this._coefficients];

    for (let i = 0; i < k; i++) {
      const se = Math.sqrt(mse * XtXinv.get(i, i));
      const coeff = allCoeffs[i]!;
      const tStat = coeff / se;
      const pVal = tTestPValue(tStat, dfResidual);

      standardErrors.push(se);
      tStatistics.push(tStat);
      pValues.push(pVal);
      confidenceIntervals.push([coeff - tCrit * se, coeff + tCrit * se]);
    }

    // F-statistic
    const ess = tss - rss;
    const fStatistic = p > 0 ? ess / p / (rss / dfResidual) : 0;
    const fPValue = p > 0 ? fTestPValue(fStatistic, p, dfResidual) : 1;

    // AIC / BIC
    const aic = n * Math.log(rss / n) + 2 * k;
    const bic = n * Math.log(rss / n) + Math.log(n) * k;

    return {
      rSquared,
      adjustedRSquared,
      standardErrors,
      tStatistics,
      pValues,
      confidenceIntervals,
      fStatistic,
      fPValue,
      residualStandardError,
      aic,
      bic,
      degreesOfFreedom: dfResidual,
      nObservations: n,
    };
  }

  summary(): string {
    const stats = this.statistics();
    const allCoeffs = this._fitIntercept
      ? [this._intercept, ...this._coefficients]
      : [...this._coefficients];
    const names = this._fitIntercept
      ? ["(Intercept)", ...this._coefficients.map((_, i) => `x${i + 1}`)]
      : this._coefficients.map((_, i) => `x${i + 1}`);

    const lines: string[] = ["Coefficients:"];
    lines.push(
      padRight("", 15) +
        padRight("Estimate", 12) +
        padRight("Std. Error", 12) +
        padRight("t value", 10) +
        padRight("Pr(>|t|)", 12) +
        "",
    );

    for (let i = 0; i < allCoeffs.length; i++) {
      const stars = significanceStars(stats.pValues[i]!);
      lines.push(
        padRight(names[i]!, 15) +
          padRight(allCoeffs[i]!.toFixed(4), 12) +
          padRight(stats.standardErrors[i]!.toFixed(4), 12) +
          padRight(stats.tStatistics[i]!.toFixed(2), 10) +
          padRight(stats.pValues[i]!.toFixed(4), 12) +
          stars,
      );
    }

    lines.push("---");
    lines.push(`Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1`);
    lines.push("");
    lines.push(
      `Residual standard error: ${stats.residualStandardError.toFixed(4)} on ${stats.degreesOfFreedom} degrees of freedom`,
    );
    lines.push(
      `Multiple R-squared: ${stats.rSquared.toFixed(4)}, Adjusted R-squared: ${stats.adjustedRSquared.toFixed(4)}`,
    );
    lines.push(
      `F-statistic: ${stats.fStatistic.toFixed(2)} on ${this._coefficients.length} and ${stats.degreesOfFreedom} DF, p-value: ${stats.fPValue.toFixed(6)}`,
    );

    return lines.join("\n");
  }

  // -------------------------------------------------------------------------
  // Protected helpers
  // -------------------------------------------------------------------------

  protected normalizeInput(X: DataInput): DataMatrix {
    if (X.length === 0) throw new Error("Input data cannot be empty");
    if (typeof X[0] === "number") {
      // 1D → 2D: each element becomes a single-column row
      return (X as number[]).map((v) => [v]);
    }
    return X as DataMatrix;
  }

  protected addInterceptColumn(X: DataMatrix): DataMatrix {
    return X.map((row) => [1, ...row]);
  }

  protected validateFitInput(X: DataMatrix, y: DataVector): void {
    if (X.length !== y.length) {
      throw new Error(`X has ${X.length} rows but y has ${y.length} elements`);
    }
    if (X.length === 0) throw new Error("Input data cannot be empty");
  }

  protected assertFitted(): void {
    if (!this._fitted) {
      throw new Error("Model has not been fitted. Call fit() first.");
    }
  }
}

function padRight(s: string, len: number): string {
  return s.length >= len ? s : s + " ".repeat(len - s.length);
}

function significanceStars(p: number): string {
  if (p < 0.001) return " ***";
  if (p < 0.01) return " **";
  if (p < 0.05) return " *";
  if (p < 0.1) return " .";
  return "";
}

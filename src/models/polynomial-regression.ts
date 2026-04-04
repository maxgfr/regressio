import type { DataInput, DataMatrix, DataVector, PolynomialOptions } from "../types";
import { BaseRegression } from "./base";
import { LinearRegression } from "./linear-regression";

export class PolynomialRegression extends BaseRegression {
  private _degree: number;
  private _inner: LinearRegression;

  constructor(options: PolynomialOptions = {}) {
    super(options);
    this._degree = options.degree ?? 2;
    this._inner = new LinearRegression({ fitIntercept: this._fitIntercept });
  }

  private expandFeatures(X: DataMatrix): DataMatrix {
    return X.map((row) => {
      const x = row[0]!;
      const expanded: number[] = [];
      for (let d = 1; d <= this._degree; d++) {
        expanded.push(x ** d);
      }
      return expanded;
    });
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    if (Xmat[0]!.length !== 1) {
      throw new Error(
        "PolynomialRegression expects a single feature. Use LinearRegression with manual feature engineering for multiple features.",
      );
    }

    this._X = Xmat;
    this._y = y;

    const Xexpanded = this.expandFeatures(Xmat);
    this._inner.fit(Xexpanded, y);

    this._coefficients = this._inner.coefficients;
    this._intercept = this._inner.intercept;
    this._yHat = this.predict(Xmat);
    this._fitted = true;
    return this;
  }

  predict(X: DataInput): DataVector {
    const Xmat = this.normalizeInput(X);
    const Xexpanded = this.expandFeatures(Xmat);
    return this._inner.predict(Xexpanded);
  }
}

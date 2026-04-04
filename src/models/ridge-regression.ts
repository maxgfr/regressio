import { choleskyDecomposition, solveCholesky } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import type { DataInput, DataVector, RegularizedOptions } from "../types";
import { BaseRegression } from "./base";

export class RidgeRegression extends BaseRegression {
  private _alpha: number;

  constructor(options: RegularizedOptions = {}) {
    super(options);
    this._alpha = options.alpha ?? 1.0;
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    this._X = Xmat;
    this._y = y;

    const Xdesign = this._fitIntercept ? this.addInterceptColumn(Xmat) : Xmat;
    const A = Matrix.fromArray(Xdesign);
    const b = Matrix.columnVector(y);

    // X^T X + alpha * I (don't regularize intercept)
    const XtX = A.transpose().multiply(A);
    const p = XtX.cols;
    const startIdx = this._fitIntercept ? 1 : 0;
    for (let i = startIdx; i < p; i++) {
      XtX.set(i, i, XtX.get(i, i) + this._alpha);
    }

    // X^T y
    const Xty = A.transpose().multiply(b);

    // Solve via Cholesky
    const { L } = choleskyDecomposition(XtX);
    const beta = solveCholesky(L, Xty);

    const betaArray = beta.toFlatArray();
    if (this._fitIntercept) {
      this._intercept = betaArray[0]!;
      this._coefficients = betaArray.slice(1);
    } else {
      this._intercept = 0;
      this._coefficients = betaArray;
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

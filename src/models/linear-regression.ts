import { qrDecomposition, solveQR } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import type { BaseModelOptions, DataInput, DataVector } from "../types";
import { BaseRegression } from "./base";

export class LinearRegression extends BaseRegression {
  constructor(options: BaseModelOptions = {}) {
    super(options);
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    this._X = Xmat;
    this._y = y;

    const Xdesign = this._fitIntercept ? this.addInterceptColumn(Xmat) : Xmat;
    const A = Matrix.fromArray(Xdesign);
    const b = Matrix.columnVector(y);

    const { Q, R } = qrDecomposition(A);
    const beta = solveQR(Q, R, b);

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

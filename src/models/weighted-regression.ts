import { qrDecomposition, solveQR } from "../core/decompositions";
import { Matrix } from "../core/matrix";
import type { DataInput, DataVector, WeightedOptions } from "../types";
import { BaseRegression } from "./base";

export class WeightedRegression extends BaseRegression {
  private _weights: DataVector | undefined;

  constructor(options: WeightedOptions = {}) {
    super(options);
    this._weights = options.weights;
  }

  fit(X: DataInput, y: DataVector, weights?: DataVector): this {
    const Xmat = this.normalizeInput(X);
    this.validateFitInput(Xmat, y);

    const w = weights ?? this._weights;
    if (!w) throw new Error("Weights must be provided either in constructor or fit()");
    if (w.length !== y.length) {
      throw new Error(`Weights length (${w.length}) must match y length (${y.length})`);
    }

    this._X = Xmat;
    this._y = y;

    // Transform: multiply each row of X and corresponding y by sqrt(w)
    const Xdesign = this._fitIntercept ? this.addInterceptColumn(Xmat) : Xmat;
    const sqrtW = w.map((wi) => Math.sqrt(wi));

    const Xweighted = Xdesign.map((row, i) => row.map((val) => val * sqrtW[i]!));
    const yWeighted = y.map((yi, i) => yi * sqrtW[i]!);

    const A = Matrix.fromArray(Xweighted);
    const b = Matrix.columnVector(yWeighted);

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

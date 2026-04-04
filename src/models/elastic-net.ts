import type { ElasticNetOptions } from "../types";
import { LassoRegression } from "./lasso-regression";

export class ElasticNet extends LassoRegression {
  private _l1Ratio: number;

  constructor(options: ElasticNetOptions = {}) {
    super({
      fitIntercept: options.fitIntercept,
      alpha: options.alpha,
      maxIterations: options.maxIterations,
      tolerance: options.tolerance,
    });
    this._l1Ratio = options.l1Ratio ?? 0.5;
  }

  protected override coordinateUpdate(
    rho: number,
    colNormSq: number,
    n: number,
    _j: number,
  ): number {
    const l1Penalty = n * this._alpha * this._l1Ratio;
    const l2Penalty = n * this._alpha * (1 - this._l1Ratio);
    return this.softThreshold(rho, l1Penalty) / (colNormSq + l2Penalty);
  }
}

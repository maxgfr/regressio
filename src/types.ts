/** A 1D array of numbers (single feature or response vector). */
export type DataVector = number[];

/** A 2D array of numbers (multiple features, rows = observations, cols = features). */
export type DataMatrix = number[][];

/** Input data: 1D (single feature) auto-converts to 2D internally. */
export type DataInput = DataVector | DataMatrix;

// ---------------------------------------------------------------------------
// Model options
// ---------------------------------------------------------------------------

export interface BaseModelOptions {
  /** Whether to fit an intercept term (default: true). */
  fitIntercept?: boolean;
}

export interface RegularizedOptions extends BaseModelOptions {
  /** Regularization strength (default: 1.0). */
  alpha?: number;
}

export interface ElasticNetOptions extends RegularizedOptions {
  /** L1 ratio: 1 = pure Lasso, 0 = pure Ridge (default: 0.5). */
  l1Ratio?: number;
  /** Maximum iterations for coordinate descent (default: 1000). */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-4). */
  tolerance?: number;
}

export interface LassoOptions extends RegularizedOptions {
  /** Maximum iterations for coordinate descent (default: 1000). */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-4). */
  tolerance?: number;
}

export interface PolynomialOptions extends BaseModelOptions {
  /** Polynomial degree (default: 2). */
  degree?: number;
}

export interface WeightedOptions extends BaseModelOptions {
  /** Observation weights — higher = more reliable. */
  weights?: DataVector;
}

export interface RobustOptions extends BaseModelOptions {
  /** M-estimator method (default: "huber"). */
  method?: "huber" | "tukey";
  /** Tuning constant. Huber default: 1.345, Tukey default: 4.685. */
  tuningConstant?: number;
  /** Maximum IRLS iterations (default: 50). */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-4). */
  tolerance?: number;
}

export interface LogisticOptions extends BaseModelOptions {
  /** Maximum iterations for IRLS (default: 100). */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-6). */
  tolerance?: number;
}

// ---------------------------------------------------------------------------
// Fit result
// ---------------------------------------------------------------------------

export interface FitResult {
  /** Coefficients (excluding intercept). */
  coefficients: number[];
  /** Intercept term (0 if fitIntercept is false). */
  intercept: number;
}

// ---------------------------------------------------------------------------
// Regression statistics (linear models)
// ---------------------------------------------------------------------------

export interface RegressionStatistics {
  /** Coefficient of determination. */
  rSquared: number;
  /** Adjusted R-squared (penalises extra predictors). */
  adjustedRSquared: number;
  /** Standard errors of coefficients. */
  standardErrors: number[];
  /** t-statistics for each coefficient. */
  tStatistics: number[];
  /** Two-tailed p-values for each coefficient. */
  pValues: number[];
  /** 95 % confidence intervals for each coefficient [lower, upper]. */
  confidenceIntervals: [number, number][];
  /** F-statistic for overall model significance. */
  fStatistic: number;
  /** p-value for the F-statistic. */
  fPValue: number;
  /** Residual standard error (σ̂). */
  residualStandardError: number;
  /** Akaike Information Criterion. */
  aic: number;
  /** Bayesian Information Criterion. */
  bic: number;
  /** Residual degrees of freedom (n − p − 1). */
  degreesOfFreedom: number;
  /** Number of observations. */
  nObservations: number;
}

// ---------------------------------------------------------------------------
// Classification statistics (logistic regression)
// ---------------------------------------------------------------------------

export interface ConfusionMatrix {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
}

export interface ClassificationStatistics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: ConfusionMatrix;
  /** McFadden's pseudo R-squared. */
  pseudoRSquared: number;
  /** Log-likelihood of the fitted model. */
  logLikelihood: number;
  /** AIC. */
  aic: number;
  /** BIC. */
  bic: number;
}

// ---------------------------------------------------------------------------
// Diagnostics types
// ---------------------------------------------------------------------------

export interface ResidualDiagnostics {
  /** Raw residuals (y − ŷ). */
  raw: number[];
  /** Studentized residuals. */
  studentized: number[];
  /** Cook's distance for each observation. */
  cooksDistance: number[];
  /** Leverage (hat matrix diagonal) for each observation. */
  leverage: number[];
}

export interface TestResult {
  /** Test statistic value. */
  statistic: number;
  /** p-value. */
  pValue: number;
}

// ---------------------------------------------------------------------------
// Prediction types
// ---------------------------------------------------------------------------

export interface PredictionInterval {
  /** Point prediction ŷ. */
  predicted: number;
  /** Lower bound. */
  lower: number;
  /** Upper bound. */
  upper: number;
}

export interface BootstrapResult {
  /** Mean of bootstrapped coefficients. */
  coefficients: number[];
  /** Empirical confidence intervals for each coefficient [lower, upper]. */
  confidenceIntervals: [number, number][];
  /** Bootstrap standard errors. */
  standardErrors: number[];
}

// ---------------------------------------------------------------------------
// Preprocessing types
// ---------------------------------------------------------------------------

export interface ScalingParams {
  means: number[];
  stds: number[];
}

export interface NormalizationParams {
  mins: number[];
  maxs: number[];
}

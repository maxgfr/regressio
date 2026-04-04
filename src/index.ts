// Models

export { isWasmActive } from "./core/engine";
// Core (for advanced users)
export { Matrix } from "./core/matrix";
export { conditionNumber, correlationMatrix, vif } from "./diagnostics/multicollinearity";
// Diagnostics
export {
  cooksDistance,
  leverage,
  residualDiagnostics,
  studentizedResiduals,
} from "./diagnostics/residuals";
export { breuschPagan, durbinWatson, shapiroWilk } from "./diagnostics/tests";
export { ElasticNet } from "./models/elastic-net";
export type { KNNOptions } from "./models/knn";
export { KNearestNeighbors } from "./models/knn";
export { LassoRegression } from "./models/lasso-regression";
export { LinearRegression } from "./models/linear-regression";
export { LogisticRegression } from "./models/logistic-regression";
export type {
  MulticlassLogisticOptions,
  MulticlassStatistics,
} from "./models/multiclass-logistic-regression";
export { MulticlassLogisticRegression } from "./models/multiclass-logistic-regression";
export type {
  ActivationFunction,
  LayerConfig,
  NeuralNetworkOptions,
} from "./models/neural-network";
export { NeuralNetwork } from "./models/neural-network";
export { PolynomialRegression } from "./models/polynomial-regression";
export { RidgeRegression } from "./models/ridge-regression";
export { RobustRegression } from "./models/robust-regression";
export { WeightedRegression } from "./models/weighted-regression";
export { bootstrapCoefficients } from "./predictions/bootstrap";
// Predictions
export { confidenceInterval, predictionInterval } from "./predictions/intervals";
export { oneHotEncode } from "./preprocessing/encoding";
export { interactionFeatures, polynomialFeatures } from "./preprocessing/features";
export { dropMissing, imputeMean, imputeMedian } from "./preprocessing/missing";
// Preprocessing
export { normalize, standardize, unnormalize, unstandardize } from "./preprocessing/scaling";

// Types
export type {
  BaseModelOptions,
  BootstrapResult,
  ClassificationStatistics,
  ConfusionMatrix,
  DataInput,
  DataMatrix,
  DataVector,
  ElasticNetOptions,
  FitResult,
  LassoOptions,
  LogisticOptions,
  NormalizationParams,
  PolynomialOptions,
  PredictionInterval,
  RegressionStatistics,
  RegularizedOptions,
  ResidualDiagnostics,
  RobustOptions,
  ScalingParams,
  TestResult,
  WeightedOptions,
} from "./types";

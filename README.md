# regressio

Zero-dependency TypeScript regression, classification & statistics library with full statistical outputs, diagnostics, and preprocessing. Ships with an optional Rust/WASM engine for accelerated linear algebra.

## Install

```bash
bun add regressio
# or
npm install regressio
# or
pnpm add regressio
```

## Quick Start

```typescript
import { LinearRegression } from 'regressio';

const model = new LinearRegression();
model.fit([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1]);

console.log(model.coefficients);  // [2.02]
console.log(model.intercept);     // 0.06
console.log(model.predict([6]));  // [12.18]
console.log(model.summary());     // R-style formatted summary table
```

## Models

### Regression

| Model | Class | What it does |
|-------|-------|--------------|
| **OLS** | `LinearRegression` | Fits a linear relationship between features and target using Ordinary Least Squares solved via QR decomposition. The foundational regression method. |
| **Polynomial** | `PolynomialRegression` | Fits non-linear curves by expanding a single feature into polynomial terms (x, x², x³, ...) then applying OLS. |
| **Ridge (L2)** | `RidgeRegression` | Adds an L2 penalty (sum of squared coefficients) to OLS to handle multicollinearity and prevent overfitting. Shrinks coefficients toward zero but never exactly to zero. |
| **Lasso (L1)** | `LassoRegression` | Adds an L1 penalty (sum of absolute coefficients) via coordinate descent. Forces some coefficients to exactly zero, performing automatic feature selection. |
| **Elastic Net** | `ElasticNet` | Combines L1 and L2 penalties. Balances Lasso's feature selection with Ridge's stability for correlated features. |
| **WLS** | `WeightedRegression` | Weighted Least Squares. Assigns different importance to each observation. Useful when some data points are more reliable than others. |
| **Robust** | `RobustRegression` | Resistant to outliers. Uses Iteratively Reweighted Least Squares (IRLS) with Huber or Tukey bisquare M-estimators to downweight extreme values. |

### Classification

| Model | Class | What it does |
|-------|-------|--------------|
| **Logistic** | `LogisticRegression` | Binary classification (0/1). Models the probability of class membership using a sigmoid function, fitted via Newton-Raphson/IRLS. |
| **Multiclass Logistic** | `MulticlassLogisticRegression` | Extends logistic regression to K classes using softmax. Fitted via gradient descent on the cross-entropy loss. |
| **K-Nearest Neighbors** | `KNearestNeighbors` | Non-parametric method. Predicts by majority vote (classification) or mean (regression) of the k closest training points. Supports Euclidean and Manhattan distances. |

### Neural Network

| Model | Class | What it does |
|-------|-------|--------------|
| **Feedforward NN** | `NeuralNetwork` | Multi-layer perceptron with backpropagation. Configurable hidden layers, activations (relu, sigmoid, tanh, softmax), and learning rate. Supports both regression and classification tasks. |

### Usage

```typescript
import {
  LinearRegression,
  PolynomialRegression,
  RidgeRegression,
  LassoRegression,
  ElasticNet,
  WeightedRegression,
  RobustRegression,
  LogisticRegression,
  MulticlassLogisticRegression,
  KNearestNeighbors,
  NeuralNetwork,
} from 'regressio';

// --- Regression ---

// OLS: multiple regression
const ols = new LinearRegression();
ols.fit([[1, 2], [3, 4], [5, 6]], [10, 22, 34]);

// Polynomial: fit a cubic curve
const poly = new PolynomialRegression({ degree: 3 });
poly.fit([1, 2, 3, 4, 5], [1, 8, 27, 64, 125]);

// Ridge: regularized regression for correlated features
const ridge = new RidgeRegression({ alpha: 0.5 });
ridge.fit(X, y);

// Lasso: automatic feature selection
const lasso = new LassoRegression({ alpha: 0.1 });
lasso.fit(X, y);
// Some coefficients will be exactly 0

// Elastic Net: mix of L1 and L2
const enet = new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 });
enet.fit(X, y);

// Weighted Least Squares: different reliability per observation
const wls = new WeightedRegression();
wls.fit(X, y, weights);

// Robust: resistant to outliers
const robust = new RobustRegression({ method: 'huber' });
robust.fit(X, y);

// --- Classification ---

// Binary logistic regression
const logit = new LogisticRegression();
logit.fit(X, y); // y must be 0/1
logit.predictProbability(Xnew); // [0.12, 0.87, ...]

// Multiclass logistic regression (softmax)
const multi = new MulticlassLogisticRegression({ learningRate: 0.05 });
multi.fit(X, y); // y = 0, 1, 2, ...
multi.predictProbability(Xnew); // [[0.7, 0.2, 0.1], ...]

// K-Nearest Neighbors (classification or regression)
const knn = new KNearestNeighbors({ k: 5, mode: 'classification' });
knn.fit(X, y);
knn.predict(Xnew);

// --- Neural Network ---

// Regression with a neural network
const nn = new NeuralNetwork({
  layers: [
    { units: 16, activation: 'relu' },
    { units: 8, activation: 'relu' },
  ],
  learningRate: 0.01,
  epochs: 200,
  task: 'regression',
});
nn.fit(X, y);
nn.predict(Xnew);

// Classification with a neural network
const clf = new NeuralNetwork({
  layers: [{ units: 10, activation: 'sigmoid' }],
  learningRate: 0.1,
  epochs: 100,
  task: 'classification',
});
clf.fit(X, y); // y = 0, 1, 2, ...
clf.predict(Xnew);
```

## Statistical Outputs

Every linear model (OLS, Ridge, Lasso, Elastic Net, WLS, Robust, Polynomial) provides `statistics()` and `summary()`:

```typescript
const stats = model.statistics();
// {
//   rSquared,              -- proportion of variance explained (0 to 1)
//   adjustedRSquared,      -- R² penalized for number of predictors
//   standardErrors,        -- uncertainty of each coefficient estimate
//   tStatistics,           -- coefficient / standard error for each predictor
//   pValues,               -- probability of observing the t-stat under H0 (no effect)
//   confidenceIntervals,   -- 95% confidence range for each coefficient
//   fStatistic,            -- overall model significance test
//   fPValue,               -- p-value for the F-test
//   residualStandardError, -- estimated standard deviation of residuals
//   aic,                   -- Akaike Information Criterion (lower = better fit/complexity trade-off)
//   bic,                   -- Bayesian Information Criterion (stronger complexity penalty than AIC)
//   degreesOfFreedom,      -- n - k (observations minus parameters)
//   nObservations,         -- number of data points
// }

console.log(model.summary());
// Coefficients:
//                 Estimate    Std. Error  t value   Pr(>|t|)
// (Intercept)     0.0600      0.1200      0.50      0.6300
// x1              2.0200      0.0400      50.20     0.0000 ***
// ---
// Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Binary logistic regression provides classification metrics:

```typescript
const stats = logit.statistics();
// { accuracy, precision, recall, f1Score, confusionMatrix,
//   pseudoRSquared, logLikelihood, aic, bic }
```

Multiclass logistic regression provides per-class metrics:

```typescript
const stats = multi.statistics();
// { accuracy, precision (per class), recall (per class),
//   nClasses, logLikelihood }
```

## Diagnostics

Functions to validate model assumptions and detect problems.

| Function | What it does |
|----------|--------------|
| `residualDiagnostics(X, y, yHat)` | Returns raw residuals, studentized residuals, Cook's distance, and leverage for each observation. |
| `studentizedResiduals(X, y, yHat)` | Residuals scaled by their estimated standard deviation. Values > 2-3 suggest outliers. |
| `cooksDistance(X, y, yHat)` | Measures how much each observation influences the fitted model. Values > 4/n flag influential points. |
| `leverage(X)` | Hat matrix diagonal. Measures how far each observation's features are from the center. High leverage = unusual feature values. |
| `durbinWatson(residuals)` | Tests for autocorrelation in residuals. Returns statistic in [0,4]: ~2 = no autocorrelation, <2 = positive, >2 = negative. Critical for time series. |
| `breuschPagan(X, residuals)` | Tests for heteroscedasticity (non-constant variance). Low p-value = variance depends on X, meaning standard errors are unreliable. |
| `shapiroWilk(data)` | Tests whether data follows a normal distribution. Low p-value = non-normal. Important because p-values and CIs assume normal residuals. |
| `vif(X)` | Variance Inflation Factor for each feature. VIF > 10 signals multicollinearity (features are too correlated). |
| `correlationMatrix(X)` | Pairwise Pearson correlation matrix. Pairs with |r| > 0.9 suggest redundant features. |
| `conditionNumber(X)` | Ratio of largest to smallest singular value of X. Values > 30 signal numerical instability from multicollinearity. |

```typescript
import {
  residualDiagnostics, leverage, cooksDistance, studentizedResiduals,
  durbinWatson, breuschPagan, shapiroWilk,
  vif, correlationMatrix, conditionNumber,
} from 'regressio';

const diag = residualDiagnostics(X, y, yHat);
const dw = durbinWatson(model.residuals());
const bp = breuschPagan(X, model.residuals());
const sw = shapiroWilk(model.residuals());
const vifs = vif(X);
const corr = correlationMatrix(X);
const kappa = conditionNumber(X);
```

## Preprocessing

Functions to prepare data before fitting models.

| Function | What it does |
|----------|--------------|
| `standardize(X)` | Z-score normalization: transforms each feature to mean=0, std=1. Essential for Lasso/Ridge/Elastic Net and neural networks. |
| `unstandardize(X, params)` | Reverses standardization back to the original scale. |
| `normalize(X)` | Min-max scaling: transforms each feature to [0, 1] range. |
| `unnormalize(X, params)` | Reverses normalization back to the original scale. |
| `oneHotEncode(column, categories?, dropFirst?)` | Converts categorical values to binary columns. Use `dropFirst=true` to avoid the multicollinearity trap. |
| `polynomialFeatures(X, degree)` | Generates polynomial terms (x, x², x³, ...) for each feature. Use with `LinearRegression` for polynomial fitting with multiple features. |
| `interactionFeatures(X, pairs?)` | Generates interaction terms (xi * xj) for all or specified feature pairs. |
| `dropMissing(X, y?)` | Removes rows containing NaN or null values. |
| `imputeMean(X)` | Replaces NaN values with the column mean. |
| `imputeMedian(X)` | Replaces NaN values with the column median. More robust to outliers than mean imputation. |

```typescript
import {
  standardize, unstandardize, normalize, unnormalize,
  oneHotEncode, polynomialFeatures, interactionFeatures,
  dropMissing, imputeMean, imputeMedian,
} from 'regressio';

const { transformed, means, stds } = standardize(X);
const original = unstandardize(transformed, { means, stds });
const { transformed: normed, mins, maxs } = normalize(X);
const dummies = oneHotEncode(['cat', 'dog', 'cat'], undefined, true);
const polyX = polynomialFeatures(X, 3);
const interX = interactionFeatures(X);
const clean = dropMissing(X, y);
const imputed = imputeMean(X);
```

## Prediction Intervals

Functions to quantify prediction uncertainty.

| Function | What it does |
|----------|--------------|
| `confidenceInterval(X, y, yHat, newX, newYHat)` | Confidence interval on the **mean** prediction. Answers: "where is the true regression line?" Narrower near the center of the training data. |
| `predictionInterval(X, y, yHat, newX, newYHat)` | Prediction interval for a **new individual** observation. Always wider than the confidence interval because it includes observation noise. |
| `bootstrapCoefficients(X, y, nBootstrap?)` | Non-parametric bootstrap: resamples data with replacement, refits the model many times, and returns empirical confidence intervals on coefficients. No distributional assumptions. |

```typescript
import { confidenceInterval, predictionInterval, bootstrapCoefficients } from 'regressio';

const ci = confidenceInterval(X, y, yHat, newX, newYHat);
// [{ predicted, lower, upper }, ...]

const pi = predictionInterval(X, y, yHat, newX, newYHat);
// Always wider than ci

const boot = bootstrapCoefficients(X, y, 1000);
// { coefficients, confidenceIntervals, standardErrors }
```

## Advanced: Matrix Class

Low-level matrix operations for advanced users. Backed by `Float64Array` in row-major order.

```typescript
import { Matrix } from 'regressio';

const A = Matrix.fromArray([[1, 2], [3, 4]]);
const B = Matrix.identity(2);
const C = A.multiply(B);
console.log(C.determinant());  // -2
console.log(C.trace());        // 5
console.log(C.transpose().toArray());
```

## WASM Engine (Optional)

For faster matrix operations on large datasets, build and load the Rust/WASM engine. When active, `Matrix.multiply()`, QR decomposition, Cholesky decomposition, and back-substitution are dispatched to compiled Rust code.

```bash
cd rust && wasm-pack build --target bundler --out-dir ../pkg
```

```typescript
import { useWasmEngine, useTypescriptEngine, isWasmActive, getEngine } from 'regressio';

// Load WASM engine (async, loads the .wasm file)
await useWasmEngine();
console.log(isWasmActive()); // true

// All subsequent matrix operations use WASM
const model = new LinearRegression();
model.fit(X, y); // QR decomposition runs in Rust

// Switch back to pure TypeScript
useTypescriptEngine();
```

## License

MIT

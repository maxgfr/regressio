## Project

regressio is a zero-dependency TypeScript regression & statistics library with a Rust/WASM engine for acceleration.

- 11 models: 7 regression (OLS, Polynomial, Ridge, Lasso, ElasticNet, WLS, Robust), 3 classification (Logistic, Multiclass, KNN), 1 neural network
- WASM auto-loads at import time — no `useWasmEngine()` call needed
- `isWasmActive()` is the only engine export; all `use*` functions were removed
- Rust source is in `rust/src/lib.rs`, compiled via `bun run build:wasm` (needs Rust + wasm32-unknown-unknown target)
- `wasm-pack` is an npm devDependency, no system install needed
- `bunup` resolves `#wasm-engine` at build time and embeds the WASM binary in `dist/`
- The `pkg/` directory is committed and used at dev/build time; only `dist/` is published to npm
- 22 Rust WASM functions: matrix ops (multiply, transpose, add, subtract, scale, dot, norm, determinant), decompositions (QR, Cholesky, SVD, eigenvalues with tridiagonal reduction), solvers (forward/back sub), model algorithms (coordinate descent, softmax, euclidean/manhattan distances, correlation matrix, bootstrap OLS, VIF, IRLS logistic)

## Tooling

Default to using Bun instead of Node.js.

- Use `bun test` to run the 242 tests across 25 files
- Use `bun run build:wasm` to recompile Rust to WASM
- Use `bun run build` (= `bun run build:ts`) to build the TS library via bunup
- Use `bun run check` to lint + test
- Use `bun install` instead of npm/yarn/pnpm

## Architecture

```
src/
  core/           engine.ts (WASM dispatch), matrix.ts, decompositions.ts, distributions.ts
  models/         base.ts + 10 model files
  diagnostics/    residuals.ts, multicollinearity.ts, tests.ts
  preprocessing/  scaling.ts, features.ts, encoding.ts, missing.ts
  predictions/    intervals.ts, bootstrap.ts
rust/src/lib.rs   WASM-accelerated functions
pkg/              Pre-compiled WASM binaries (committed)
tests/            25 test files mirroring src/ structure
```

## Key patterns

- All regression models extend `BaseRegression` (except LogisticRegression, MulticlassLogisticRegression, KNN, NeuralNetwork which are standalone)
- Engine dispatch: each engine function returns `Float64Array | null` — null means WASM unavailable, caller falls back to TypeScript
- Matrix class uses `Float64Array` in row-major order; all decompositions work on this format

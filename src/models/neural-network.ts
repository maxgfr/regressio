import type { DataInput, DataMatrix, DataVector } from "../types";

export type ActivationFunction = "relu" | "sigmoid" | "tanh" | "linear" | "softmax";

export interface LayerConfig {
  /** Number of neurons in this layer. */
  units: number;
  /** Activation function (default: "relu"). */
  activation?: ActivationFunction;
}

export interface NeuralNetworkOptions {
  /** Hidden layer configurations. Output layer is added automatically. */
  layers: LayerConfig[];
  /** Learning rate (default: 0.01). */
  learningRate?: number;
  /** Number of training epochs (default: 100). */
  epochs?: number;
  /** Task type (default: "regression"). */
  task?: "regression" | "classification";
}

interface Layer {
  weights: Float64Array; // (inputSize x units) flat
  biases: Float64Array; // (units)
  inputSize: number;
  units: number;
  activation: ActivationFunction;
}

/**
 * Feedforward Neural Network (Multi-Layer Perceptron) with backpropagation.
 * Supports regression and classification tasks.
 */
export class NeuralNetwork {
  private _layers: Layer[] = [];
  private _learningRate: number;
  private _epochs: number;
  private _task: "regression" | "classification";
  private _fitted = false;
  private _outputSize = 0;
  private _classes: number[] = [];

  constructor(options: NeuralNetworkOptions) {
    this._learningRate = options.learningRate ?? 0.01;
    this._epochs = options.epochs ?? 100;
    this._task = options.task ?? "regression";

    // Store layer configs — actual initialization happens in fit()
    this._layerConfigs = options.layers;
  }

  private _layerConfigs: LayerConfig[];

  private initializeLayers(inputSize: number, outputSize: number): void {
    this._layers = [];
    let prevSize = inputSize;

    // Hidden layers
    for (const config of this._layerConfigs) {
      this._layers.push(this.createLayer(prevSize, config.units, config.activation ?? "relu"));
      prevSize = config.units;
    }

    // Output layer
    const outputActivation =
      this._task === "classification" && outputSize > 1 ? "softmax" : "linear";
    this._layers.push(this.createLayer(prevSize, outputSize, outputActivation));
  }

  private createLayer(inputSize: number, units: number, activation: ActivationFunction): Layer {
    // Glorot/Xavier uniform initialization: U(-scale, scale) with scale = sqrt(6 / (fan_in + fan_out))
    const scale = Math.sqrt(6 / (inputSize + units));
    const weights = new Float64Array(inputSize * units);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() * 2 - 1) * scale;
    }
    return {
      weights,
      biases: new Float64Array(units),
      inputSize,
      units,
      activation,
    };
  }

  fit(X: DataInput, y: DataVector): this {
    const Xmat = this.normalizeInput(X);
    if (Xmat.length !== y.length) {
      throw new Error(`X has ${Xmat.length} rows but y has ${y.length} elements`);
    }

    const inputSize = Xmat[0]!.length;
    const n = Xmat.length;

    if (this._task === "classification") {
      this._classes = [...new Set(y)].sort((a, b) => a - b);
      this._outputSize = this._classes.length;
    } else {
      this._outputSize = 1;
    }

    this.initializeLayers(inputSize, this._outputSize);

    // Training loop
    for (let epoch = 0; epoch < this._epochs; epoch++) {
      for (let i = 0; i < n; i++) {
        const input = Xmat[i]!;

        // Forward pass
        const activations = this.forward(input);

        // Compute output error
        const outputLayer = activations[activations.length - 1]!;
        const target = this.encodeTarget(y[i]!);
        const outputError = new Float64Array(this._outputSize);

        for (let j = 0; j < this._outputSize; j++) {
          outputError[j] = outputLayer[j]! - target[j]!;
        }

        // Backpropagation
        this.backward(activations, outputError);
      }
    }

    this._fitted = true;
    return this;
  }

  predict(X: DataInput): DataVector {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    const Xmat = this.normalizeInput(X);

    return Xmat.map((row) => {
      const activations = this.forward(row);
      const output = activations[activations.length - 1]!;

      if (this._task === "classification") {
        // Argmax
        let maxIdx = 0;
        let maxVal = output[0]!;
        for (let c = 1; c < output.length; c++) {
          if (output[c]! > maxVal) {
            maxVal = output[c]!;
            maxIdx = c;
          }
        }
        return this._classes[maxIdx]!;
      }
      return output[0]!;
    });
  }

  /** Return raw output (probabilities for classification, values for regression). */
  predictRaw(X: DataInput): number[][] {
    if (!this._fitted) throw new Error("Model has not been fitted. Call fit() first.");
    const Xmat = this.normalizeInput(X);
    return Xmat.map((row) => {
      const activations = this.forward(row);
      return Array.from(activations[activations.length - 1]!);
    });
  }

  // ---------------------------------------------------------------------------
  // Forward pass
  // ---------------------------------------------------------------------------

  private forward(input: number[]): Float64Array[] {
    const activations: Float64Array[] = [new Float64Array(input)];

    let current = new Float64Array(input);
    for (const layer of this._layers) {
      const z = new Float64Array(layer.units);

      // z = W^T * x + b
      for (let j = 0; j < layer.units; j++) {
        let sum = layer.biases[j]!;
        for (let i = 0; i < layer.inputSize; i++) {
          sum += current[i]! * layer.weights[i * layer.units + j]!;
        }
        z[j] = sum;
      }

      // Apply activation
      current = this.activate(z, layer.activation);
      activations.push(current);
    }

    return activations;
  }

  // ---------------------------------------------------------------------------
  // Backward pass (stochastic gradient descent)
  // ---------------------------------------------------------------------------

  private backward(activations: Float64Array[], outputError: Float64Array): void {
    // Compute all deltas first (before any weight updates)
    const deltas: Float64Array[] = new Array(this._layers.length);
    deltas[this._layers.length - 1] = outputError;

    for (let l = this._layers.length - 2; l >= 0; l--) {
      const layer = this._layers[l]!;
      const output = activations[l + 1]!;
      const nextLayer = this._layers[l + 1]!;
      const nextDelta = deltas[l + 1]!;

      const delta = new Float64Array(layer.units);
      for (let j = 0; j < layer.units; j++) {
        let sum = 0;
        for (let k = 0; k < nextLayer.units; k++) {
          sum += nextDelta[k]! * nextLayer.weights[j * nextLayer.units + k]!;
        }
        delta[j] = sum * this.activationDerivative(output[j]!, layer.activation);
      }
      deltas[l] = delta;
    }

    // Now update all weights using the pre-computed deltas
    const lr = this._learningRate;
    for (let l = 0; l < this._layers.length; l++) {
      const layer = this._layers[l]!;
      const input = activations[l]!;
      const delta = deltas[l]!;

      for (let j = 0; j < layer.units; j++) {
        for (let i = 0; i < layer.inputSize; i++) {
          layer.weights[i * layer.units + j] -= lr * delta[j]! * input[i]!;
        }
        layer.biases[j] -= lr * delta[j]!;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Activations
  // ---------------------------------------------------------------------------

  private activate(z: Float64Array, fn: ActivationFunction): Float64Array {
    const result = new Float64Array(z.length);
    switch (fn) {
      case "relu":
        for (let i = 0; i < z.length; i++) result[i] = Math.max(0, z[i]!);
        break;
      case "sigmoid":
        for (let i = 0; i < z.length; i++) result[i] = this.sigmoid(z[i]!);
        break;
      case "tanh":
        for (let i = 0; i < z.length; i++) result[i] = Math.tanh(z[i]!);
        break;
      case "linear":
        result.set(z);
        break;
      case "softmax": {
        const max = Math.max(...z);
        let sum = 0;
        for (let i = 0; i < z.length; i++) {
          result[i] = Math.exp(z[i]! - max);
          sum += result[i]!;
        }
        for (let i = 0; i < z.length; i++) result[i] /= sum;
        break;
      }
    }
    return result;
  }

  private activationDerivative(output: number, fn: ActivationFunction): number {
    switch (fn) {
      case "relu":
        return output > 0 ? 1 : 0;
      case "sigmoid":
        return output * (1 - output);
      case "tanh":
        return 1 - output * output;
      case "linear":
        return 1;
      case "softmax":
        return output * (1 - output); // simplified for cross-entropy loss
    }
  }

  private sigmoid(z: number): number {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const expZ = Math.exp(z);
    return expZ / (1 + expZ);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private encodeTarget(y: number): Float64Array {
    if (this._task === "regression") {
      return new Float64Array([y]);
    }
    // One-hot
    const encoded = new Float64Array(this._outputSize);
    const idx = this._classes.indexOf(y);
    if (idx >= 0) encoded[idx] = 1;
    return encoded;
  }

  private normalizeInput(X: DataInput): DataMatrix {
    if (X.length === 0) throw new Error("Input data cannot be empty");
    if (typeof X[0] === "number") {
      return (X as number[]).map((v) => [v]);
    }
    return X as DataMatrix;
  }
}

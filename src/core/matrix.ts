import { engineMatrixMultiply } from "./engine";

/**
 * Matrix class backed by a flat Float64Array in row-major order.
 * Provides all linear algebra primitives needed by the regression models.
 * Matrix multiply is dispatched to the WASM engine when available.
 */
export class Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array | number[]) {
    this.rows = rows;
    this.cols = cols;
    if (data) {
      this.data = data instanceof Float64Array ? data : new Float64Array(data);
    } else {
      this.data = new Float64Array(rows * cols);
    }
  }

  // -------------------------------------------------------------------------
  // Factory methods
  // -------------------------------------------------------------------------

  static fromArray(arr: number[][]): Matrix {
    const rows = arr.length;
    if (rows === 0) return new Matrix(0, 0);
    const cols = arr[0]!.length;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const row = arr[i]!;
      if (row.length !== cols) {
        throw new Error(`Row ${i} has ${row.length} columns, expected ${cols}`);
      }
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = row[j]!;
      }
    }
    return new Matrix(rows, cols, data);
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static ones(rows: number, cols: number): Matrix {
    const data = new Float64Array(rows * cols).fill(1);
    return new Matrix(rows, cols, data);
  }

  static identity(n: number): Matrix {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      m.data[i * n + i] = 1;
    }
    return m;
  }

  static columnVector(arr: number[]): Matrix {
    return new Matrix(arr.length, 1, new Float64Array(arr));
  }

  static rowVector(arr: number[]): Matrix {
    return new Matrix(1, arr.length, new Float64Array(arr));
  }

  static diagonal(values: number[]): Matrix {
    const n = values.length;
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      m.data[i * n + i] = values[i]!;
    }
    return m;
  }

  // -------------------------------------------------------------------------
  // Element access
  // -------------------------------------------------------------------------

  get(i: number, j: number): number {
    return this.data[i * this.cols + j]!;
  }

  set(i: number, j: number, value: number): void {
    this.data[i * this.cols + j] = value;
  }

  getColumn(j: number): Matrix {
    const col = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      col[i] = this.data[i * this.cols + j]!;
    }
    return new Matrix(this.rows, 1, col);
  }

  getRow(i: number): Matrix {
    const start = i * this.cols;
    return new Matrix(1, this.cols, this.data.slice(start, start + this.cols));
  }

  setColumn(j: number, col: Matrix): void {
    for (let i = 0; i < this.rows; i++) {
      this.data[i * this.cols + j] = col.data[i]!;
    }
  }

  // -------------------------------------------------------------------------
  // Core operations
  // -------------------------------------------------------------------------

  transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j * this.rows + i] = this.data[i * this.cols + j]!;
      }
    }
    return result;
  }

  /** Matrix multiplication — dispatched to WASM engine when available. */
  multiply(other: Matrix): Matrix {
    if (this.cols !== other.rows) {
      throw new Error(
        `Matrix multiply: incompatible dimensions (${this.rows}x${this.cols}) * (${other.rows}x${other.cols})`,
      );
    }
    const resultData = engineMatrixMultiply(
      this.data,
      this.rows,
      this.cols,
      other.data,
      other.rows,
      other.cols,
    );
    return new Matrix(this.rows, other.cols, resultData);
  }

  add(other: Matrix): Matrix {
    this.assertSameDimensions(other, "add");
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! + other.data[i]!;
    }
    return result;
  }

  subtract(other: Matrix): Matrix {
    this.assertSameDimensions(other, "subtract");
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! - other.data[i]!;
    }
    return result;
  }

  scale(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i]! * scalar;
    }
    return result;
  }

  // -------------------------------------------------------------------------
  // In-place operations
  // -------------------------------------------------------------------------

  addInPlace(other: Matrix): void {
    this.assertSameDimensions(other, "addInPlace");
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] += other.data[i]!;
    }
  }

  subtractInPlace(other: Matrix): void {
    this.assertSameDimensions(other, "subtractInPlace");
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] -= other.data[i]!;
    }
  }

  scaleInPlace(scalar: number): void {
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] *= scalar;
    }
  }

  // -------------------------------------------------------------------------
  // Properties
  // -------------------------------------------------------------------------

  /** Frobenius norm: sqrt(sum of squares). */
  norm(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      sum += this.data[i]! * this.data[i]!;
    }
    return Math.sqrt(sum);
  }

  /** Trace: sum of diagonal elements (square matrices only). */
  trace(): number {
    if (this.rows !== this.cols) {
      throw new Error("Trace is only defined for square matrices");
    }
    let sum = 0;
    for (let i = 0; i < this.rows; i++) {
      sum += this.data[i * this.cols + i]!;
    }
    return sum;
  }

  /** Determinant (square matrices only, via LU-like elimination). */
  determinant(): number {
    if (this.rows !== this.cols) {
      throw new Error("Determinant is only defined for square matrices");
    }
    const n = this.rows;
    if (n === 0) return 1;
    if (n === 1) return this.data[0]!;
    if (n === 2) return this.data[0]! * this.data[3]! - this.data[1]! * this.data[2]!;

    // Gaussian elimination with partial pivoting
    const a = new Float64Array(this.data);
    let det = 1;
    for (let col = 0; col < n; col++) {
      // Find pivot
      let maxVal = Math.abs(a[col * n + col]!);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(a[row * n + col]!);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxVal < 1e-15) return 0;

      // Swap rows
      if (maxRow !== col) {
        det = -det;
        for (let j = 0; j < n; j++) {
          const tmp = a[col * n + j]!;
          a[col * n + j] = a[maxRow * n + j]!;
          a[maxRow * n + j] = tmp;
        }
      }

      det *= a[col * n + col]!;

      // Eliminate below
      const pivot = a[col * n + col]!;
      for (let row = col + 1; row < n; row++) {
        const factor = a[row * n + col]! / pivot;
        for (let j = col + 1; j < n; j++) {
          a[row * n + j] -= factor * a[col * n + j]!;
        }
      }
    }
    return det;
  }

  // -------------------------------------------------------------------------
  // Submatrix / utility
  // -------------------------------------------------------------------------

  submatrix(rowStart: number, rowEnd: number, colStart: number, colEnd: number): Matrix {
    const rows = rowEnd - rowStart;
    const cols = colEnd - colStart;
    const result = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[i * cols + j] = this.data[(rowStart + i) * this.cols + (colStart + j)]!;
      }
    }
    return result;
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }

  toArray(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.data[i * this.cols + j]!);
      }
      result.push(row);
    }
    return result;
  }

  toFlatArray(): number[] {
    return Array.from(this.data);
  }

  /** Dot product for two column vectors. */
  dot(other: Matrix): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      sum += this.data[i]! * other.data[i]!;
    }
    return sum;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private assertSameDimensions(other: Matrix, op: string): void {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error(
        `Matrix ${op}: incompatible dimensions (${this.rows}x${this.cols}) vs (${other.rows}x${other.cols})`,
      );
    }
  }
}

import { describe, expect, test } from "bun:test";
import { Matrix } from "../../src/core/matrix";

describe("Matrix", () => {
  describe("creation", () => {
    test("fromArray creates correct matrix", () => {
      const m = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(m.rows).toBe(2);
      expect(m.cols).toBe(3);
      expect(m.get(0, 0)).toBe(1);
      expect(m.get(1, 2)).toBe(6);
    });

    test("fromArray rejects ragged arrays", () => {
      expect(() =>
        Matrix.fromArray([
          [1, 2],
          [3, 4, 5],
        ]),
      ).toThrow();
    });

    test("zeros creates zero matrix", () => {
      const m = Matrix.zeros(3, 2);
      expect(m.rows).toBe(3);
      expect(m.cols).toBe(2);
      for (let i = 0; i < 6; i++) {
        expect(m.data[i]).toBe(0);
      }
    });

    test("ones creates all-ones matrix", () => {
      const m = Matrix.ones(2, 3);
      for (let i = 0; i < 6; i++) {
        expect(m.data[i]).toBe(1);
      }
    });

    test("identity creates identity matrix", () => {
      const m = Matrix.identity(3);
      expect(m.get(0, 0)).toBe(1);
      expect(m.get(1, 1)).toBe(1);
      expect(m.get(2, 2)).toBe(1);
      expect(m.get(0, 1)).toBe(0);
      expect(m.get(1, 0)).toBe(0);
    });

    test("columnVector creates Nx1 matrix", () => {
      const m = Matrix.columnVector([1, 2, 3]);
      expect(m.rows).toBe(3);
      expect(m.cols).toBe(1);
      expect(m.get(1, 0)).toBe(2);
    });

    test("rowVector creates 1xN matrix", () => {
      const m = Matrix.rowVector([1, 2, 3]);
      expect(m.rows).toBe(1);
      expect(m.cols).toBe(3);
    });

    test("diagonal creates diagonal matrix", () => {
      const m = Matrix.diagonal([2, 3, 5]);
      expect(m.get(0, 0)).toBe(2);
      expect(m.get(1, 1)).toBe(3);
      expect(m.get(2, 2)).toBe(5);
      expect(m.get(0, 1)).toBe(0);
    });
  });

  describe("element access", () => {
    test("getColumn extracts column", () => {
      const m = Matrix.fromArray([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const col = m.getColumn(1);
      expect(col.rows).toBe(3);
      expect(col.cols).toBe(1);
      expect(col.toFlatArray()).toEqual([2, 4, 6]);
    });

    test("getRow extracts row", () => {
      const m = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const row = m.getRow(1);
      expect(row.rows).toBe(1);
      expect(row.cols).toBe(3);
      expect(row.toFlatArray()).toEqual([4, 5, 6]);
    });

    test("setColumn replaces column", () => {
      const m = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      m.setColumn(0, Matrix.columnVector([10, 20]));
      expect(m.get(0, 0)).toBe(10);
      expect(m.get(1, 0)).toBe(20);
      expect(m.get(0, 1)).toBe(2);
    });
  });

  describe("operations", () => {
    test("transpose swaps rows and columns", () => {
      const m = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const t = m.transpose();
      expect(t.rows).toBe(3);
      expect(t.cols).toBe(2);
      expect(t.get(0, 0)).toBe(1);
      expect(t.get(0, 1)).toBe(4);
      expect(t.get(2, 0)).toBe(3);
      expect(t.get(2, 1)).toBe(6);
    });

    test("multiply produces correct result", () => {
      const a = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      const b = Matrix.fromArray([
        [5, 6],
        [7, 8],
      ]);
      const c = a.multiply(b);
      expect(c.get(0, 0)).toBe(19);
      expect(c.get(0, 1)).toBe(22);
      expect(c.get(1, 0)).toBe(43);
      expect(c.get(1, 1)).toBe(50);
    });

    test("multiply validates dimensions", () => {
      const a = Matrix.fromArray([[1, 2]]);
      const b = Matrix.fromArray([[1, 2]]);
      expect(() => a.multiply(b)).toThrow("incompatible dimensions");
    });

    test("non-square matrix multiplication", () => {
      // (2x3) * (3x1) = (2x1)
      const a = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = Matrix.columnVector([1, 2, 3]);
      const c = a.multiply(b);
      expect(c.rows).toBe(2);
      expect(c.cols).toBe(1);
      expect(c.get(0, 0)).toBe(14); // 1+4+9
      expect(c.get(1, 0)).toBe(32); // 4+10+18
    });

    test("add element-wise", () => {
      const a = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      const b = Matrix.fromArray([
        [10, 20],
        [30, 40],
      ]);
      const c = a.add(b);
      expect(c.toArray()).toEqual([
        [11, 22],
        [33, 44],
      ]);
    });

    test("subtract element-wise", () => {
      const a = Matrix.fromArray([
        [10, 20],
        [30, 40],
      ]);
      const b = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      const c = a.subtract(b);
      expect(c.toArray()).toEqual([
        [9, 18],
        [27, 36],
      ]);
    });

    test("scale multiplies all elements", () => {
      const m = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      const s = m.scale(3);
      expect(s.toArray()).toEqual([
        [3, 6],
        [9, 12],
      ]);
    });

    test("add rejects dimension mismatch", () => {
      const a = Matrix.fromArray([[1, 2]]);
      const b = Matrix.fromArray([[1], [2]]);
      expect(() => a.add(b)).toThrow("incompatible dimensions");
    });
  });

  describe("in-place operations", () => {
    test("addInPlace modifies receiver", () => {
      const a = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      const b = Matrix.fromArray([
        [5, 6],
        [7, 8],
      ]);
      a.addInPlace(b);
      expect(a.toArray()).toEqual([
        [6, 8],
        [10, 12],
      ]);
    });

    test("subtractInPlace modifies receiver", () => {
      const a = Matrix.fromArray([
        [5, 6],
        [7, 8],
      ]);
      const b = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      a.subtractInPlace(b);
      expect(a.toArray()).toEqual([
        [4, 4],
        [4, 4],
      ]);
    });

    test("scaleInPlace modifies receiver", () => {
      const a = Matrix.fromArray([[2, 4]]);
      a.scaleInPlace(0.5);
      expect(a.toArray()).toEqual([[1, 2]]);
    });
  });

  describe("properties", () => {
    test("norm computes Frobenius norm", () => {
      const m = Matrix.fromArray([[3, 4]]);
      expect(m.norm()).toBeCloseTo(5, 10);
    });

    test("trace sums diagonal", () => {
      const m = Matrix.fromArray([
        [1, 2],
        [3, 4],
      ]);
      expect(m.trace()).toBe(5);
    });

    test("trace rejects non-square", () => {
      expect(() => Matrix.fromArray([[1, 2, 3]]).trace()).toThrow();
    });

    test("determinant 1x1", () => {
      expect(Matrix.fromArray([[7]]).determinant()).toBe(7);
    });

    test("determinant 2x2", () => {
      const m = Matrix.fromArray([
        [3, 8],
        [4, 6],
      ]);
      expect(m.determinant()).toBeCloseTo(3 * 6 - 8 * 4, 10);
    });

    test("determinant 3x3", () => {
      const m = Matrix.fromArray([
        [6, 1, 1],
        [4, -2, 5],
        [2, 8, 7],
      ]);
      // det = 6(-14-40) - 1(28-10) + 1(32+4) = -306
      expect(m.determinant()).toBeCloseTo(-306, 8);
    });

    test("determinant of identity is 1", () => {
      expect(Matrix.identity(4).determinant()).toBeCloseTo(1, 10);
    });

    test("determinant of singular matrix is 0", () => {
      const m = Matrix.fromArray([
        [1, 2],
        [2, 4],
      ]);
      expect(m.determinant()).toBeCloseTo(0, 10);
    });
  });

  describe("utility", () => {
    test("clone creates independent copy", () => {
      const a = Matrix.fromArray([[1, 2]]);
      const b = a.clone();
      b.set(0, 0, 99);
      expect(a.get(0, 0)).toBe(1);
      expect(b.get(0, 0)).toBe(99);
    });

    test("submatrix extracts block", () => {
      const m = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const sub = m.submatrix(0, 2, 1, 3);
      expect(sub.rows).toBe(2);
      expect(sub.cols).toBe(2);
      expect(sub.toArray()).toEqual([
        [2, 3],
        [5, 6],
      ]);
    });

    test("toArray round-trips fromArray", () => {
      const original = [
        [1, 2],
        [3, 4],
      ];
      expect(Matrix.fromArray(original).toArray()).toEqual(original);
    });

    test("dot product of column vectors", () => {
      const a = Matrix.columnVector([1, 2, 3]);
      const b = Matrix.columnVector([4, 5, 6]);
      expect(a.dot(b)).toBe(32); // 4+10+18
    });
  });

  describe("larger matrices", () => {
    test("100x100 multiply identity = self", () => {
      const n = 100;
      const data = new Float64Array(n * n);
      for (let i = 0; i < n * n; i++) data[i] = Math.random();
      const A = new Matrix(n, n, data);
      const I = Matrix.identity(n);
      const result = A.multiply(I);
      for (let i = 0; i < n * n; i++) {
        expect(result.data[i]).toBeCloseTo(A.data[i]!, 10);
      }
    });

    test("(A^T)^T = A", () => {
      const A = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const ATT = A.transpose().transpose();
      expect(ATT.rows).toBe(A.rows);
      expect(ATT.cols).toBe(A.cols);
      for (let i = 0; i < A.data.length; i++) {
        expect(ATT.data[i]).toBe(A.data[i]);
      }
    });
  });
});

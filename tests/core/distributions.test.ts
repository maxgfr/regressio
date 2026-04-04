import { describe, expect, test } from "bun:test";
import {
  chi2CDFExact,
  chi2InverseCDF,
  chi2TestPValue,
  erf,
  fCDF,
  fTestPValue,
  logGamma,
  normalCDF,
  normalInverseCDF,
  normalPDF,
  regularizedIncompleteBeta,
  tCDF,
  tInverseCDF,
  tPDF,
  tTestPValue,
} from "../../src/core/distributions";

const TOL = 1e-6;

function expectClose(actual: number, expected: number, tol = TOL) {
  expect(Math.abs(actual - expected)).toBeLessThan(tol);
}

describe("logGamma", () => {
  test("logGamma(1) = 0", () => {
    expectClose(logGamma(1), 0, 1e-12);
  });

  test("logGamma(0.5) = log(sqrt(pi))", () => {
    expectClose(logGamma(0.5), Math.log(Math.sqrt(Math.PI)), 1e-10);
  });

  test("logGamma(5) = log(24)", () => {
    expectClose(logGamma(5), Math.log(24), 1e-10);
  });

  test("logGamma(10) = log(362880)", () => {
    expectClose(logGamma(10), Math.log(362880), 1e-8);
  });
});

describe("regularizedIncompleteBeta", () => {
  test("I_0(a,b) = 0", () => {
    expect(regularizedIncompleteBeta(0, 2, 3)).toBe(0);
  });

  test("I_1(a,b) = 1", () => {
    expect(regularizedIncompleteBeta(1, 2, 3)).toBe(1);
  });

  test("known value: I_0.5(1,1) = 0.5", () => {
    expectClose(regularizedIncompleteBeta(0.5, 1, 1), 0.5, 1e-10);
  });

  test("symmetry: I_x(a,b) + I_{1-x}(b,a) = 1", () => {
    const x = 0.3;
    const a = 2;
    const b = 5;
    const sum = regularizedIncompleteBeta(x, a, b) + regularizedIncompleteBeta(1 - x, b, a);
    expectClose(sum, 1, 1e-10);
  });
});

describe("erf", () => {
  test("erf(0) = 0", () => {
    expect(erf(0)).toBe(0);
  });

  test("erf is odd: erf(-x) = -erf(x)", () => {
    expectClose(erf(-1), -erf(1), 1e-10);
  });

  test("erf(1) ≈ 0.8427", () => {
    expectClose(erf(1), 0.8427007929, 1e-4);
  });
});

describe("Normal distribution", () => {
  test("PDF at mean equals 1/sqrt(2π)", () => {
    expectClose(normalPDF(0), 1 / Math.sqrt(2 * Math.PI), 1e-10);
  });

  test("CDF(0) = 0.5 for standard normal", () => {
    expectClose(normalCDF(0), 0.5, 1e-8);
  });

  test("CDF(1.96) ≈ 0.975", () => {
    expectClose(normalCDF(1.96), 0.975, 1e-3);
  });

  test("CDF(-1.96) ≈ 0.025", () => {
    expectClose(normalCDF(-1.96), 0.025, 1e-3);
  });

  test("inverseCDF(0.5) = 0", () => {
    expectClose(normalInverseCDF(0.5), 0, 1e-6);
  });

  test("inverseCDF(0.975) ≈ 1.96", () => {
    expectClose(normalInverseCDF(0.975), 1.96, 0.01);
  });

  test("round-trip: inverseCDF(CDF(x)) ≈ x", () => {
    for (const x of [-2, -1, 0, 0.5, 1, 2]) {
      expectClose(normalInverseCDF(normalCDF(x)), x, 0.01);
    }
  });
});

describe("Student's t-distribution", () => {
  test("tCDF(0, df) = 0.5 (symmetric)", () => {
    expectClose(tCDF(0, 10), 0.5, 1e-10);
    expectClose(tCDF(0, 1), 0.5, 1e-10);
  });

  test("symmetry: tCDF(-t) = 1 - tCDF(t)", () => {
    const t = 2.5;
    const df = 10;
    expectClose(tCDF(-t, df), 1 - tCDF(t, df), 1e-10);
  });

  test("t=2.776, df=4 → CDF ≈ 0.975", () => {
    expectClose(tCDF(2.776, 4), 0.975, 1e-3);
  });

  test("tInverseCDF(0.5, df) = 0", () => {
    expectClose(tInverseCDF(0.5, 10), 0, 1e-6);
  });

  test("tInverseCDF(0.975, 10) ≈ 2.228", () => {
    expectClose(tInverseCDF(0.975, 10), 2.228, 0.01);
  });

  test("round-trip: tInverseCDF(tCDF(x, df), df) ≈ x", () => {
    expectClose(tInverseCDF(tCDF(1.5, 5), 5), 1.5, 1e-4);
  });

  test("tTestPValue: t=0 gives p=1", () => {
    expectClose(tTestPValue(0, 10), 1, 1e-10);
  });

  test("tTestPValue: t=1.96, large df gives p ≈ 0.05", () => {
    expectClose(tTestPValue(1.96, 10000), 0.05, 0.01);
  });

  test("tPDF at t=0 is maximum", () => {
    const peak = tPDF(0, 10);
    expect(tPDF(1, 10)).toBeLessThan(peak);
    expect(tPDF(-1, 10)).toBeLessThan(peak);
  });
});

describe("Fisher F-distribution", () => {
  test("fCDF(0, df1, df2) = 0", () => {
    expect(fCDF(0, 2, 10)).toBe(0);
  });

  test("F(3.89, 2, 10) ≈ 0.95", () => {
    // F(0.95, 2, 10) critical value ≈ 4.103
    expectClose(fCDF(4.103, 2, 10), 0.95, 0.01);
  });

  test("fTestPValue gives 1 for F=0", () => {
    expectClose(fTestPValue(0, 5, 20), 1, 1e-10);
  });

  test("fTestPValue: large F gives p ≈ 0", () => {
    expect(fTestPValue(100, 5, 20)).toBeLessThan(0.001);
  });
});

describe("Chi-squared distribution", () => {
  test("chi2CDF(0, df) = 0", () => {
    expect(chi2CDFExact(0, 5)).toBe(0);
  });

  test("chi2CDF known value: chi2(3.841, df=1) ≈ 0.95", () => {
    expectClose(chi2CDFExact(3.841, 1), 0.95, 0.01);
  });

  test("chi2CDF known value: chi2(5.991, df=2) ≈ 0.95", () => {
    expectClose(chi2CDFExact(5.991, 2), 0.95, 0.01);
  });

  test("chi2InverseCDF(0.95, 1) ≈ 3.841", () => {
    expectClose(chi2InverseCDF(0.95, 1), 3.841, 0.01);
  });

  test("chi2TestPValue: large stat gives p ≈ 0", () => {
    expect(chi2TestPValue(100, 5)).toBeLessThan(0.001);
  });

  test("round-trip: chi2InverseCDF(chi2CDFExact(x, df), df) ≈ x", () => {
    expectClose(chi2InverseCDF(chi2CDFExact(7, 5), 5), 7, 0.01);
  });
});

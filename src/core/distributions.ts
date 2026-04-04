// ---------------------------------------------------------------------------
// Mathematical primitives
// ---------------------------------------------------------------------------

const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** Log-gamma function via Lanczos approximation (~15 digits precision). */
export function logGamma(x: number): number {
  if (x <= 0) throw new Error("logGamma: x must be positive");
  if (x < 0.5) {
    // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = LANCZOS_COEFFS[0]!;
  for (let i = 1; i < LANCZOS_G + 2; i++) {
    a += LANCZOS_COEFFS[i]! / (x + i);
  }
  const t = x + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Log of the beta function: log(B(a,b)) = logΓ(a) + logΓ(b) − logΓ(a+b). */
export function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/**
 * Regularized incomplete beta function I_x(a,b) via continued fraction (Lentz's method).
 * Uses the symmetry relation I_x(a,b) = 1 − I_{1−x}(b,a) when x > (a+1)/(a+b+2).
 */
export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  // Use symmetry for better convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedIncompleteBeta(1 - x, b, a);
  }

  const lnPrefix = a * Math.log(x) + b * Math.log(1 - x) - logBeta(a, b) - Math.log(a);

  // Lentz's continued fraction
  const maxIter = 200;
  const eps = 1e-14;
  const tiny = 1e-30;

  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIter; m++) {
    // Even step: d_{2m}
    const m2 = 2 * m;
    let num = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;

    // Odd step: d_{2m+1}
    num = -((a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + num * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + num / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;

    if (Math.abs(delta - 1) < eps) break;
  }

  return Math.exp(lnPrefix) * h;
}

/** Error function erf(x) via tanh-based approximation (high precision). */
export function erf(x: number): number {
  if (x === 0) return 0;
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  // Use series expansion for small x, asymptotic for large x
  if (ax < 0.5) {
    // Taylor series: erf(x) = 2/sqrt(pi) * (x - x^3/3 + x^5/10 - x^7/42 + ...)
    const x2 = ax * ax;
    let term = ax;
    let sum = ax;
    for (let n = 1; n < 30; n++) {
      term *= -x2 / n;
      sum += term / (2 * n + 1);
    }
    return sign * (2 / Math.sqrt(Math.PI)) * sum;
  }
  // Abramowitz & Stegun 7.1.26 for larger x
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}

/** Inverse error function via rational approximation. */
export function inverseErf(x: number): number {
  if (x <= -1) return -Infinity;
  if (x >= 1) return Infinity;

  const a = 0.147;
  const ln1mx2 = Math.log(1 - x * x);
  const term = 2 / (Math.PI * a) + ln1mx2 / 2;
  const sign = x >= 0 ? 1 : -1;
  return sign * Math.sqrt(Math.sqrt(term * term - ln1mx2 / a) - term);
}

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

export function normalPDF(x: number, mu = 0, sigma = 1): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export function normalCDF(x: number, mu = 0, sigma = 1): number {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}

export function normalInverseCDF(p: number, mu = 0, sigma = 1): number {
  return mu + sigma * Math.SQRT2 * inverseErf(2 * p - 1);
}

// ---------------------------------------------------------------------------
// Student's t-distribution
// ---------------------------------------------------------------------------

export function tPDF(t: number, df: number): number {
  const coeff = Math.exp(logGamma((df + 1) / 2) - logGamma(df / 2));
  return (coeff / Math.sqrt(df * Math.PI)) * (1 + (t * t) / df) ** (-(df + 1) / 2);
}

export function tCDF(t: number, df: number): number {
  if (df <= 0) throw new Error("tCDF: df must be positive");
  const x = df / (t * t + df);
  const ib = regularizedIncompleteBeta(x, df / 2, 0.5);
  return t >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
}

/** Inverse t-CDF via bisection. */
export function tInverseCDF(p: number, df: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (Math.abs(p - 0.5) < 1e-15) return 0;

  // Bisection search
  let lo = -1000;
  let hi = 1000;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (tCDF(mid, df) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

/** Two-tailed p-value from t-statistic. */
export function tTestPValue(tStat: number, df: number): number {
  return 2 * (1 - tCDF(Math.abs(tStat), df));
}

// ---------------------------------------------------------------------------
// Fisher F-distribution
// ---------------------------------------------------------------------------

export function fPDF(x: number, df1: number, df2: number): number {
  if (x <= 0) return 0;
  const half1 = df1 / 2;
  const half2 = df2 / 2;
  const lnCoeff = half1 * Math.log(df1) + half2 * Math.log(df2) - logBeta(half1, half2);
  return Math.exp(lnCoeff + (half1 - 1) * Math.log(x) - (half1 + half2) * Math.log(df1 * x + df2));
}

export function fCDF(x: number, df1: number, df2: number): number {
  if (x <= 0) return 0;
  const z = (df1 * x) / (df1 * x + df2);
  return regularizedIncompleteBeta(z, df1 / 2, df2 / 2);
}

/** Inverse F-CDF via bisection. */
export function fInverseCDF(p: number, df1: number, df2: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;

  let lo = 0;
  let hi = 1000;
  // Expand hi if needed
  while (fCDF(hi, df1, df2) < p) hi *= 2;

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (fCDF(mid, df1, df2) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

/** Right-tail p-value from F-statistic. */
export function fTestPValue(fStat: number, df1: number, df2: number): number {
  return 1 - fCDF(fStat, df1, df2);
}

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

export function chi2PDF(x: number, df: number): number {
  if (x <= 0) return 0;
  const k2 = df / 2;
  return Math.exp((k2 - 1) * Math.log(x) - x / 2 - k2 * Math.LN2 - logGamma(k2));
}

export function chi2CDF(x: number, df: number): number {
  return chi2CDFExact(x, df);
}

/** Chi² CDF = P(df/2, x/2) where P is the regularized lower incomplete gamma function. */
export function chi2CDFExact(x: number, df: number): number {
  if (x <= 0) return 0;
  return regularizedLowerGamma(df / 2, x / 2);
}

/**
 * Regularized lower incomplete gamma function P(a,x) = γ(a,x)/Γ(a).
 * Uses series expansion for x < a+1, continued fraction otherwise.
 */
function regularizedLowerGamma(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x < a + 1) {
    return lowerGammaSeries(a, x);
  }
  return 1 - upperGammaCF(a, x);
}

/** Series expansion for P(a,x): P = e^{-x} x^a / Γ(a) * Σ x^n / (a+1)(a+2)...(a+n) */
function lowerGammaSeries(a: number, x: number): number {
  let term = 1 / a;
  let sum = term;
  for (let n = 1; n < 300; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < 1e-15 * Math.abs(sum)) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

/** Continued fraction for Q(a,x) = 1 - P(a,x) using Lentz's method. */
function upperGammaCF(a: number, x: number): number {
  const tiny = 1e-30;
  // CF: Q(a,x) = e^{-x} x^a / Γ(a) * 1/(x+1-a- 1*(1-a)/(x+3-a- 2*(2-a)/(x+5-a- ...)))
  // Using modified Lentz:
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;

  for (let n = 1; n < 300; n++) {
    const an = -n * (n - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) break;
  }

  return h * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

/** Inverse Chi²-CDF via bisection. */
export function chi2InverseCDF(p: number, df: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;

  let lo = 0;
  let hi = Math.max(df * 4, 100);
  while (chi2CDFExact(hi, df) < p) hi *= 2;

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (chi2CDFExact(mid, df) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

/** Right-tail p-value from Chi² statistic. */
export function chi2TestPValue(stat: number, df: number): number {
  return 1 - chi2CDFExact(stat, df);
}

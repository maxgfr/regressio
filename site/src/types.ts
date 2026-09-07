export type ApiStatus = "pass" | "fail";

export interface ApiCheck {
  name: string;
  family: string;
  status: ApiStatus;
  value: string;
  duration: number;
  error?: string;
}

export interface PlotPoint {
  x: number;
  y: number;
  predicted: number;
  robust: number;
  polynomial: number;
}

export interface SuiteResult {
  checks: ApiCheck[];
  startedAt: string;
  duration: number;
  wasm: boolean;
  plot: PlotPoint[];
  coefficients: {
    ols: [number, number];
    robust: [number, number];
    polynomial: number[];
  };
  diagnostics: Record<string, number>;
  examples: Record<string, string>;
}

export type WorkerResponse =
  | { type: "progress"; completed: number; total: number; name: string }
  | { type: "complete"; result: SuiteResult }
  | { type: "fatal"; error: string };

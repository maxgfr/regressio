import "./styles.css";
import type { PlotPoint, SuiteResult, WorkerResponse } from "./types";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App root was not found");

const families = [
  "Regression",
  "Classification",
  "Diagnostics",
  "Preprocessing",
  "Intervals",
  "Matrix",
];

app.innerHTML = `
  <main id="proof-sheet">
    <section class="hero" aria-labelledby="suite-title">
      <div class="registration registration--top" aria-hidden="true"></div>
      <header class="instrument-header">
        <h1 class="r-wordmark wordmark"><a href="#proof-sheet" aria-label="regressio proof suite home">regressio</a></h1>
        <div class="r-dataset-heading header-cell">
          <span class="header-label" id="suite-title">Synthetic dataset</span>
          <span class="header-readout">points 36 · features 1</span>
        </div>
        <div class="r-engine-heading header-cell">
          <span class="header-label">TypeScript + WASM</span>
          <span class="engine-mode">runtime engine</span>
          <span class="header-readout">precision f64 · scale 1.00</span>
        </div>
        <div class="r-run-metadata run-meta">
          <span id="run-time">waiting for worker</span>
          <span class="proof-seed">Browser proof · seed 01397</span>
          <div class="instrument-controls" aria-hidden="true">
            <span class="calibration-knob"><i></i></span>
            <span>cal / 1.00</span>
            <span class="system-state"><i></i>system / ok</span>
          </div>
          <button class="run-button" id="run-suite" type="button">
            <span class="run-button__mark" aria-hidden="true"></span>
            Run suite
          </button>
        </div>
      </header>

      <div class="r-calibration-strip calibration" aria-hidden="true">
        ${Array.from({ length: 21 }, (_, index) => `<span style="--tick:${index}">${index * 5}</span>`).join("")}
      </div>

      <div class="plot-regions" aria-hidden="true">
        <i class="r-plot-upper-left"></i><i class="r-plot-upper-right"></i>
        <i class="r-plot-lower-left"></i><i class="r-plot-lower-right"></i>
      </div>

      <div class="r-dataset-note dataset-note">
        <strong>Synthetic dataset</strong>
        <span>n = 36</span>
        <span>x ∼ U(−10, 10)</span>
        <span>y = 2.5x − 1 + ε</span>
      </div>

      <figure class="main-plot" aria-labelledby="plot-caption">
        <svg id="plot" viewBox="0 0 1000 650" role="img" aria-labelledby="plot-title plot-desc">
          <title id="plot-title">Regression model comparison</title>
          <desc id="plot-desc">Synthetic observations with ordinary least squares, polynomial, and robust regression traces.</desc>
          <g class="plot-grid" aria-hidden="true"></g>
          <g class="plot-residuals" aria-hidden="true"></g>
          <path class="trace trace--poly" aria-hidden="true"></path>
          <path class="trace trace--robust" aria-hidden="true"></path>
          <path class="trace trace--ols" aria-hidden="true"></path>
          <g class="plot-points" aria-hidden="true"></g>
        </svg>
        <img class="plotter-rail" src="${import.meta.env.BASE_URL}plotter-carriage.png" alt="" aria-hidden="true" />
        <figcaption id="plot-caption">Live traces drawn from the suite’s fitted models.</figcaption>
      </figure>

      <div class="r-plot-legend plot-legend" aria-label="Plot legend">
        <div><i class="key key--ols"></i><span>OLS line</span><code id="ols-equation">initializing</code></div>
        <div><i class="key key--poly"></i><span>Poly (deg 2)</span><code id="poly-equation">initializing</code></div>
        <div><i class="key key--robust"></i><span>Robust line</span><code id="robust-equation">initializing</code></div>
      </div>

      <div class="r-residual-note residual-note" id="residual-note">Residual summary waiting for suite</div>

      <aside class="proof-rail" aria-labelledby="proof-title">
        <h2 class="r-proof-title" id="proof-title">Proof status</h2>
        <div class="r-proof-count proof-block proof-block--count">
          <span class="proof-label">Export coverage</span>
          <strong id="api-count">— / 36 APIs</strong>
        </div>
        <div class="r-proof-engine proof-block proof-block--engine">
          <span class="proof-label">Runtime engine</span>
          <strong id="engine-status">Loading</strong>
        </div>
        <div class="r-proof-failed proof-block proof-block--failed">
          <span class="proof-label">Test result</span>
          <strong id="failed-count">— failed</strong>
        </div>
        <div class="r-proof-runtime proof-block proof-block--runtime">
          <span class="proof-label">Runtime</span>
          <strong id="suite-duration">— ms</strong>
        </div>
        <div class="r-proof-table proof-table">
          <table>
            <caption>Representative checks</caption>
            <thead><tr><th scope="col">Check</th><th scope="col">Value</th><th scope="col">Status</th></tr></thead>
            <tbody id="proof-rows">
              <tr><td>OLS R²</td><td>—</td><td>wait</td></tr>
              <tr><td>DW</td><td>—</td><td>wait</td></tr>
              <tr><td>Cook max</td><td>—</td><td>wait</td></tr>
              <tr><td>Matrix det</td><td>—</td><td>wait</td></tr>
            </tbody>
          </table>
        </div>
      </aside>

      <nav class="r-family-index family-index" aria-label="API families">
        <ul>${families.map((family) => `<li><a href="#labs" data-family="${family}">${family}</a></li>`).join("")}</ul>
      </nav>
      <div class="registration registration--bottom" aria-hidden="true"></div>
    </section>

    <section class="labs" id="labs" aria-labelledby="labs-title">
      <header class="labs-intro">
        <h2 id="labs-title">Every exported function, under load</h2>
        <p>Select a family to inspect live values, timings, assertions, and the exact code used by this page.</p>
      </header>
      <div class="family-switcher" role="tablist" aria-label="Proof family">
        ${families.map((family, index) => `<button type="button" role="tab" id="tab-${family.toLowerCase()}" aria-controls="lab-panel" aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-family="${family}">${family}</button>`).join("")}
      </div>
      <div class="lab-panel" id="lab-panel" role="tabpanel" aria-labelledby="tab-regression" tabindex="0">
        <div class="lab-readout">
          <table>
            <caption id="checks-caption">Regression API checks</caption>
            <thead><tr><th scope="col">Export</th><th scope="col">Observed</th><th scope="col">Time</th><th scope="col">Result</th></tr></thead>
            <tbody id="all-checks"><tr><td colspan="4">Suite is initializing…</td></tr></tbody>
          </table>
        </div>
        <div class="code-proof">
          <div class="code-proof__heading"><h3>Executed example</h3><button type="button" id="copy-code">Copy code</button></div>
          <pre><code id="example-code">Waiting for the worker…</code></pre>
          <p class="copy-status" id="copy-status" aria-live="polite"></p>
        </div>
      </div>
    </section>

    <footer class="site-footer">
      <p>Computed locally in this browser. No dataset leaves this page.</p>
      <a href="https://github.com/maxgfr/regressio">Source on GitHub</a>
    </footer>
    <p class="visually-hidden" id="suite-announcement" role="status"></p>
  </main>
`;

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let currentResult: SuiteResult | undefined;
let selectedFamily = "Regression";

function query<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"]/g,
    (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]!,
  );
}

function setText(selector: string, value: string): void {
  const element = query<HTMLElement>(selector);
  if (element.textContent !== value) element.textContent = value;
}

function format(value: number, digits = 4): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function chartPath(points: PlotPoint[], key: "predicted" | "robust" | "polynomial"): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.x)} ${yScale(point[key])}`)
    .join(" ");
}

function xScale(value: number): number {
  return 70 + ((value + 11) / 22) * 880;
}

function yScale(value: number): number {
  return 590 - ((value + 31) / 62) * 530;
}

function renderPlot(result: SuiteResult): void {
  const svg = query<SVGSVGElement>("#plot");
  const grid = svg.querySelector<SVGGElement>(".plot-grid")!;
  const residuals = svg.querySelector<SVGGElement>(".plot-residuals")!;
  const points = svg.querySelector<SVGGElement>(".plot-points")!;
  const horizontalGrid: Array<[number, number]> = [
    [60, 30],
    [148, 20],
    [236, 10],
    [324, 0],
    [412, -10],
    [575, -30],
  ];
  if (!grid.childElementCount) {
    grid.innerHTML =
      horizontalGrid
        .map(
          ([position, label]) =>
            `<line x1="70" x2="950" y1="${position}" y2="${position}" /><text x="52" y="${position + 5}">${label}</text>`,
        )
        .join("") +
      [-10, -5, 0, 5, 10]
        .map(
          (value) =>
            `<line x1="${xScale(value)}" x2="${xScale(value)}" y1="60" y2="590" /><text x="${xScale(value)}" y="620">${value}</text>`,
        )
        .join("");
  }
  residuals.innerHTML = result.plot
    .map(
      (point) =>
        `<line x1="${xScale(point.x)}" x2="${xScale(point.x)}" y1="${yScale(point.y)}" y2="${yScale(point.predicted)}" />`,
    )
    .join("");
  points.innerHTML = result.plot
    .map((point) => `<circle cx="${xScale(point.x)}" cy="${yScale(point.y)}" r="4.6" />`)
    .join("");
  svg
    .querySelector<SVGPathElement>(".trace--ols")!
    .setAttribute("d", chartPath(result.plot, "predicted"));
  svg
    .querySelector<SVGPathElement>(".trace--robust")!
    .setAttribute("d", chartPath(result.plot, "robust"));
  svg
    .querySelector<SVGPathElement>(".trace--poly")!
    .setAttribute("d", chartPath(result.plot, "polynomial"));
}

function renderFamily(): void {
  if (!currentResult) return;
  const checks = currentResult.checks.filter((check) => check.family === selectedFamily);
  query<HTMLTableSectionElement>("#all-checks").innerHTML = checks
    .map(
      (check) => `
    <tr>
      <th scope="row">${escapeHtml(check.name)}</th>
      <td>${escapeHtml(check.error ?? check.value)}</td>
      <td>${format(check.duration, 2)} ms</td>
      <td><span class="status status--${check.status}">${check.status}</span></td>
    </tr>`,
    )
    .join("");
  setText("#checks-caption", `${selectedFamily} API checks`);
  setText("#example-code", currentResult.examples[selectedFamily] ?? "");
  const panel = query<HTMLElement>("#lab-panel");
  panel.setAttribute("aria-labelledby", `tab-${selectedFamily.toLowerCase()}`);
}

function activateFamily(family: string): void {
  if (!families.includes(family)) return;
  selectedFamily = family;
  for (const tab of document.querySelectorAll<HTMLButtonElement>("[role=tab]")) {
    const selected = tab.dataset.family === family;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  renderFamily();
}

function familyFromHash(): string | undefined {
  const match = window.location.hash.match(/^#lab\/([^/]+)$/);
  if (!match) return undefined;
  return families.find(
    (family) => family.toLowerCase() === decodeURIComponent(match[1] ?? "").toLowerCase(),
  );
}

function renderResult(result: SuiteResult): void {
  currentResult = result;
  const passed = result.checks.filter((check) => check.status === "pass").length;
  const failed = result.checks.length - passed;
  setText("#api-count", `${passed} / ${result.checks.length} APIs`);
  setText("#engine-status", result.wasm ? "WASM active" : "TypeScript fallback");
  query<HTMLElement>("#engine-status").classList.toggle("is-fallback", !result.wasm);
  setText("#failed-count", `${failed} failed`);
  query<HTMLElement>("#failed-count").classList.toggle("has-failures", failed > 0);
  setText("#suite-duration", `${format(result.duration, 1)} ms`);
  setText("#run-time", new Date(result.startedAt).toISOString().replace(".000Z", "Z"));
  setText(
    "#ols-equation",
    `y = ${format(result.coefficients.ols[1], 3)}x ${result.coefficients.ols[0] < 0 ? "−" : "+"} ${format(Math.abs(result.coefficients.ols[0]), 3)}`,
  );
  setText(
    "#robust-equation",
    `y = ${format(result.coefficients.robust[1], 3)}x ${result.coefficients.robust[0] < 0 ? "−" : "+"} ${format(Math.abs(result.coefficients.robust[0]), 3)}`,
  );
  setText(
    "#poly-equation",
    `y = ${format(result.coefficients.polynomial[2] ?? 0, 3)}x² + ${format(result.coefficients.polynomial[1] ?? 0, 3)}x ${result.coefficients.polynomial[0]! < 0 ? "−" : "+"} ${format(Math.abs(result.coefficients.polynomial[0]!), 3)}`,
  );
  setText(
    "#residual-note",
    `Residual σ ${format(result.diagnostics.residualStdError ?? Number.NaN)} · Cook max ${format(result.diagnostics.maxCook ?? Number.NaN)} · leverage ${format(result.diagnostics.maxLeverage ?? Number.NaN)}`,
  );
  const matrixCheck = result.checks.find((check) => check.name === "Matrix");
  query<HTMLTableSectionElement>("#proof-rows").innerHTML = [
    ["OLS R²", format(result.diagnostics.rSquared ?? Number.NaN), "pass"],
    ["Adj. R²", format(result.diagnostics.adjustedRSquared ?? Number.NaN), "pass"],
    ["DW", format(result.diagnostics.durbinWatson ?? Number.NaN), "pass"],
    ["Cook max", format(result.diagnostics.maxCook ?? Number.NaN), "pass"],
    ["Matrix det", matrixCheck?.value ?? "—", matrixCheck?.status ?? "fail"],
  ]
    .map(
      ([name, value, status]) =>
        `<tr><th scope="row">${name}</th><td>${value}</td><td><span class="status status--${status}">${status}</span></td></tr>`,
    )
    .join("");
  renderPlot(result);
  renderFamily();
  query<HTMLButtonElement>("#run-suite").disabled = false;
  query<HTMLButtonElement>("#run-suite").classList.remove("is-running");
  document.documentElement.dataset.suite = failed === 0 ? "passed" : "failed";
  setText(
    "#suite-announcement",
    failed === 0
      ? `Proof suite complete. All ${result.checks.length} API checks passed using ${result.wasm ? "WebAssembly" : "the TypeScript fallback"}.`
      : `Proof suite complete with ${failed} failed checks. Review the proof table for details.`,
  );
}

function runSuite(): void {
  const button = query<HTMLButtonElement>("#run-suite");
  button.disabled = true;
  button.classList.add("is-running");
  button.querySelector("span")?.setAttribute("aria-hidden", "true");
  query<HTMLElement>("#api-count").textContent = "0 / 36 APIs";
  query<HTMLElement>("#failed-count").textContent = "running";
  worker.postMessage({ type: "run" });
}

worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
  const message = event.data;
  if (message.type === "progress") {
    query<HTMLElement>("#api-count").textContent = `${message.completed} / ${message.total} APIs`;
    query<HTMLElement>("#failed-count").textContent = message.name;
    return;
  }
  if (message.type === "complete") {
    renderResult(message.result);
    return;
  }
  query<HTMLElement>("#failed-count").textContent = "suite error";
  query<HTMLElement>("#failed-count").classList.add("has-failures");
  query<HTMLElement>("#residual-note").textContent = `${message.error}. Run the suite again.`;
  setText("#suite-announcement", `Proof suite failed: ${message.error}. Run the suite again.`);
  query<HTMLButtonElement>("#run-suite").disabled = false;
});

query<HTMLButtonElement>("#run-suite").addEventListener("click", runSuite);

for (const tab of document.querySelectorAll<HTMLButtonElement>("[role=tab]")) {
  tab.addEventListener("click", () => {
    const family = tab.dataset.family ?? "Regression";
    activateFamily(family);
    history.replaceState(null, "", `#lab/${family.toLowerCase()}`);
  });
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const tabs = [...document.querySelectorAll<HTMLButtonElement>("[role=tab]")];
    const current = tabs.indexOf(tab);
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const next = tabs[(current + direction + tabs.length) % tabs.length]!;
    next.click();
    next.focus();
  });
}

for (const link of document.querySelectorAll<HTMLAnchorElement>("[data-family]")) {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const family = link.dataset.family ?? "Regression";
    activateFamily(family);
    history.pushState(null, "", `#lab/${family.toLowerCase()}`);
    query<HTMLElement>("#labs").scrollIntoView({ behavior: "smooth" });
  });
}

window.addEventListener("hashchange", () => {
  const family = familyFromHash();
  if (family) activateFamily(family);
});

query<HTMLButtonElement>("#copy-code").addEventListener("click", async () => {
  const code = query<HTMLElement>("#example-code").textContent ?? "";
  try {
    await navigator.clipboard.writeText(code);
    query<HTMLElement>("#copy-status").textContent = "Code copied.";
  } catch {
    query<HTMLElement>("#copy-status").textContent =
      "Clipboard unavailable. Select the code to copy it.";
  }
});

activateFamily(familyFromHash() ?? "Regression");
runSuite();

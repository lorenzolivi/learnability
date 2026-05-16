import { useState, useMemo, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ── Envelope models ──────────────────────────────────────────────
const envelopeModels = {
  exponential: {
    label: "Exponential  f(ℓ) = A·e^{−ℓ/τ}",
    fn: (ell, A, tau) => A * Math.exp(-ell / tau),
    params: [
      { key: "A", label: "A (amplitude)", min: 0.1, max: 10, step: 0.1, default: 2.0 },
      { key: "tau", label: "τ (time constant)", min: 5, max: 500, step: 5, default: 80 },
    ],
  },
  powerlaw: {
    label: "Power-law  f(ℓ) = A·ℓ^{−β}",
    fn: (ell, A, beta) => A * Math.pow(Math.max(ell, 1), -beta),
    params: [
      { key: "A", label: "A (amplitude)", min: 0.1, max: 50, step: 0.5, default: 10.0 },
      { key: "beta", label: "β (decay exponent)", min: 0.1, max: 3.0, step: 0.05, default: 0.8 },
    ],
  },
  stretched_exp: {
    label: "Stretched exp  f(ℓ) = A·e^{−(ℓ/τ)^γ}",
    fn: (ell, A, tau, gamma) => A * Math.exp(-Math.pow(ell / tau, gamma)),
    params: [
      { key: "A", label: "A (amplitude)", min: 0.1, max: 10, step: 0.1, default: 2.0 },
      { key: "tau", label: "τ (time constant)", min: 5, max: 500, step: 5, default: 80 },
      { key: "gamma", label: "γ (stretch)", min: 0.1, max: 2.0, step: 0.05, default: 0.5 },
    ],
  },
};

// ── Master proportionality ───────────────────────────────────────
// N(ℓ) ∝ f(ℓ)^{-κ_α}   where κ_α = α / (α − 1)
const kappa = (alpha) => alpha / (alpha - 1);
const sampleComplexity = (fEll, alpha, sigma, C) => {
  const k = kappa(alpha);
  const fSafe = Math.max(fEll, 1e-30);
  return C * Math.pow(sigma, alpha) * Math.pow(fSafe, -k);
};

// ── Helpers ──────────────────────────────────────────────────────
const fmtSci = (v) => {
  if (v === 0) return "0";
  if (!isFinite(v)) return "∞";
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / Math.pow(10, e);
  if (e === 0) return m.toFixed(2);
  return `${m.toFixed(1)}×10^${e}`;
};

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-300 rounded p-2 text-xs shadow">
      <p className="font-semibold">ℓ = {d.ell}</p>
      <p>f(ℓ) = {fmtSci(d.f)}</p>
      <p>N(ℓ) = {fmtSci(d.N)}</p>
      <p className="italic text-gray-500">{d.regime}</p>
    </div>
  );
};

// ── Main component ───────────────────────────────────────────────
export default function SampleComplexityPlot() {
  // Envelope selection
  const [envType, setEnvType] = useState("exponential");
  const envModel = envelopeModels[envType];

  // Per-envelope-model parameter state
  const [envParams, setEnvParams] = useState(() => {
    const init = {};
    for (const [k, m] of Object.entries(envelopeModels)) {
      init[k] = {};
      for (const p of m.params) init[k][p.key] = p.default;
    }
    return init;
  });

  // Global parameters
  const [alpha, setAlpha] = useState(1.5);
  const [sigma, setSigma] = useState(1.0);
  const [C, setC] = useState(1.0);
  const [Nbudget, setNbudget] = useState(10000);
  const [logX, setLogX] = useState(false);
  const [logY, setLogY] = useState(true);
  const [ellMax, setEllMax] = useState(256);

  const setParam = useCallback(
    (key, val) =>
      setEnvParams((prev) => ({
        ...prev,
        [envType]: { ...prev[envType], [key]: val },
      })),
    [envType]
  );

  // Compute data
  const { data, regimeBounds } = useMemo(() => {
    const params = envParams[envType];
    const paramVals = envModel.params.map((p) => params[p.key]);
    const pts = [];

    for (let ell = 1; ell <= ellMax; ell++) {
      const f = envModel.fn(ell, ...paramVals);
      const N = sampleComplexity(f, alpha, sigma, C);
      let regime = "Learnable";
      if (N < 10) regime = "Easy";
      else if (N > Nbudget) regime = "Hard";
      pts.push({ ell, f, N, regime });
    }

    // Find regime boundaries (first ℓ where regime changes)
    let easyEnd = 0;
    let hardStart = ellMax + 1;
    for (const p of pts) {
      if (p.regime === "Easy") easyEnd = p.ell;
    }
    for (const p of pts) {
      if (p.regime === "Hard" && p.ell < hardStart) hardStart = p.ell;
    }

    return { data: pts, regimeBounds: { easyEnd, hardStart } };
  }, [envType, envParams, alpha, sigma, C, Nbudget, ellMax, envModel]);

  const kappaVal = kappa(alpha);

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-1">
          Sample Complexity &amp; Learnability Regimes
        </h1>
        <p className="text-sm text-gray-500 mb-1">
          N(ℓ) ∝ σ^α · f(ℓ)^{"{"}−κ_α{"}"} where κ_α = α/(α−1) ={" "}
          {kappaVal.toFixed(2)}
        </p>

        {/* ── Preamble / disclaimer ────────────────────── */}
        <details className="mb-4 bg-amber-50 border border-amber-200 rounded-lg">
          <summary className="px-3 py-2 cursor-pointer text-sm font-medium text-amber-800">
            About this visualization (read before interpreting)
          </summary>
          <div className="px-3 pb-3 text-xs text-amber-900 space-y-2">
            <p>
              This is a <strong>schematic illustration</strong> of the
              learnability regimes described in the paper, not a realistic
              simulation. It is meant to build qualitative intuition for how
              the envelope decay shape, the noise tail index α, and the
              sample budget interact to produce three regimes (easy,
              learnable, hard). It should <em>not</em> be taken as a
              substitute for the full analysis.
            </p>
            <p className="font-semibold">Key simplifications:</p>
            <ul className="list-disc pl-4 space-y-1">
              <li>
                <strong>The envelope f(ℓ) is assumed known and parametric.</strong>{" "}
                In practice, f(ℓ) is an empirical quantity that must itself be
                estimated from data via the recurrent network's gating
                dynamics. Its shape is architecture-dependent (LSTM vs GRU vs
                vanilla RNN) and cannot be prescribed a priori.
              </li>
              <li>
                <strong>The proportionality constant is arbitrary.</strong>{" "}
                The master law N(ℓ) ∝ f(ℓ)<sup>−κ<sub>α</sub></sup> captures
                the correct scaling exponent, but the actual prefactor depends
                on the α-stable scale parameter σ̂, the detection precision ε,
                the noise tolerance, and the specific estimator used
                (Koutrouvelis ECF, McCulloch). Here it is replaced by
                adjustable knobs C and σ that have no calibrated meaning.
              </li>
              <li>
                <strong>The regime boundaries are ad hoc.</strong>{" "}
                "Easy" is defined as N(ℓ) &lt; 10 and "Hard" as
                N(ℓ) &gt; N<sub>budget</sub>. In the paper, the learnability
                window emerges from comparing the matched-statistic
                signal-to-noise ratio against a detection threshold that
                depends on the full α-stable distribution — not from fixed
                cutoffs.
              </li>
              <li>
                <strong>α is treated as a global constant.</strong>{" "}
                In practice, the tail index α(ℓ) can vary with the lag ℓ and
                must be estimated per lag from finite samples, introducing its
                own uncertainty. The paper handles this via bootstrap
                confidence intervals and reliability checks.
              </li>
              <li>
                <strong>No estimation noise.</strong>{" "}
                This plot shows the deterministic mapping f(ℓ) → N(ℓ). The
                real pipeline must contend with finite-sample estimation error
                in f̂(ℓ), α̂(ℓ), and σ̂(ℓ), which is where the bulk of the
                paper's methodology (multi-projection aggregation,
                cross-sequence averaging, robust estimators) is needed.
              </li>
            </ul>
            <p>
              In short: the paper's contribution is precisely the machinery
              needed to estimate these quantities from data for a given trained
              network — something this toy visualization assumes away entirely.
            </p>
          </div>
        </details>

        <div className="flex flex-col lg:flex-row gap-4">
          {/* ── Controls panel ─────────────────────────────── */}
          <div className="lg:w-72 flex-shrink-0 space-y-4">
            {/* Envelope selector */}
            <div className="bg-white rounded-lg shadow p-3">
              <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                Envelope f(ℓ)
              </h2>
              {Object.entries(envelopeModels).map(([k, m]) => (
                <label
                  key={k}
                  className={`block cursor-pointer px-2 py-1 rounded text-sm mb-1 ${
                    envType === k
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-gray-600 hover:bg-gray-50"
                  }`}
                >
                  <input
                    type="radio"
                    name="env"
                    value={k}
                    checked={envType === k}
                    onChange={() => setEnvType(k)}
                    className="mr-2"
                  />
                  {m.label}
                </label>
              ))}
            </div>

            {/* Envelope parameters */}
            <div className="bg-white rounded-lg shadow p-3">
              <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                Envelope parameters
              </h2>
              {envModel.params.map((p) => (
                <div key={p.key} className="mb-2">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>{p.label}</span>
                    <span className="font-mono">
                      {envParams[envType][p.key]}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    value={envParams[envType][p.key]}
                    onChange={(e) => setParam(p.key, parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              ))}
            </div>

            {/* Noise / tail parameters */}
            <div className="bg-white rounded-lg shadow p-3">
              <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                Noise &amp; scaling
              </h2>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>α (tail index)</span>
                  <span className="font-mono">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={1.05}
                  max={2.0}
                  step={0.01}
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-gray-400 mt-0.5">
                  κ_α = {kappaVal.toFixed(2)}
                  {alpha >= 1.95
                    ? " (Gaussian-like)"
                    : alpha <= 1.2
                    ? " (very heavy-tailed)"
                    : ""}
                </div>
              </div>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>σ (noise scale)</span>
                  <span className="font-mono">{sigma.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={5.0}
                  step={0.05}
                  value={sigma}
                  onChange={(e) => setSigma(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>C (proportionality const)</span>
                  <span className="font-mono">{C.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.01}
                  max={10}
                  step={0.01}
                  value={C}
                  onChange={(e) => setC(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            {/* Display options */}
            <div className="bg-white rounded-lg shadow p-3">
              <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                Display
              </h2>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>N_budget (sample budget)</span>
                  <span className="font-mono">{Nbudget.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  min={100}
                  max={100000}
                  step={100}
                  value={Nbudget}
                  onChange={(e) => setNbudget(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              <div className="mb-2">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>ℓ_max</span>
                  <span className="font-mono">{ellMax}</span>
                </div>
                <input
                  type="range"
                  min={32}
                  max={1024}
                  step={16}
                  value={ellMax}
                  onChange={(e) => setEllMax(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
              <div className="flex gap-4 mt-2">
                <label className="flex items-center gap-1 text-xs text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={logX}
                    onChange={(e) => setLogX(e.target.checked)}
                  />
                  Log x-axis
                </label>
                <label className="flex items-center gap-1 text-xs text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={logY}
                    onChange={(e) => setLogY(e.target.checked)}
                  />
                  Log y-axis
                </label>
              </div>
            </div>

            {/* Regime summary */}
            <div className="bg-white rounded-lg shadow p-3">
              <h2 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                Regime summary
              </h2>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded" style={{ background: "#bbf7d0" }} />
                  <span className="text-gray-700">
                    Easy: ℓ ≤ {regimeBounds.easyEnd || "—"}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded" style={{ background: "#bfdbfe" }} />
                  <span className="text-gray-700">
                    Learnable: {regimeBounds.easyEnd > 0 ? regimeBounds.easyEnd + 1 : 1}
                    {" ≤ ℓ ≤ "}
                    {regimeBounds.hardStart <= ellMax
                      ? regimeBounds.hardStart - 1
                      : ellMax}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded" style={{ background: "#fecaca" }} />
                  <span className="text-gray-700">
                    Hard: ℓ ≥{" "}
                    {regimeBounds.hardStart <= ellMax
                      ? regimeBounds.hardStart
                      : "—"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* ── Charts ────────────────────────────────────── */}
          <div className="flex-1 space-y-4">
            {/* N(ℓ) plot */}
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-2">
                Sample complexity N(ℓ)
              </h2>
              <ResponsiveContainer width="100%" height={340}>
                <LineChart
                  data={data}
                  margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />

                  {/* Regime background bands */}
                  {regimeBounds.easyEnd > 0 && (
                    <ReferenceArea
                      x1={1}
                      x2={regimeBounds.easyEnd}
                      fill="#bbf7d0"
                      fillOpacity={0.35}
                    />
                  )}
                  {regimeBounds.hardStart <= ellMax && (
                    <ReferenceArea
                      x1={regimeBounds.hardStart}
                      x2={ellMax}
                      fill="#fecaca"
                      fillOpacity={0.35}
                    />
                  )}

                  <XAxis
                    dataKey="ell"
                    type="number"
                    domain={[1, ellMax]}
                    scale={logX ? "log" : "auto"}
                    tickFormatter={(v) => Math.round(v)}
                    label={{
                      value: "Lag ℓ",
                      position: "insideBottom",
                      offset: -10,
                      style: { fontSize: 12 },
                    }}
                    allowDataOverflow
                  />
                  <YAxis
                    scale={logY ? "log" : "auto"}
                    domain={logY ? [1, "auto"] : [0, "auto"]}
                    tickFormatter={(v) =>
                      logY
                        ? v >= 1000
                          ? `${(v / 1000).toFixed(0)}k`
                          : v
                        : v >= 1000
                        ? `${(v / 1000).toFixed(0)}k`
                        : v
                    }
                    label={{
                      value: "N(ℓ)",
                      angle: -90,
                      position: "insideLeft",
                      offset: 0,
                      style: { fontSize: 12 },
                    }}
                    allowDataOverflow
                  />
                  <Tooltip content={<CustomTooltip />} />

                  {/* N_budget reference line */}
                  <ReferenceLine
                    y={Nbudget}
                    stroke="#ef4444"
                    strokeDasharray="6 3"
                    label={{
                      value: `N_budget = ${Nbudget.toLocaleString()}`,
                      position: "insideTopRight",
                      style: { fontSize: 10, fill: "#ef4444" },
                    }}
                  />

                  <Line
                    type="monotone"
                    dataKey="N"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* f(ℓ) plot */}
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-2">
                Envelope f(ℓ)
              </h2>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart
                  data={data}
                  margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="ell"
                    type="number"
                    domain={[1, ellMax]}
                    scale={logX ? "log" : "auto"}
                    label={{
                      value: "Lag ℓ",
                      position: "insideBottom",
                      offset: -10,
                      style: { fontSize: 12 },
                    }}
                    allowDataOverflow
                  />
                  <YAxis
                    scale={logY ? "log" : "auto"}
                    domain={logY ? ["auto", "auto"] : [0, "auto"]}
                    tickFormatter={(v) => (v < 0.01 ? v.toExponential(0) : v.toFixed(2))}
                    label={{
                      value: "f(ℓ)",
                      angle: -90,
                      position: "insideLeft",
                      offset: 0,
                      style: { fontSize: 12 },
                    }}
                    allowDataOverflow
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone"
                    dataKey="f"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

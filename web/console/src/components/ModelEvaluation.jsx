import { useId, useState } from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { DATASET_STATS } from '../data/sampleDataset.js';

const N_CLASSES = DATASET_STATS.languages;
const TEST_SIZE = DATASET_STATS.splits.find((s) => s.id === 'test').size;
const NFR1_P95_MS = 300; // NFR1: p95 < 300 ms for 1,000 characters on CPU
const HIGH_CONFUSION = 0.1; // pairs at or above this rate get a warning pill

// Majority-class baseline on a balanced test set: it always predicts one language.
//   accuracy = 1/235; the predicted class has precision 1/235 and recall 1, every other class has F1 = 0,
//   so macro-F1 = (2/236)/235 ≈ 0.00004 (not 0.4%).
const BASELINE_ACC = 1 / N_CLASSES;
const BASELINE_MACRO_F1 = 2 / (N_CLASSES + 1) / N_CLASSES;

// Mock benchmark figures (FR8). Replace with the metrics saved next to each model version.
const MODELS = [
  {
    id: 'svm',
    pickerLabel: 'Linear SVM (v2 - Candidate)',
    name: 'Linear SVM',
    version: 'v2',
    algorithm: 'Linear SVM',
    params: 'C = 1.0',
    valAcc: 0.938,
    testAcc: 0.932,
    macroF1: 0.926,
    p95: 28,
    active: false,
  },
  {
    id: 'nb',
    pickerLabel: 'Multinomial NB (v1 - Active Baseline)',
    name: 'Multinomial NB',
    version: 'v1',
    algorithm: 'Multinomial Naive Bayes',
    params: 'alpha = 0.1',
    valAcc: 0.921,
    testAcc: 0.915,
    macroF1: 0.908,
    p95: 12,
    active: true,
  },
  {
    id: 'majority',
    pickerLabel: 'Majority Class (Baseline)',
    name: 'Majority Class',
    version: 'baseline',
    algorithm: 'Majority class',
    params: 'strategy = most_frequent',
    valAcc: BASELINE_ACC,
    testAcc: BASELINE_ACC,
    macroF1: BASELINE_MACRO_F1,
    p95: null, // not measured
    active: false,
  },
];
const TABLE_ORDER = ['nb', 'svm', 'majority']; // FR8: primary, comparison, baseline
const ACTIVE = MODELS.find((m) => m.active);
const ANALYSIS_MODEL = `${ACTIVE.name} ${ACTIVE.version}`;

// Confusion rate = share of a language's test samples predicted as its counterpart (worse direction).
const CONFUSED_PAIRS = [
  { id: 'bos', a: { name: 'Bosnian', code: 'bos' }, b: [{ name: 'Croatian', code: 'hrv' }, { name: 'Serbian', code: 'srp' }], note: 'Latin script overlap', rate: 0.18 },
  { id: 'dan', a: { name: 'Danish', code: 'dan' }, b: [{ name: 'Norwegian Bokmål', code: 'nob' }], rate: 0.12 },
  { id: 'ind', a: { name: 'Indonesian', code: 'ind' }, b: [{ name: 'Malay', code: 'msa' }], rate: 0.09 },
  { id: 'spa', a: { name: 'Spanish', code: 'spa' }, b: [{ name: 'Portuguese', code: 'por' }], note: 'short texts', rate: 0.06 },
];
const MAX_RATE = Math.max(...CONFUSED_PAIRS.map((p) => p.rate));

// Misclassified test snippets (mock). truth/pred = { name, code }.
const L = (name, code) => ({ name, code });
const ERRORS = [
  { id: 'e1', text: 'Dobar dan', lang: 'hr', truth: L('Croatian', 'hrv'), pred: L('Bosnian', 'bos'), conf: 0.52, cause: 'Short text under 40 chars' },
  { id: 'e2', text: 'Un animal normal.', lang: 'es', truth: L('Spanish', 'spa'), pred: L('Portuguese', 'por'), conf: 0.47, cause: 'Short text under 40 chars' },
  { id: 'e3', text: 'Det er en stor dag for hele familien i dag.', lang: 'da', truth: L('Danish', 'dan'), pred: L('Norwegian Bokmål', 'nob'), conf: 0.58, cause: 'Orthographic overlap' },
  { id: 'e4', text: 'Saya pergi ke pasar untuk membeli sayur.', lang: 'ms', truth: L('Malay', 'msa'), pred: L('Indonesian', 'ind'), conf: 0.61, cause: 'Orthographic overlap' },
  { id: 'e5', text: 'Ovo je jedan od najstarijih gradova u zemlji.', lang: 'sr-Latn', truth: L('Serbian', 'srp'), pred: L('Croatian', 'hrv'), conf: 0.49, cause: 'Orthographic overlap' },
  { id: 'e6', text: 'Meeting lúc 3pm, nhớ gửi deck và report cho team.', lang: 'vi', truth: L('Vietnamese', 'vie'), pred: L('English', 'eng'), conf: 0.55, cause: 'Code-mixing / Loanwords' },
  { id: 'e7', text: 'Madrid, 2019. ISBN 978-84-376-0494-7.', lang: 'es', truth: L('Spanish', 'spa'), pred: L('English', 'eng'), conf: 0.38, cause: 'Names & numerals (little linguistic signal)' },
];

// ---------------------------------------------------------------- formatters
const fmtPct = (v) => `${(v * 100).toFixed(1)}%`;
const fmtF1 = (v) => (v < 0.001 ? '<0.001' : v.toFixed(3));
const fmtMs = (v) => (v === null ? '—' : `${v}ms`);
const sign = (n) => (n > 0 ? '+' : n < 0 ? '−' : '±');
const delta = (n, digits, unit) => `${sign(n)}${Math.abs(n).toFixed(digits)}${unit}`;

// -------------------------------------------------------------------- pieces
function LangBadge({ lang, tone }) {
  const styles =
    tone === 'good'
      ? 'border-teal/30 bg-teal/10 text-teal'
      : 'border-red/30 bg-red/10 text-red';
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${styles}`}>
      {lang.name}
      <span className="font-mono font-normal">{lang.code}</span>
    </span>
  );
}

function Metric({ label, value, sublabel, extra, highlight = false }) {
  return (
    <li className="rounded-xl border border-border bg-surface p-5">
      <p className="text-[13px] font-medium text-muted">{label}</p>
      <p className={`mt-2 text-[32px] font-bold leading-none ${highlight ? 'text-teal' : 'text-ink'}`}>{value}</p>
      <p className="mt-2 text-sm text-muted">{sublabel}</p>
      {extra}
    </li>
  );
}

/**
 * Model Evaluation & Error Analysis (DS-03, DS-04, FR8, FR9).
 * Benchmark figures, confusion rates and error samples are mock data (see the constants above).
 */
export default function ModelEvaluation() {
  const [selectedId, setSelectedId] = useState('svm');
  const groupName = useId();
  const model = MODELS.find((m) => m.id === selectedId);
  const vsActive = model.id !== ACTIVE.id;

  const deltaLine = (text) =>
    vsActive ? <p className="mt-1 text-xs text-muted">{text} vs active</p> : <p className="mt-1 text-xs text-muted">Active model</p>;

  const latencyOk = model.p95 !== null && model.p95 < NFR1_P95_MS;

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------ header + picker */}
      <div>
        <h2 className="text-2xl font-bold text-ink">Model Evaluation &amp; Error Analysis</h2>
        <p className="mt-1 text-sm text-muted">
          Comparative test metrics, confusion matrix highlights, and error breakdown across{' '}
          {N_CLASSES} languages.
        </p>

        <fieldset className="mt-4">
          <legend className="sr-only">Model to evaluate</legend>
          <div className="grid gap-2 md:grid-cols-3">
            {MODELS.map((m) => (
              <label key={m.id} className="block cursor-pointer">
                <input
                  type="radio"
                  name={groupName}
                  value={m.id}
                  checked={m.id === selectedId}
                  onChange={() => setSelectedId(m.id)}
                  className="peer sr-only"
                />
                <span className="flex min-h-[44px] items-center rounded-xl border border-field bg-surface px-4 py-2 text-sm text-ink transition-colors hover:bg-subtle peer-checked:border-teal peer-checked:bg-chip peer-checked:font-medium peer-checked:text-teal peer-focus-visible:ring-4 peer-focus-visible:ring-teal/30">
                  {m.pickerLabel}
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <p role="status" className="sr-only">
          Showing metrics for {model.pickerLabel}
        </p>
      </div>

      {/* ------------------------------------------- headline metrics grid */}
      <ul aria-label={`Headline metrics for ${model.pickerLabel}`} className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          highlight
          label="Macro-F1 Score"
          value={fmtF1(model.macroF1)}
          sublabel={`Mean unweighted F1 over ${N_CLASSES} classes`}
          extra={deltaLine(delta(model.macroF1 - ACTIVE.macroF1, 3, ''))}
        />
        <Metric
          label="Test Accuracy"
          value={fmtPct(model.testAcc)}
          sublabel="Share of test samples labeled correctly"
          extra={deltaLine(delta((model.testAcc - ACTIVE.testAcc) * 100, 1, ' pts'))}
        />
        <Metric
          label="Inference Latency (p95)"
          value={fmtMs(model.p95)}
          sublabel={
            model.p95 === null ? (
              'Not measured'
            ) : (
              <span className={`inline-flex items-center gap-1.5 ${latencyOk ? 'text-teal' : 'text-red'}`}>
                {latencyOk ? (
                  <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <AlertTriangle className="h-4 w-4" aria-hidden="true" />
                )}
                {latencyOk ? `Meets NFR1 (< ${NFR1_P95_MS}ms)` : `Exceeds NFR1 (< ${NFR1_P95_MS}ms)`}
              </span>
            )
          }
          extra={
            model.p95 !== null && vsActive ? (
              <p className="mt-1 text-xs text-muted">{delta(model.p95 - ACTIVE.p95, 0, 'ms')} vs active</p>
            ) : model.p95 !== null ? (
              <p className="mt-1 text-xs text-muted">Active model</p>
            ) : null
          }
        />
        <Metric
          label="Test Samples Evaluated"
          value={TEST_SIZE.toLocaleString('en-US')}
          sublabel="Single-pass evaluation on clean test set"
        />
      </ul>

      {/* --------------------------------------------- comparison table (FR8) */}
      <section aria-labelledby="compare-title" className="rounded-xl border border-border bg-surface">
        <h3 id="compare-title" className="p-4 text-lg font-bold text-ink sm:p-5">
          Algorithm Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <caption className="sr-only">Comparison of the trained models and the baseline</caption>
            <thead>
              <tr className="border-y border-border text-xs text-muted">
                {['Model / Version', 'Algorithm', 'Hyperparameters', 'Val Accuracy', 'Test Accuracy', 'Test Macro-F1', 'p95 Latency'].map((h, i) => (
                  <th key={h} scope="col" className={`px-4 py-3 font-medium sm:px-5 ${i >= 3 ? 'text-right' : ''}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {TABLE_ORDER.map((id) => {
                const m = MODELS.find((x) => x.id === id);
                const selected = m.id === selectedId;
                return (
                  <tr key={m.id} aria-current={selected ? 'true' : undefined} className={selected ? 'bg-subtle' : ''}>
                    <td className="px-4 py-3.5 sm:px-5">
                      <span className="font-medium text-ink">{m.name}</span>{' '}
                      <span className="font-mono text-xs text-muted">{m.version}</span>
                      {m.active && (
                        <span className="ml-2 rounded-md bg-chip px-2 py-0.5 font-mono text-xs text-teal">Active</span>
                      )}
                    </td>
                    <td className="px-4 py-3.5 text-ink sm:px-5">{m.algorithm}</td>
                    <td className="px-4 py-3.5 font-mono text-xs text-ink sm:px-5">{m.params}</td>
                    <td className="px-4 py-3.5 text-right tabular-nums text-ink sm:px-5">{fmtPct(m.valAcc)}</td>
                    <td className="px-4 py-3.5 text-right tabular-nums text-ink sm:px-5">{fmtPct(m.testAcc)}</td>
                    <td className="px-4 py-3.5 text-right tabular-nums text-ink sm:px-5">{fmtF1(m.macroF1)}</td>
                    <td className="px-4 py-3.5 text-right tabular-nums text-ink sm:px-5">{fmtMs(m.p95)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ------------------------------------------ most confused pairs (FR9) */}
      <section aria-labelledby="pairs-title" className="rounded-xl border border-border bg-surface p-4 sm:p-5">
        <h3 id="pairs-title" className="text-lg font-bold text-ink">
          Most Confused Language Pairs
        </h3>
        <p className="mt-1 text-xs text-muted">
          From {ANALYSIS_MODEL}. Confusion rate: share of a language&apos;s test samples predicted as
          its counterpart (worse direction).
        </p>
        <ul className="mt-4 space-y-3">
          {CONFUSED_PAIRS.map((p) => {
            const high = p.rate >= HIGH_CONFUSION;
            return (
              <li key={p.id} className="rounded-xl bg-subtle p-3.5">
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
                  <p className="min-w-0 text-sm text-ink">
                    <span className="font-medium">{p.a.name}</span>{' '}
                    <span className="font-mono text-xs text-muted">({p.a.code})</span>
                    <span className="text-muted"> vs </span>
                    {p.b.map((l, i) => (
                      <span key={l.code}>
                        {i > 0 && <span className="text-muted"> / </span>}
                        <span className="font-medium">{l.name}</span>{' '}
                        <span className="font-mono text-xs text-muted">({l.code})</span>
                      </span>
                    ))}
                    {p.note && <span className="text-muted"> ({p.note})</span>}
                  </p>
                  <div className="flex items-center gap-2">
                    {high && (
                      <span className="inline-flex items-center gap-1 rounded-full border border-amber/40 bg-amber/10 px-2.5 py-0.5 text-xs font-medium text-ink">
                        <AlertTriangle className="h-3.5 w-3.5 text-amber" aria-hidden="true" />
                        High confusion
                      </span>
                    )}
                    <span className="text-sm font-bold tabular-nums text-ink">Confused: ~{Math.round(p.rate * 100)}%</span>
                  </div>
                </div>
                <div aria-hidden="true" className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-border">
                  <div
                    className={`h-full rounded-full ${high ? 'bg-amber' : 'bg-teal'}`}
                    style={{ width: `${(p.rate / MAX_RATE) * 100}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      {/* ------------------------------------------------ error analysis (FR9) */}
      <section aria-labelledby="errors-title" className="rounded-xl border border-border bg-surface">
        <div className="p-4 sm:p-5">
          <h3 id="errors-title" className="text-lg font-bold text-ink">
            Error Breakdown &amp; Failure Modes
          </h3>
          <p className="mt-1 text-xs text-muted">Misclassified test snippets from {ANALYSIS_MODEL}.</p>
        </div>

        {/* md and up: table */}
        <div className="hidden overflow-x-auto md:block">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Misclassified test snippets with their root cause</caption>
            <thead>
              <tr className="border-y border-border text-xs text-muted">
                {['Snippet Text', 'Ground Truth', 'Model Prediction', 'Confidence', 'Root Cause Diagnosis'].map((h) => (
                  <th key={h} scope="col" className="px-4 py-3 font-medium sm:px-5">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {ERRORS.map((e) => (
                <tr key={e.id} className="align-top">
                  <td className="max-w-xs px-4 py-3.5 text-ink sm:px-5">
                    <span lang={e.lang} className="break-words">{e.text}</span>
                  </td>
                  <td className="px-4 py-3.5 sm:px-5"><LangBadge lang={e.truth} tone="good" /></td>
                  <td className="px-4 py-3.5 sm:px-5"><LangBadge lang={e.pred} tone="bad" /></td>
                  <td className="px-4 py-3.5 tabular-nums text-ink sm:px-5">{Math.round(e.conf * 100)}%</td>
                  <td className="px-4 py-3.5 text-ink sm:px-5">{e.cause}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* phones: one card per error */}
        <ul aria-label="Misclassified test snippets" className="divide-y divide-border border-t border-border md:hidden">
          {ERRORS.map((e) => (
            <li key={e.id} className="space-y-2.5 p-4">
              <p lang={e.lang} className="break-words text-sm text-ink">“{e.text}”</p>
              <p className="flex flex-wrap items-center gap-2 text-xs text-muted">
                <span className="sr-only">Ground truth:</span>
                <LangBadge lang={e.truth} tone="good" />
                <span aria-hidden="true">→</span>
                <span className="sr-only">Model prediction:</span>
                <LangBadge lang={e.pred} tone="bad" />
                <span className="tabular-nums">{Math.round(e.conf * 100)}% confidence</span>
              </p>
              <p className="text-sm text-muted">{e.cause}</p>
            </li>
          ))}
        </ul>
      </section>

      <p className="text-xs text-muted">
        Simulated data: benchmark figures, confusion rates and error samples are placeholders until the
        models are trained and <span className="font-mono">ml/evaluate.py</span> is implemented.
      </p>
    </div>
  );
}

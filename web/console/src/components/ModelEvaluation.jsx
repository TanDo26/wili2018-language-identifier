import { useEffect, useId, useState } from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { apiFetch } from '../api/client.js';
import { DATASET_STATS, stats } from '../data/sampleDataset.js';
import labelsCsv from '../../../../data/processed/labels.csv?raw';

const N_CLASSES = DATASET_STATS.languages;
const TEST_SIZE = DATASET_STATS.splits.find((s) => s.id === 'test').size;
const NFR1_P95_MS = 300; // NFR1: p95 < 300 ms for 1,000 characters on CPU
const HIGH_CONFUSION = 0.1; // pairs at or above this rate get a warning pill
const TOP_PAIRS = 5;

const ALGORITHM_NAMES = { naive_bayes: 'Multinomial NB', svm: 'Linear SVM', baseline: 'Majority Class' };

// code -> { name, train, val, test } (stats.json); wiki_code (labels.csv, no quoted fields) is the closest
// thing to a BCP-47 tag for the lang attribute, which picks fonts and the screen reader voice.
const LANGS = Object.fromEntries(stats.languages.map((l) => [l.code, l]));
const BCP47 = Object.fromEntries(
  labelsCsv.trim().split(/\r?\n/).slice(1).map((line) => {
    const [code, , wiki] = line.split(',');
    return [code, wiki];
  }),
);
const lang = (code) => ({ code, name: LANGS[code]?.name ?? code });

// Majority-class baseline (ml/baseline.py) computed from the split sizes: it always predicts the language
// with the most train paragraphs, so accuracy = that language's share of the split. That class has
// precision = share and recall = 1, every other class has F1 = 0, so macro-F1 = 2·share / (1 + share) / 235.
const MAJORITY = stats.languages.reduce((a, b) => (b.train > a.train ? b : a));
const share = (split) => MAJORITY[split] / stats.splits[split].num_samples;
const BASELINE = {
  id: 'majority',
  pickerLabel: 'Majority Class (Baseline)',
  name: 'Majority Class',
  version: 'baseline',
  algorithm: 'Majority class',
  params: 'strategy = most_frequent',
  valAcc: share('val'),
  testAcc: share('test'),
  macroF1: (2 * share('test')) / (1 + share('test')) / N_CLASSES,
  p95: null, // not measured
  active: false,
  pairs: [],
  errors: [],
};

const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/** GET /api/v1/models item + GET /api/v1/models/{version} body (null if it failed) -> one row of this page. */
function toModel({ version, status }, details) {
  const { meta = {}, metrics = {}, train_info: info = {} } = details ?? {};
  const name = ALGORITHM_NAMES[meta.algorithm] ?? meta.algorithm ?? 'Unknown';
  return {
    id: version,
    pickerLabel: `${name} (${version} - ${capitalize(status)})`,
    name,
    version,
    algorithm: meta.algorithm ?? '—',
    params: info.hyperparameter ? `${info.hyperparameter} = ${info.value}` : '—',
    valAcc: info.val?.full?.accuracy ?? null,
    testAcc: metrics.accuracy ?? null,
    macroF1: metrics.macro_f1 ?? null,
    p95: metrics.latency_ms_p95 ?? null,
    active: status === 'active',
    // Confusion rate = share of the true language's test paragraphs predicted as the other one.
    pairs: (metrics.top_confused_pairs ?? []).slice(0, TOP_PAIRS).map((p) => ({
      id: `${p.true}-${p.pred}`,
      a: lang(p.true),
      b: [lang(p.pred)],
      count: p.count,
      rate: p.count / (LANGS[p.true]?.test ?? p.count),
    })),
    // null: metrics.json was written before ml/evaluate.py recorded error_samples
    errors:
      metrics.error_samples?.map((e) => ({
        id: `${e.id}@${e.length}`,
        text: e.text,
        lang: BCP47[e.true],
        truth: lang(e.true),
        pred: lang(e.pred),
        conf: e.confidence,
        cause: e.length === 'full' ? 'Full paragraph' : `Cut to ${e.length} characters`,
      })) ?? null,
  };
}

// ---------------------------------------------------------------- formatters
const fmtPct = (v) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`);
const fmtF1 = (v) => (v === null ? '—' : v < 0.001 ? '<0.001' : v.toFixed(3));
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
 * Each trained version comes from GET /api/v1/models and GET /api/v1/models/{version}
 * (metrics.json of ml/evaluate.py, train_info.json of ml/train.py); the baseline row is computed above.
 */
export default function ModelEvaluation() {
  const [state, setState] = useState({ phase: 'loading', models: [], reason: '' }); // 'loading' | 'live' | 'offline'
  const [selectedId, setSelectedId] = useState(null);
  const groupName = useId();

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    (async () => {
      try {
        const list = await apiFetch('/api/v1/models', { signal });
        // A version whose details fail to load keeps its row, only without figures.
        const details = await Promise.all(
          list.map((m) => apiFetch(`/api/v1/models/${m.version}`, { signal }).catch(() => null)),
        );
        if (!signal.aborted) setState({ phase: 'live', models: list.map((m, i) => toModel(m, details[i])), reason: '' });
      } catch (err) {
        if (!signal.aborted) setState({ phase: 'offline', models: [], reason: err?.message ?? '' });
      }
    })();
    return () => controller.abort();
  }, []);

  // FR8 order: active model first, then the other versions, then the baseline.
  const trained = [...state.models].sort((a, b) => b.active - a.active);
  const models = [...trained, BASELINE];
  const active = trained.find((m) => m.active) ?? null;
  const model = models.find((m) => m.id === selectedId) ?? active ?? models[0];
  const vsActive = active !== null && model.id !== active.id;

  const compare = (key, digits, unit, scale = 1) => {
    if (!active || model[key] === null) return null;
    if (!vsActive) return <p className="mt-1 text-xs text-muted">Active model</p>;
    if (active[key] === null) return null;
    return <p className="mt-1 text-xs text-muted">{delta((model[key] - active[key]) * scale, digits, unit)} vs active</p>;
  };

  const latencyOk = model.p95 !== null && model.p95 < NFR1_P95_MS;
  const maxRate = Math.max(0, ...model.pairs.map((p) => p.rate));

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
            {models.map((m) => (
              <label key={m.id} className="block cursor-pointer">
                <input
                  type="radio"
                  name={groupName}
                  value={m.id}
                  checked={m.id === model.id}
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

      {state.phase === 'loading' && <p className="text-sm text-muted">Loading trained models from the API…</p>}
      {state.phase !== 'loading' && trained.length === 0 && (
        <p role="alert" className="rounded-xl border border-amber/40 bg-amber/10 p-3.5 text-sm text-ink">
          {state.phase === 'offline' ? `API unreachable (${state.reason}). ` : 'The API has no trained model yet. '}
          Run <span className="font-mono">make retrain</span> and start the API; until then only the baseline is shown.
        </p>
      )}

      {/* ------------------------------------------- headline metrics grid */}
      <ul aria-label={`Headline metrics for ${model.pickerLabel}`} className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          highlight
          label="Macro-F1 Score"
          value={fmtF1(model.macroF1)}
          sublabel={`Mean unweighted F1 over ${N_CLASSES} classes`}
          extra={compare('macroF1', 3, '')}
        />
        <Metric
          label="Test Accuracy"
          value={fmtPct(model.testAcc)}
          sublabel="Share of test samples labeled correctly"
          extra={compare('testAcc', 1, ' pts', 100)}
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
          extra={compare('p95', 2, 'ms')}
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
              {models.map((m) => {
                const selected = m.id === model.id;
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
      {model.pairs.length > 0 && (
        <section aria-labelledby="pairs-title" className="rounded-xl border border-border bg-surface p-4 sm:p-5">
          <h3 id="pairs-title" className="text-lg font-bold text-ink">
            Most Confused Language Pairs
          </h3>
          <p className="mt-1 text-xs text-muted">
            From {model.name} {model.version}. Confusion rate: share of the first language&apos;s test paragraphs
            predicted as the second.
          </p>
          <ul className="mt-4 space-y-3">
            {model.pairs.map((p) => {
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
                    </p>
                    <div className="flex items-center gap-2">
                      {high && (
                        <span className="inline-flex items-center gap-1 rounded-full border border-amber/40 bg-amber/10 px-2.5 py-0.5 text-xs font-medium text-ink">
                          <AlertTriangle className="h-3.5 w-3.5 text-amber" aria-hidden="true" />
                          High confusion
                        </span>
                      )}
                      <span className="text-sm font-bold tabular-nums text-ink">
                        Confused: ~{Math.round(p.rate * 100)}% ({p.count})
                      </span>
                    </div>
                  </div>
                  <div aria-hidden="true" className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-border">
                    <div
                      className={`h-full rounded-full ${high ? 'bg-amber' : 'bg-teal'}`}
                      style={{ width: `${(p.rate / maxRate) * 100}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {/* ------------------------------------------------ error analysis (FR9) */}
      {model.errors === null && (
        <p className="text-sm text-muted">
          No error samples recorded for {model.version}. Re-run{' '}
          <span className="font-mono">python -m ml.evaluate --model-dir models/{model.version}</span>.
        </p>
      )}
      {model.errors?.length > 0 && (
        <section aria-labelledby="errors-title" className="rounded-xl border border-border bg-surface">
          <div className="p-4 sm:p-5">
            <h3 id="errors-title" className="text-lg font-bold text-ink">
              Error Breakdown &amp; Failure Modes
            </h3>
            <p className="mt-1 text-xs text-muted">
              Misclassified test snippets from {model.name} {model.version}, picked at random per text length.
            </p>
          </div>

          {/* md and up: table */}
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full text-left text-sm">
              <caption className="sr-only">Misclassified test snippets with their text length</caption>
              <thead>
                <tr className="border-y border-border text-xs text-muted">
                  {['Snippet Text', 'Ground Truth', 'Model Prediction', 'Confidence', 'Text Length'].map((h) => (
                    <th key={h} scope="col" className="px-4 py-3 font-medium sm:px-5">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {model.errors.map((e) => (
                  <tr key={e.id} className="align-top">
                    <td className="max-w-xs px-4 py-3.5 text-ink sm:px-5">
                      <span lang={e.lang} dir="auto" className="break-words">{e.text}</span>
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
            {model.errors.map((e) => (
              <li key={e.id} className="space-y-2.5 p-4">
                <p lang={e.lang} dir="auto" className="break-words text-sm text-ink">“{e.text}”</p>
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
      )}

      <p className="text-xs text-muted">
        Trained versions come from <span className="font-mono">GET /api/v1/models/&#123;version&#125;</span>{' '}
        (metrics.json of <span className="font-mono">ml/evaluate.py</span>). The baseline row is computed from{' '}
        <span className="font-mono">data/processed/stats.json</span>.
      </p>
    </div>
  );
}

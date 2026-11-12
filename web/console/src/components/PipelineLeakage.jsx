import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FileCheck2,
  Loader2,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { DATASET_STATS, TOTAL_PARAGRAPHS } from '../data/sampleDataset.js';

const SEED = 42;
const AUDIT_DURATION_MS = 1600; // simulated audit time
const NEAR_DUPLICATE_THRESHOLD = 0.9;

// Mock audit results (DS-02). Replace with an API response once an audit endpoint exists.
// kind 'exact' = SHA-256 string overlap between two splits (must be 0); 'near' = review-only check.
const AUDIT_CHECKS = [
  { id: 'train-test', kind: 'exact', name: 'Train ∩ Test Overlap', count: 0 },
  { id: 'train-val', kind: 'exact', name: 'Train ∩ Validation Overlap', count: 0 },
  { id: 'val-test', kind: 'exact', name: 'Validation ∩ Test Overlap', count: 0 },
  { id: 'near-dup', kind: 'near', name: `Near-duplicate Jaccard similarity (>${NEAR_DUPLICATE_THRESHOLD.toFixed(2)})`, count: 14 },
];

// Features agreed for the TF-IDF model (the team's `features.py`).
const TFIDF_CONFIG = [
  ['analyzer', "'char'"],
  ['ngram_range', '(1, 3)'],
  ['sublinear_tf', 'true'],
  ['min_df', '2'],
  ['max_features', '50000'],
];
const SAMPLE_NGRAMS = [' vi', 'ngư', 'nam', 'the', 'ing'];

const fmtK = (n) => `${n / 1000}k`;
const pctOf = (part, whole) => +((part / whole) * 100).toFixed(1);

function statusOf(check) {
  if (check.kind === 'near') return 'acceptable';
  return check.count === 0 ? 'passed' : 'failed';
}

function resultText(check) {
  if (check.kind === 'near') return `${check.count} pairs flagged for review`;
  const base = `${check.count} duplicate samples`;
  return check.id === 'train-test' && check.count === 0 ? `${base} (100% disjoint)` : base;
}

const STATUS_UI = {
  passed: { label: 'PASSED', Icon: CheckCircle2, wrap: 'text-teal', icon: 'text-teal' },
  acceptable: {
    label: 'ACCEPTABLE',
    Icon: AlertTriangle,
    wrap: 'rounded-full border border-amber/40 bg-amber/10 px-2.5 py-0.5 text-ink',
    icon: 'text-amber',
  },
  failed: { label: 'FAILED', Icon: XCircle, wrap: 'text-red', icon: 'text-red' },
};

function Spinner({ className = 'h-4 w-4' }) {
  return <Loader2 className={`${className} animate-spin motion-reduce:animate-none`} aria-hidden="true" />;
}

function Card({ title, id, children }) {
  return (
    <section aria-labelledby={id} className="rounded-xl border border-border bg-surface p-5">
      <h3 id={id} className="text-base font-bold text-ink">
        {title}
      </h3>
      {children}
    </section>
  );
}

/**
 * Data Pipeline & Leakage Verification (DS-02, FR7, NFR2).
 * Shows the agreed preprocessing / split / feature configuration and a leakage audit.
 * The audit is simulated: figures come from the constants above, not from the pipeline.
 */
export default function PipelineLeakage() {
  const [running, setRunning] = useState(false);
  const [lastRun, setLastRun] = useState(null);
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const runAudit = () => {
    if (running) return;
    setRunning(true);
    timer.current = setTimeout(() => {
      setRunning(false);
      setLastRun(new Date());
    }, AUDIT_DURATION_MS);
  };

  // ---- derived values
  const exactChecks = AUDIT_CHECKS.filter((c) => c.kind === 'exact');
  const clean = exactChecks.every((c) => c.count === 0);
  const trainTest = AUDIT_CHECKS.find((c) => c.id === 'train-test');
  const testSize = DATASET_STATS.splits.find((s) => s.id === 'test').size;
  const overlapPct = pctOf(trainTest.count, testSize).toFixed(1);
  const passedCount = AUDIT_CHECKS.filter((c) => statusOf(c) === 'passed').length;
  const acceptableCount = AUDIT_CHECKS.filter((c) => statusOf(c) === 'acceptable').length;

  let liveText = 'Showing results from the last pipeline run.';
  if (running) liveText = 'Running leakage audit…';
  else if (lastRun)
    liveText = `Leakage audit finished at ${lastRun.toLocaleTimeString('en-GB')}: ${passedCount} passed, ${acceptableCount} acceptable.`;

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------- header + trigger */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-2xl font-bold text-ink">Data Pipeline &amp; Leakage Verification</h2>
          <p className="mt-1 text-sm text-muted">
            Preprocessing, train/validation/test split configuration, and data leakage audit.
          </p>
          <p role="status" aria-live="polite" className="mt-2 text-xs text-muted">
            {liveText}
          </p>
        </div>
        <button
          type="button"
          onClick={runAudit}
          disabled={running}
          aria-busy={running}
          className="inline-flex min-h-[44px] shrink-0 items-center justify-center gap-2 rounded-xl bg-teal px-5 font-medium text-white shadow-sm transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-wait"
        >
          {running ? <Spinner /> : <ShieldCheck className="h-4 w-4" aria-hidden="true" />}
          {running ? 'Running audit…' : 'Run Leakage Audit'}
        </button>
      </div>

      {/* ---------------------------------------------- status overview cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card id="card-clean" title="Data Cleaning & Normalization">
          <span className="mt-3 inline-block rounded-full bg-chip px-2.5 py-1 font-mono text-xs text-teal">
            NFC Unicode + Deduplicated
          </span>
          <p className="mt-4 text-sm text-ink">
            <strong className="font-bold">{TOTAL_PARAGRAPHS.toLocaleString('en-US')}</strong>{' '}
            paragraphs cleaned, <strong className="font-bold">0</strong> corrupted lines removed
          </p>
        </Card>

        <Card id="card-split" title={`Reproducible Split (Seed ${SEED})`}>
          <div
            role="img"
            aria-label={`Split distribution: ${DATASET_STATS.splits
              .map((s) => `${s.label} ${pctOf(s.size, TOTAL_PARAGRAPHS)}%`)
              .join(', ')}`}
            className="mt-4 flex h-3 gap-0.5 overflow-hidden rounded-full"
          >
            {DATASET_STATS.splits.map((s, i) => (
              <div
                key={s.id}
                style={{ width: `${pctOf(s.size, TOTAL_PARAGRAPHS)}%` }}
                className={['bg-teal', 'bg-ink', 'bg-field'][i]}
              />
            ))}
          </div>
          <ul className="mt-3 space-y-1.5 text-sm text-ink">
            {DATASET_STATS.splits.map((s, i) => (
              <li key={s.id} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className={`h-2.5 w-2.5 rounded-sm ${['bg-teal', 'bg-ink', 'bg-field'][i]}`}
                  />
                  {s.short.charAt(0).toUpperCase() + s.short.slice(1)}:
                </span>
                <span>
                  {pctOf(s.size, TOTAL_PARAGRAPHS)}% ({fmtK(s.size)})
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-4 flex items-center gap-1.5 font-mono text-xs text-muted">
            <FileCheck2 className="h-3.5 w-3.5" aria-hidden="true" />
            split_ids.parquet saved
          </p>
        </Card>

        <Card id="card-leak" title="Data Leakage Status">
          {running ? (
            <p className="mt-3 inline-flex items-center gap-2 rounded-xl border border-border bg-subtle px-3.5 py-2 text-base font-bold text-muted">
              <Spinner className="h-5 w-5" />
              Checking…
            </p>
          ) : clean ? (
            <p className="mt-3 inline-flex items-center gap-2 rounded-xl border border-teal/30 bg-teal/10 px-3.5 py-2 text-base font-bold text-teal">
              <ShieldCheck className="h-5 w-5" aria-hidden="true" />
              {overlapPct}% Overlap · CLEAN
            </p>
          ) : (
            <p className="mt-3 inline-flex items-center gap-2 rounded-xl border border-red/30 bg-red/10 px-3.5 py-2 text-base font-bold text-red">
              <XCircle className="h-5 w-5" aria-hidden="true" />
              {overlapPct}% Overlap · LEAKAGE
            </p>
          )}
          <p className="mt-4 text-sm text-ink">
            Train vs Test overlap: {trainTest.count} rows (checked via SHA-256 hashes)
          </p>
        </Card>
      </div>

      {/* --------------------------------------------------- audit results */}
      <section aria-labelledby="audit-title" className="rounded-xl border border-border bg-surface">
        <h3 id="audit-title" className="p-4 text-lg font-bold text-ink sm:p-5">
          Leakage Check Results Across Splits
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Cross-split leakage checks and their status</caption>
            <thead>
              <tr className="border-y border-border text-xs text-muted">
                <th scope="col" className="px-4 py-3 font-medium sm:px-5">
                  Check
                </th>
                <th scope="col" className="hidden px-4 py-3 font-medium sm:table-cell sm:px-5">
                  Result
                </th>
                <th scope="col" className="px-4 py-3 font-medium sm:px-5">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {AUDIT_CHECKS.map((c) => {
                const ui = STATUS_UI[statusOf(c)];
                return (
                  <tr key={c.id} className="align-top">
                    <td className="px-4 py-3.5 sm:px-5">
                      <span className="font-mono text-[13px] text-ink">{c.name}</span>
                      {/* On phones the result moves under the check name. */}
                      <span className="mt-1 block text-muted sm:hidden">{resultText(c)}</span>
                    </td>
                    <td className="hidden px-4 py-3.5 text-ink sm:table-cell sm:px-5">{resultText(c)}</td>
                    <td className="px-4 py-3.5 sm:px-5">
                      {running ? (
                        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-muted">
                          <Spinner className="h-4 w-4" />
                          CHECKING
                        </span>
                      ) : (
                        <span
                          className={`inline-flex items-center gap-1.5 text-xs font-medium tracking-wide ${ui.wrap}`}
                        >
                          <ui.Icon className={`h-4 w-4 ${ui.icon}`} aria-hidden="true" />
                          {ui.label}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ------------------------------------------------ feature extraction */}
      <section aria-labelledby="features-title" className="rounded-xl border border-border bg-surface p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 id="features-title" className="text-lg font-bold text-ink">
            Feature Extraction Pipeline (TF-IDF)
          </h3>
          <span className="rounded-md bg-chip px-2 py-0.5 font-mono text-xs text-[#2B4442]">features.py</span>
        </div>

        <pre
          tabIndex={0}
          aria-label="TF-IDF configuration"
          className="mt-3 overflow-x-auto rounded-lg bg-subtle p-4 font-mono text-sm leading-relaxed text-ink"
        >
          {TFIDF_CONFIG.map(([key, value]) => (
            <code key={key} className="block">
              <span className="text-teal">{key}</span>: {value}
            </code>
          ))}
        </pre>

        <h4 className="mt-4 text-[13px] font-medium text-muted">Sample character 3-grams</h4>
        <ul className="mt-2 flex flex-wrap gap-2">
          {SAMPLE_NGRAMS.map((g) => (
            <li
              key={g}
              className="whitespace-pre rounded-md border border-border bg-chip px-2 py-1 font-mono text-xs text-[#2B4442]"
            >
              &apos;{g}&apos;
            </li>
          ))}
        </ul>
      </section>

      <p className="text-xs text-muted">
        Simulated data: audit results are placeholders until the data pipeline and an audit endpoint
        are available.
      </p>
    </div>
  );
}

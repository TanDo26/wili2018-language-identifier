import { AlertTriangle, CheckCircle2, FileCheck2, ShieldCheck, XCircle } from 'lucide-react';
import { DATASET_STATS, TOTAL_PARAGRAPHS, stats } from '../data/sampleDataset.js';

const { leakage, cleaning, split } = stats;
const PAIR_NAMES = { 'train-test': 'Train ∩ Test', 'train-val': 'Train ∩ Validation', 'val-test': 'Validation ∩ Test' };
const fmt = (n) => n.toLocaleString('en-US');

// From data/processed/stats.json (python -m data.check_leakage). kind 'exact' must be 0: same id,
// identical text, or identical after dropping digits, punctuation, spaces and case.
// kind 'near' is report-only: test paragraphs sharing a 40-char passage with train (bot-made articles).
const AUDIT_CHECKS = [
  ...Object.entries(PAIR_NAMES).map(([pair, name]) => {
    const c = leakage.pairs[pair];
    return { id: pair, kind: 'exact', name: `${name} Overlap`, count: c.id + c.exact + c.near };
  }),
  {
    id: 'partial',
    kind: 'near',
    name: `Test text shared with train (≥ ${leakage.partial_overlap.window_chars} chars)`,
    count: leakage.partial_overlap.test.any,
  },
];

// shortcut: copied from DEFAULT_PRESET in ml/features.py; read meta.features of GET /api/v1/models/{v} if presets change.
const TFIDF_CONFIG = [
  ['preset', "'char_wb_1_4_100k'"],
  ['analyzer', "'char_wb'"],
  ['ngram_range', '(1, 4)'],
  ['max_features', '100000'],
  ['lowercase', 'true'],
  ['sublinear_tf', 'true'],
];
const SAMPLE_NGRAMS = [' vi', 'ngư', 'nam', 'the', 'ing'];

const fmtK = (n) => `${(n / 1000).toFixed(1)}k`;
const pctOf = (part, whole) => +((part / whole) * 100).toFixed(1);

function statusOf(check) {
  if (check.kind === 'near') return 'acceptable';
  return check.count === 0 ? 'passed' : 'failed';
}

function resultText(check) {
  if (check.kind === 'near')
    return `${fmt(check.count)} test paragraphs (${pctOf(check.count, DATASET_STATS.splits[2].size)}%); scored separately as no_overlap`;
  const base = `${fmt(check.count)} shared samples`;
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
 * Shows the preprocessing / split / feature configuration and the leakage audit of the last
 * data pipeline run (data/processed/stats.json, bundled at build time).
 */
export default function PipelineLeakage() {
  // ---- derived values
  const exactChecks = AUDIT_CHECKS.filter((c) => c.kind === 'exact');
  const clean = exactChecks.every((c) => c.count === 0);
  const trainTest = AUDIT_CHECKS.find((c) => c.id === 'train-test');
  const testSize = DATASET_STATS.splits.find((s) => s.id === 'test').size;
  const overlapPct = pctOf(trainTest.count, testSize).toFixed(1);
  return (
    <div className="space-y-6">
      {/* ------------------------------------------------------- header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-2xl font-bold text-ink">Data Pipeline &amp; Leakage Verification</h2>
          <p className="mt-1 text-sm text-muted">
            Preprocessing, train/validation/test split configuration, and data leakage audit.
          </p>
          <p className="mt-2 text-xs text-muted">
            Results of the last <span className="font-mono">python -m data.pipeline</span> run.
          </p>
        </div>
      </div>

      {/* ---------------------------------------------- status overview cards */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card id="card-clean" title="Data Cleaning & Normalization">
          <span className="mt-3 inline-block rounded-full bg-chip px-2.5 py-1 font-mono text-xs text-teal">
            NFC Unicode + Deduplicated
          </span>
          <p className="mt-4 text-sm text-ink">
            <strong className="font-bold">{fmt(TOTAL_PARAGRAPHS)}</strong> of{' '}
            {fmt(cleaning.raw.train + cleaning.raw.test)} paragraphs kept,{' '}
            <strong className="font-bold">{fmt(cleaning.removed_total.train + cleaning.removed_total.test)}</strong>{' '}
            removed (duplicates, conflicting labels, no letters, train rows also in test)
          </p>
        </Card>

        <Card id="card-split" title={`Reproducible Split (Seed ${split.seed})`}>
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
            {split.split_ids_file} saved
          </p>
        </Card>

        <Card id="card-leak" title="Data Leakage Status">
          {clean ? (
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
            Train vs Test overlap: {trainTest.count} rows (same id, identical or near-identical text)
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
                      <span
                        className={`inline-flex items-center gap-1.5 text-xs font-medium tracking-wide ${ui.wrap}`}
                      >
                        <ui.Icon className={`h-4 w-4 ${ui.icon}`} aria-hidden="true" />
                        {ui.label}
                      </span>
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
          <span className="rounded-md bg-chip px-2 py-0.5 font-mono text-xs text-[#2B4442]">ml/features.py</span>
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
        Figures come from <span className="font-mono">data/processed/stats.json</span>. Re-run{' '}
        <span className="font-mono">python -m data.pipeline</span> and rebuild the console to refresh them.
      </p>
    </div>
  );
}

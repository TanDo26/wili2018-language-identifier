import { useCallback, useEffect, useId, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Info,
  KeyRound,
  Loader2,
  RotateCcw,
  Search,
  WifiOff,
  X,
} from 'lucide-react';
import { apiFetch, requestAdminToken, useAdminToken } from '../api/client.js';
import { useLanguages } from '../hooks/useLanguages.js';
import { findByCode, languageLabel, resolveLanguage } from '../lib/languages.js';
import LanguageCombobox from './LanguageCombobox.jsx';

// ------------------------------------------------------------------ fallbacks
// Shown when the API cannot be reached. Obviously-sample values, never presented as live.
const FALLBACK_MODELS = [
  { version: 'v1', algorithm: 'Multinomial Naive Bayes', status: 'active', createdAt: '2026-10-20T09:30:00Z', dataVersion: 'sha256:sample-offline-v1', accuracy: 0.915, macroF1: 0.908 },
  { version: 'v2', algorithm: 'Linear SVM', status: 'candidate', createdAt: '2026-11-01T10:00:00Z', dataVersion: 'sha256:sample-offline-v2', accuracy: 0.932, macroF1: 0.926 },
];

// Same fields as the API's ReportItem.
const SAMPLE_REPORTS = [
  { report_id: 6, text: 'Kal meeting hai, please sab log time par aana aur apna laptop saath lana.', predicted_language: 'eng', confidence: 0.71, model_version: 'v2', reason: 'wrong_language', suggested_language: 'hin', comment: 'Hinglish: Hindi written in Latin script', status: 'new', verified_language: null, created_at: '2026-10-31T16:42:10Z' },
  { report_id: 5, text: 'Ovo je jedan od najstarijih gradova u zemlji.', predicted_language: 'hrv', confidence: 0.54, model_version: 'v2', reason: 'wrong_language', suggested_language: 'srp', comment: null, status: 'new', verified_language: null, created_at: '2026-10-31T11:05:37Z' },
  { report_id: 4, text: 'Hvala vam puno na pomoći.', predicted_language: 'hrv', confidence: 0.52, model_version: 'v1', reason: 'wrong_language', suggested_language: 'bos', comment: 'I wrote this in Bosnian, very short sentence', status: 'accepted', verified_language: 'bos', created_at: '2026-10-30T19:20:02Z' },
  { report_id: 3, text: 'Meeting lúc 3pm, nhớ gửi deck và report cho team trước khi tan làm nhé.', predicted_language: 'eng', confidence: 0.55, model_version: 'v2', reason: 'mixed_languages', suggested_language: 'vie', comment: 'Vietnamese sentence with English work terms', status: 'new', verified_language: null, created_at: '2026-10-30T09:12:44Z' },
  { report_id: 2, text: 'for (int i = 0; i < n; i++) { total += xs[i]; } // TODO', predicted_language: 'eng', confidence: 0.31, model_version: 'v1', reason: 'not_a_language', suggested_language: null, comment: 'Source code, not a language', status: 'rejected', verified_language: null, created_at: '2026-10-29T21:48:15Z' },
  { report_id: 1, text: 'Det er en stor dag for hele familien i dag.', predicted_language: 'nob', confidence: 0.58, model_version: 'v1', reason: 'wrong_language', suggested_language: 'dan', comment: 'Danish speaker here', status: 'new', verified_language: null, created_at: '2026-10-28T08:31:59Z' },
];

const REASON_LABELS = {
  wrong_language: 'Wrong language',
  mixed_languages: 'Mixed languages',
  not_a_language: 'Not a language',
  other: 'Other issue',
};
const STATUS_FILTERS = ['all', 'new', 'accepted', 'rejected'];
const SNIPPET_MAX = 200;

// ------------------------------------------------------------------- helpers
const normalize = (s) =>
  s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').toLowerCase().trim();

const fmtPct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');
const fmtScore = (v) => (typeof v === 'number' ? v.toFixed(3) : '—');
const shortHash = (v) => (!v ? '—' : v.length > 22 ? `${v.slice(0, 22)}…` : v);

const snippet = (text) => {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > SNIPPET_MAX ? `${flat.slice(0, SNIPPET_MAX).trimEnd()}…` : flat;
};

function fmtDate(value) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/** Maps a raw GET /models item to the shape the UI uses (tolerates the old is_active schema). */
function normalizeModel(raw) {
  const status = ['active', 'candidate', 'archived'].includes(raw.status)
    ? raw.status
    : raw.is_active
      ? 'active'
      : 'candidate';
  return {
    version: raw.version,
    algorithm: raw.algorithm ?? '—',
    status,
    createdAt: raw.created_at ?? null,
    dataVersion: raw.data_version ?? null,
    accuracy: raw.metrics?.accuracy ?? raw.accuracy ?? null,
    macroF1: raw.metrics?.macro_f1 ?? raw.macro_f1 ?? null,
  };
}

const isModelList = (d) =>
  Array.isArray(d) &&
  d.length > 0 &&
  d.every((m) => m && typeof m.version === 'string' && (typeof m.status === 'string' || typeof m.is_active === 'boolean'));

const isReportList = (d) =>
  Array.isArray(d) && d.every((r) => r && r.report_id !== undefined && typeof r.text === 'string');

const versionKey = (v) => (v.match(/\d+/g) ?? []).map(Number);
function compareVersions(a, b) {
  const [x, y] = [versionKey(a), versionKey(b)];
  for (let i = 0; i < Math.max(x.length, y.length); i += 1) {
    const d = (x[i] ?? 0) - (y[i] ?? 0);
    if (d !== 0) return d;
  }
  return a.localeCompare(b);
}

/** Maps API failures of promote / rollback to the messages agreed for the console. */
function actionErrorMessage(err) {
  if (err.network) return err.message;
  switch (err.status) {
    case 401:
      return 'Unauthorized: Invalid or missing X-Admin-Token.';
    case 404:
      return 'Model version not found in models/ directory.';
    case 409:
      return 'Conflict: Version is already active, or no previous version exists in history to roll back to.';
    default:
      return 'The action failed. Try again.';
  }
}

/** Maps API failures of PATCH /reports/{id} to inline messages. */
function reviewErrorMessage(err) {
  if (err.network) return err.message;
  switch (err.status) {
    case 400:
      return 'The server rejected this review: a verified language is required to accept a report.';
    case 401:
      return 'Unauthorized: Invalid or missing X-Admin-Token.';
    case 404:
      return 'This report was not found. It may have been removed; refresh the list.';
    default:
      return 'The review could not be saved. Try again.';
  }
}

// ----------------------------------------------------------------- small pieces
function Spinner({ className = 'h-4 w-4' }) {
  return <Loader2 className={`${className} animate-spin motion-reduce:animate-none`} aria-hidden="true" />;
}

const MODEL_STATUS_UI = {
  active: { label: 'Active Production', cls: 'border border-teal/30 bg-teal/10 font-semibold text-teal' },
  candidate: { label: 'Candidate', cls: 'border border-border bg-subtle text-ink' },
  archived: { label: 'Archived', cls: 'border border-dashed border-field text-muted' },
};

function ModelStatusBadge({ status }) {
  const ui = MODEL_STATUS_UI[status];
  return <span className={`inline-block rounded-full px-2.5 py-1 text-xs ${ui.cls}`}>{ui.label}</span>;
}

const REPORT_STATUS_UI = {
  new: 'border border-amber/40 bg-amber/10 text-ink',
  accepted: 'border border-teal/30 bg-teal/10 text-teal',
  rejected: 'border border-border bg-subtle text-muted',
};

function ReportStatusBadge({ status }) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-1 text-xs font-semibold capitalize ${REPORT_STATUS_UI[status] ?? REPORT_STATUS_UI.rejected}`}
    >
      {status}
    </span>
  );
}

function LangBadge({ code, languages, tone }) {
  const styles = {
    bad: 'border-red/30 bg-red/10 text-red',
    good: 'border-teal/30 bg-teal/10 text-teal',
    neutral: 'border-border bg-subtle text-ink',
  }[tone];
  const name = findByCode(languages, code)?.name;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${styles}`}>
      {name && <>{name} </>}
      <span className="font-mono font-normal">{code}</span>
    </span>
  );
}

function PromoteAction({ model, pending, disabled, onPromote }) {
  if (model.status === 'active') return <span className="text-sm text-muted">Active Production</span>;
  const promoting = pending?.type === 'promote' && pending.version === model.version;
  return (
    <button
      type="button"
      onClick={() => onPromote(model.version)}
      disabled={disabled}
      aria-label={`Promote ${model.version} to Production`}
      className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg bg-teal px-3.5 py-1.5 text-sm font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-teal sm:min-h-[38px]"
    >
      {promoting && <Spinner />}
      {promoting ? 'Promoting…' : 'Promote to Production'}
    </button>
  );
}

// --------------------------------------------------------------------- page
/**
 * Model Registry & User Feedback Audit (DS-05, FR5, FR10; MV-2..MV-5, RV-1, RV-2).
 * Protected calls send X-Admin-Token (see src/api/client.js).
 */
export default function ModelRegistryReports() {
  const [models, setModels] = useState(null);
  const [modelsStatus, setModelsStatus] = useState('loading'); // 'loading' | 'live' | 'offline'
  const [pending, setPending] = useState(null); // { type: 'promote', version } | { type: 'rollback' } | null
  const [toast, setToast] = useState('');
  const [actionError, setActionError] = useState('');

  const loadModels = useCallback(async (silent = false) => {
    if (!silent) setModelsStatus('loading');
    try {
      const data = await apiFetch('/api/v1/models');
      if (!isModelList(data)) throw new Error('Unexpected models payload');
      setModels(data.map(normalizeModel));
      setModelsStatus('live');
    } catch {
      if (!silent) {
        setModels(FALLBACK_MODELS);
        setModelsStatus('offline');
      }
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const runAction = async (pendingState, path, body, successText) => {
    setPending(pendingState);
    setActionError('');
    try {
      const res = await apiFetch(path, { method: 'POST', body, auth: true });
      const activeVersion = res?.active_version ?? body?.version;
      const previous = res?.previous_version ?? null;
      // Show the swap immediately; the background re-sync below confirms it with the server.
      setModels((prev) =>
        prev.map((m) =>
          m.version === activeVersion
            ? { ...m, status: 'active' }
            : m.status === 'active' || m.version === previous
              ? { ...m, status: 'archived' }
              : m,
        ),
      );
      setToast(successText(activeVersion, previous));
      loadModels(true);
    } catch (err) {
      setActionError(actionErrorMessage(err));
    } finally {
      setPending(null);
    }
  };

  const promote = (version) =>
    runAction(
      { type: 'promote', version },
      '/api/v1/models/promote',
      { version },
      (v, prev) =>
        `Successfully promoted model ${v} to active production engine${prev ? ` (previous: ${prev})` : ''}. No container restart required.`,
    );

  const rollback = () =>
    runAction(
      { type: 'rollback' },
      '/api/v1/models/rollback',
      undefined,
      (v, prev) =>
        `Successfully rolled back to model ${v}${prev ? ` (replacing ${prev})` : ''} as the active production engine. No container restart required.`,
    );

  const active = models?.find((m) => m.status === 'active');
  const busy = pending !== null;
  const actionsDisabled = busy || modelsStatus !== 'live';
  const versionOptions = models ? [...models.map((m) => m.version)].sort(compareVersions) : [];

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------- header + rollback */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-2xl font-bold text-ink">Model Registry &amp; User Feedback Audit</h2>
          <p className="mt-1 text-sm text-muted">
            Hot-swap production inference versions with zero downtime (NFR5) and audit end-user
            misclassification reports (FR5).
          </p>
          {active && (
            <p className="mt-3 inline-flex items-center gap-2 text-sm text-ink">
              <span className="h-2 w-2 rounded-full bg-teal" aria-hidden="true" />
              Active production: <span className="font-mono font-bold">{active.version}</span> ({active.algorithm})
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={rollback}
          disabled={actionsDisabled}
          aria-busy={pending?.type === 'rollback'}
          className="inline-flex min-h-[44px] shrink-0 items-center gap-2 self-start rounded-xl border border-amber/40 bg-surface px-4 font-medium text-ink transition-colors hover:bg-amber/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-amber/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-surface"
        >
          {pending?.type === 'rollback' ? <Spinner /> : <RotateCcw className="h-4 w-4" aria-hidden="true" />}
          {pending?.type === 'rollback' ? 'Rolling back…' : 'Rollback to Previous Version'}
        </button>
      </div>

      {/* ------------------------------------------- success banner / errors */}
      <div role="status" aria-live="polite">
        {toast && (
          <div className="flex animate-toast-in items-start gap-3 rounded-xl border border-teal/30 bg-teal/10 p-3.5 text-sm text-ink motion-reduce:animate-none">
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-teal" aria-hidden="true" />
            <p className="flex-1">{toast}</p>
            <button
              type="button"
              onClick={() => setToast('')}
              aria-label="Dismiss notification"
              className="-m-2 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg text-muted hover:bg-teal/10 hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
      {actionError && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-red/30 bg-red/10 p-3.5 text-sm text-red">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <p className="flex-1">{actionError}</p>
          <button
            type="button"
            onClick={() => setActionError('')}
            aria-label="Dismiss error"
            className="-m-2 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-lg hover:bg-red/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-red/25"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      )}

      {/* --------------------------------------------------- registry table */}
      <section aria-labelledby="registry-title" className="rounded-xl border border-border bg-surface">
        <h3 id="registry-title" className="p-4 text-lg font-bold text-ink sm:p-5">
          Registered Production Models
        </h3>

        {modelsStatus === 'offline' && (
          <div className="mx-4 mb-4 flex flex-col gap-3 rounded-xl border border-amber/40 bg-amber/10 p-3.5 text-sm text-ink sm:mx-5 sm:flex-row sm:items-center sm:justify-between">
            <p className="flex items-start gap-2">
              <WifiOff className="mt-0.5 h-4 w-4 shrink-0 text-amber" aria-hidden="true" />
              API unreachable: showing offline sample data. Promotion and rollback are disabled.
            </p>
            <button
              type="button"
              onClick={() => loadModels()}
              className="inline-flex min-h-[44px] shrink-0 items-center justify-center rounded-lg border border-amber/40 bg-surface px-4 text-sm font-medium text-ink hover:bg-amber/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-amber/30"
            >
              Retry
            </button>
          </div>
        )}

        {modelsStatus === 'loading' ? (
          <p className="flex items-center gap-2 border-t border-border p-5 text-sm text-muted">
            <Spinner /> Loading models…
          </p>
        ) : (
          <>
            {/* md and up: table */}
            <div className="hidden overflow-x-auto md:block">
              <table className="w-full text-left text-sm">
                <caption className="sr-only">Registered model versions, lifecycle status and metrics</caption>
                <thead>
                  <tr className="border-y border-border text-xs text-muted">
                    <th scope="col" className="px-5 py-3 font-medium">Version Tag</th>
                    <th scope="col" className="px-4 py-3 font-medium">Algorithm Name</th>
                    <th scope="col" className="px-4 py-3 font-medium">Status</th>
                    <th scope="col" className="px-4 py-3 font-medium">Created</th>
                    <th scope="col" className="px-4 py-3 font-medium">Data Version</th>
                    <th scope="col" className="px-4 py-3 text-right font-medium">Test Accuracy</th>
                    <th scope="col" className="px-4 py-3 text-right font-medium">Macro-F1</th>
                    <th scope="col" className="px-5 py-3 text-right font-medium">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {models.map((m) => (
                    <tr key={m.version} className={m.status === 'active' ? 'bg-subtle/60' : ''}>
                      <td className="px-5 py-3.5 font-mono text-sm font-bold text-ink">{m.version}</td>
                      <td className="px-4 py-3.5 text-ink">{m.algorithm}</td>
                      <td className="px-4 py-3.5"><ModelStatusBadge status={m.status} /></td>
                      <td className="whitespace-nowrap px-4 py-3.5 text-muted">{fmtDate(m.createdAt)}</td>
                      <td className="px-4 py-3.5 font-mono text-xs text-ink">
                        <span title={m.dataVersion ?? undefined}>{shortHash(m.dataVersion)}</span>
                      </td>
                      <td className="px-4 py-3.5 text-right tabular-nums text-ink">{fmtPct(m.accuracy)}</td>
                      <td className="px-4 py-3.5 text-right tabular-nums text-ink">{fmtScore(m.macroF1)}</td>
                      <td className="px-5 py-3 text-right">
                        <PromoteAction model={m} pending={pending} disabled={actionsDisabled} onPromote={promote} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* phones: one card per model */}
            <ul aria-label="Registered model versions" className="divide-y divide-border border-t border-border md:hidden">
              {models.map((m) => (
                <li key={m.version} className="space-y-3 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-sm font-bold text-ink">{m.version}</span>
                    <ModelStatusBadge status={m.status} />
                  </div>
                  <p className="text-sm text-ink">{m.algorithm}</p>
                  <p className="text-sm text-muted">
                    Accuracy <span className="tabular-nums text-ink">{fmtPct(m.accuracy)}</span> · Macro-F1{' '}
                    <span className="tabular-nums text-ink">{fmtScore(m.macroF1)}</span>
                  </p>
                  <p className="text-xs text-muted">
                    Created {fmtDate(m.createdAt)}
                    <br />
                    Data <span className="break-all font-mono">{m.dataVersion ?? '—'}</span>
                  </p>
                  <PromoteAction model={m} pending={pending} disabled={actionsDisabled} onPromote={promote} />
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <ReportsAudit versionOptions={versionOptions} />
    </div>
  );
}

// ------------------------------------------------------------ report triage
function ReportCard({ report, languages, canReview, onReviewed }) {
  const uid = useId();
  const verifyId = `${uid}-verify`;
  const errorId = `${uid}-error`;

  const prefill = findByCode(languages, report.suggested_language);
  const [query, setQuery] = useState(null); // null = untouched -> show the pre-filled suggestion
  const [pendingAction, setPendingAction] = useState(null); // 'accept' | 'reject' | null
  const [error, setError] = useState('');
  const [langError, setLangError] = useState(false);

  const queryValue = query ?? (prefill ? languageLabel(prefill) : '');
  const reviewable = report.status === 'new';

  const review = async (action) => {
    setError('');
    let body;
    if (action === 'accept') {
      const verified = resolveLanguage(languages, queryValue);
      if (!verified) {
        setLangError(true);
        setError('Select a verified language from the list before accepting.');
        return;
      }
      body = { status: 'accepted', verified_language: verified.code };
    } else {
      body = { status: 'rejected' };
    }
    setLangError(false);
    setPendingAction(action);
    try {
      const res = await apiFetch(`/api/v1/reports/${report.report_id}`, { method: 'PATCH', body, auth: true });
      const updated = res && typeof res.status === 'string' ? res : {};
      onReviewed({ ...report, ...body, ...updated });
    } catch (err) {
      setError(reviewErrorMessage(err));
    } finally {
      setPendingAction(null);
    }
  };

  const busy = pendingAction !== null;

  return (
    <li className="space-y-3 p-4 sm:p-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs font-bold text-ink">#{report.report_id}</span>
        <ReportStatusBadge status={report.status} />
        {report.model_version && (
          <span className="rounded bg-subtle px-2 py-0.5 font-mono text-xs text-muted">Model: {report.model_version}</span>
        )}
        <span className="ml-auto text-xs text-muted">{fmtDate(report.created_at)}</span>
      </div>

      <p className="break-words text-sm text-ink">“{snippet(report.text)}”</p>

      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        <div className="flex flex-wrap items-center gap-2">
          <dt className="text-muted">Predicted</dt>
          <dd className="flex flex-wrap items-center gap-2">
            <LangBadge code={report.predicted_language} languages={languages} tone="bad" />
            {typeof report.confidence === 'number' && (
              <span className="text-xs tabular-nums text-muted">{Math.round(report.confidence * 100)}% confidence</span>
            )}
          </dd>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <dt className="text-muted">Reason</dt>
          <dd>
            <span className="rounded-full bg-chip px-2.5 py-1 text-xs font-medium text-[#2B4442]">
              {REASON_LABELS[report.reason] ?? report.reason}
            </span>
          </dd>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <dt className="text-muted">Suggested</dt>
          <dd>
            {report.suggested_language ? (
              <LangBadge code={report.suggested_language} languages={languages} tone="neutral" />
            ) : (
              <span className="text-muted">—</span>
            )}
          </dd>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <dt className="text-muted">Verified</dt>
          <dd>
            {report.verified_language ? (
              <LangBadge code={report.verified_language} languages={languages} tone="good" />
            ) : (
              <span className="text-muted">—</span>
            )}
          </dd>
        </div>
      </dl>

      {report.comment && <p className="text-sm text-muted">Comment: {report.comment}</p>}

      {reviewable && (
        <div className="rounded-xl bg-subtle p-3.5">
          <label htmlFor={verifyId} className="block text-sm font-medium text-ink">
            Verified language
          </label>
          <div className="mt-2">
            <LanguageCombobox
              id={verifyId}
              languages={languages}
              query={queryValue}
              onQueryChange={(v) => {
                setQuery(v);
                setLangError(false);
              }}
              onPick={(lang) => {
                setQuery(languageLabel(lang));
                setLangError(false);
              }}
              disabled={!canReview || busy}
              invalid={langError}
              describedBy={error ? errorId : undefined}
              placeholder="Search a language to verify"
            />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => review('accept')}
              disabled={!canReview || busy}
              className="inline-flex min-h-[44px] items-center gap-2 rounded-lg bg-teal px-4 text-sm font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-teal"
            >
              {pendingAction === 'accept' ? <Spinner /> : <Check className="h-4 w-4" aria-hidden="true" />}
              Accept
              <span className="sr-only"> report {report.report_id}</span>
            </button>
            <button
              type="button"
              onClick={() => review('reject')}
              disabled={!canReview || busy}
              className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-field bg-surface px-4 text-sm font-medium text-ink transition-colors hover:bg-chip focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-surface"
            >
              {pendingAction === 'reject' ? <Spinner /> : <X className="h-4 w-4" aria-hidden="true" />}
              Reject
              <span className="sr-only"> report {report.report_id}</span>
            </button>
          </div>
          {error && (
            <p id={errorId} role="alert" className="mt-3 flex items-start gap-1.5 text-sm text-red">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              {error}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

function ReportsAudit({ versionOptions }) {
  const token = useAdminToken();
  const { languages } = useLanguages();
  const searchId = useId();
  const statusId = useId();
  const versionId = useId();

  const [statusFilter, setStatusFilter] = useState('all');
  const [versionFilter, setVersionFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const [phase, setPhase] = useState('loading'); // 'loading' | 'live' | 'unauthorized' | 'offline'
  const [offlineReason, setOfflineReason] = useState('');
  const [items, setItems] = useState([]);

  // Load reports whenever a filter, the token or the retry counter changes.
  useEffect(() => {
    const controller = new AbortController();
    setPhase('loading');

    const params = new URLSearchParams();
    if (statusFilter !== 'all') params.set('status', statusFilter);
    if (versionFilter !== 'all') params.set('model_version', versionFilter);
    const qs = params.toString();

    apiFetch(`/api/v1/reports${qs ? `?${qs}` : ''}`, { auth: true, signal: controller.signal })
      .then((data) => {
        if (!isReportList(data)) throw new Error('Unexpected reports payload');
        setItems(data);
        setPhase('live');
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
        if (err?.status === 401) {
          setItems([]);
          setPhase('unauthorized');
          return;
        }
        // Server down, endpoint missing or odd response: show sample rows, filtered the same way.
        setItems(
          SAMPLE_REPORTS.filter(
            (r) =>
              (statusFilter === 'all' || r.status === statusFilter) &&
              (versionFilter === 'all' || r.model_version === versionFilter),
          ),
        );
        setOfflineReason(err?.network ? 'API unreachable' : err?.status ? `API answered HTTP ${err.status}` : 'Unexpected response');
        setPhase('offline');
      });

    return () => controller.abort();
  }, [token, statusFilter, versionFilter, reloadKey]);

  const onReviewed = useCallback((updated) => {
    setItems((prev) => prev.map((r) => (r.report_id === updated.report_id ? updated : r)));
  }, []);

  const q = normalize(query);
  const visible = items.filter((r) => {
    if (!q) return true;
    const names = [r.predicted_language, r.suggested_language, r.verified_language]
      .filter(Boolean)
      .flatMap((c) => [c, findByCode(languages, c)?.name ?? '']);
    return [r.text, r.comment ?? '', r.reason, REASON_LABELS[r.reason] ?? '', r.model_version ?? '', `#${r.report_id}`, ...names].some(
      (h) => normalize(String(h)).includes(q),
    );
  });

  const filtersActive = statusFilter !== 'all' || versionFilter !== 'all';
  const versions = [...new Set([...versionOptions, ...(versionFilter !== 'all' ? [versionFilter] : [])])];
  const selectCls =
    'mt-1 block min-h-[44px] w-full rounded-xl border border-field bg-surface px-3 text-base text-ink outline-none focus:border-teal focus:ring-4 focus:ring-teal/25 sm:text-sm';

  return (
    <section aria-labelledby="reports-title" className="rounded-xl border border-border bg-surface">
      <div className="space-y-3 border-b border-border p-4 sm:p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 id="reports-title" className="text-lg font-bold text-ink">
              Logged User Misclassification Reports (SQLite reports.db)
            </h3>
            <p className="mt-1 text-sm text-muted">
              Real-world failure cases flagged by end users for active learning and dataset retraining.
            </p>
          </div>
          {phase !== 'loading' && phase !== 'unauthorized' && (
            <span className="rounded-full bg-chip px-3 py-1 font-mono text-xs text-teal">
              {items.length} {filtersActive ? 'reports match filters' : 'reports logged'}
            </span>
          )}
        </div>

        {phase === 'offline' && (
          <div className="flex flex-col gap-3 rounded-xl border border-amber/40 bg-amber/10 p-3.5 text-sm text-ink sm:flex-row sm:items-center sm:justify-between">
            <p className="flex items-start gap-2">
              <WifiOff className="mt-0.5 h-4 w-4 shrink-0 text-amber" aria-hidden="true" />
              {offlineReason}: showing offline sample reports. Review actions are disabled.
            </p>
            <button
              type="button"
              onClick={() => setReloadKey((k) => k + 1)}
              className="inline-flex min-h-[44px] shrink-0 items-center justify-center rounded-lg border border-amber/40 bg-surface px-4 text-sm font-medium text-ink hover:bg-amber/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-amber/30"
            >
              Retry
            </button>
          </div>
        )}

        <div className="grid gap-3 sm:grid-cols-[1fr_1fr_2fr]">
          <div>
            <label htmlFor={statusId} className="text-xs font-medium text-muted">Status</label>
            <select id={statusId} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className={selectCls}>
              {STATUS_FILTERS.map((s) => (
                <option key={s} value={s}>{s === 'all' ? 'All statuses' : s[0].toUpperCase() + s.slice(1)}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor={versionId} className="text-xs font-medium text-muted">Model version</label>
            <select id={versionId} value={versionFilter} onChange={(e) => setVersionFilter(e.target.value)} className={selectCls}>
              <option value="all">All versions</option>
              {versions.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor={searchId} className="text-xs font-medium text-muted">Search</label>
            <div className="relative mt-1">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden="true" />
              <input
                id={searchId}
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by language code or keyword..."
                autoComplete="off"
                className="block min-h-[44px] w-full rounded-full border border-transparent bg-subtle pl-10 pr-4 text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:border-teal focus:bg-surface focus:ring-4 focus:ring-teal/25 sm:text-sm"
              />
            </div>
          </div>
        </div>

        {phase === 'live' || phase === 'offline' ? (
          <p aria-live="polite" className="text-xs text-muted">
            Showing {visible.length} of {items.length} reports.
          </p>
        ) : null}
      </div>

      {phase === 'loading' ? (
        <p className="flex items-center gap-2 p-5 text-sm text-muted">
          <Spinner /> Loading reports…
        </p>
      ) : phase === 'unauthorized' ? (
        <div role="alert" className="m-4 flex flex-col gap-3 rounded-xl border border-red/30 bg-red/10 p-4 text-sm text-red sm:m-5 sm:flex-row sm:items-center sm:justify-between">
          <p className="flex items-start gap-2">
            <KeyRound className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            Unauthorized: Invalid or missing X-Admin-Token. Enter a valid admin token to view reports.
          </p>
          <button
            type="button"
            onClick={requestAdminToken}
            className="inline-flex min-h-[44px] shrink-0 items-center justify-center rounded-lg border border-red/40 bg-surface px-4 text-sm font-medium text-red hover:bg-red/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-red/25"
          >
            Enter admin token
          </button>
        </div>
      ) : visible.length === 0 ? (
        <div className="p-8 text-center">
          <p className="text-sm text-ink">
            {items.length === 0 ? 'No reports match these filters.' : `No reports match “${query.trim()}”.`}
          </p>
          {items.length > 0 && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="mt-3 inline-flex min-h-[44px] items-center gap-1.5 rounded-lg px-4 text-sm font-medium text-teal hover:bg-subtle focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
            >
              <X className="h-4 w-4" aria-hidden="true" />
              Clear search
            </button>
          )}
        </div>
      ) : (
        <ul aria-label="Misclassification reports" className="divide-y divide-border">
          {visible.map((r) => (
            <ReportCard
              key={r.report_id}
              report={r}
              languages={languages}
              canReview={phase === 'live'}
              onReviewed={onReviewed}
            />
          ))}
        </ul>
      )}

      {phase === 'live' && languages.length === 0 && (
        <p className="flex items-start gap-2 border-t border-border p-4 text-xs text-muted sm:px-5">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          The language list could not be loaded, so verified languages cannot be picked right now.
        </p>
      )}
    </section>
  );
}

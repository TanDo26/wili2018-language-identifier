import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertCircle, CheckCircle2, Info, Loader2, X } from 'lucide-react';
import LanguageCombobox from './LanguageCombobox.jsx';
import { useLanguages } from '../hooks/useLanguages.js';
import { findByCode, languageLabel, resolveLanguage } from '../lib/languages.js';

const SNIPPET_MAX = 100;
const COMMENT_MAX = 500;
const AUTO_CLOSE_MS = 2000;
const FOCUSABLE = 'button, input, textarea, select, a[href], [tabindex]:not([tabindex="-1"])';

/** The four fixed values accepted by the API for ReportRequest.reason (RP-2). */
export const REPORT_REASONS = [
  { value: 'wrong_language', label: 'Wrong language predicted' },
  { value: 'mixed_languages', label: 'Text contains mixed languages' },
  { value: 'not_a_language', label: 'Not a natural language (code, gibberish, symbols)' },
  { value: 'other', label: 'Other issue' },
];

/**
 * Validation rules for the report form (pure, unit-tested).
 *  - reason is required and must be one of REPORT_REASONS.
 *  - the suggested language is optional. "I don't know" (dontKnow) or an empty box -> null.
 *  - a typed language must be one of the loaded WiLI languages (strict: no free-form codes) and must
 *    differ from the language the model predicted.
 * Returns { valid, errors: { reason?, language? }, suggestedLanguage: string | null }.
 */
export function validateReportForm({ reason, dontKnow, query, languages, predictedCode }) {
  const errors = {};
  if (!REPORT_REASONS.some((r) => r.value === reason)) errors.reason = 'Select what went wrong.';

  let suggestedLanguage = null;
  if (!dontKnow && query.trim() !== '') {
    const match = resolveLanguage(languages, query);
    if (!match) errors.language = 'Choose one of the languages from the list.';
    else if (match.code.toLowerCase() === String(predictedCode).toLowerCase())
      errors.language = "That's the language the model predicted. Choose a different one.";
    else suggestedLanguage = match.code;
  }
  return { valid: Object.keys(errors).length === 0, errors, suggestedLanguage };
}

function snippet(text) {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > SNIPPET_MAX ? `${flat.slice(0, SNIPPET_MAX).trimEnd()}…` : flat;
}

/**
 * Misclassification report dialog (EU-06, FR5; RP-1..RP-4).
 *
 * Props
 * @param {boolean} open  Whether the dialog is shown. Form state resets every time it opens.
 * @param {() => void} onClose
 * @param {object} result  PredictResponse being disputed (needs prediction_id and language_code).
 * @param {string} [text]  The analysed text, shown for context only. It is never sent: the server
 *   looks up text and predicted language from prediction_id.
 * @param {(payload: object) => Promise<unknown>} onSubmit  POSTs the ReportRequest
 *   { prediction_id, reason, suggested_language, comment? }; rejects with an Error whose message is shown.
 */
export default function ReportDialog({ open, ...props }) {
  if (!open || !props.result) return null;
  return createPortal(<ReportDialogInner {...props} />, document.body);
}

function ReportDialogInner({ onClose, text = '', result, onSubmit }) {
  const uid = useId();
  const titleId = `${uid}-title`;
  const langId = `${uid}-lang`;
  const langHintId = `${uid}-lang-hint`;
  const langErrorId = `${uid}-lang-error`;
  const reasonErrorId = `${uid}-reason-error`;
  const commentId = `${uid}-comment`;
  const commentCountId = `${uid}-comment-count`;

  const dialogRef = useRef(null);
  const firstReasonRef = useRef(null);
  const langInputRef = useRef(null);
  const doneRef = useRef(null);

  const { languages, source, loading, unavailable } = useLanguages();

  const [reason, setReason] = useState('');
  const [dontKnow, setDontKnow] = useState(false);
  const [query, setQuery] = useState('');
  const [comment, setComment] = useState('');
  const [attempted, setAttempted] = useState(false);
  const [status, setStatus] = useState('form'); // 'form' | 'submitting' | 'success'
  const [submitError, setSubmitError] = useState('');

  const suggestions = (result.top_candidates ?? [])
    .filter((c) => c.code !== result.language_code)
    .map((c) => findByCode(languages, c.code))
    .filter(Boolean);

  // Focus in on open, restore on close; lock page scroll behind the dialog.
  useEffect(() => {
    const trigger = document.activeElement;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialogRef.current?.focus();
    return () => {
      document.body.style.overflow = prevOverflow;
      if (trigger instanceof HTMLElement && document.contains(trigger)) trigger.focus();
    };
  }, []);

  // Esc closes; Tab / Shift+Tab stay inside the dialog (also if focus was lost to <body>).
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (e.defaultPrevented) return; // an open suggestion list already used this Esc
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== 'Tab') return;
      const root = dialogRef.current;
      if (!root) return;
      const items = [...root.querySelectorAll(FOCUSABLE)].filter((el) => !el.disabled);
      if (items.length === 0) {
        e.preventDefault();
        root.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      const outside = !root.contains(active) || active === root;
      if (e.shiftKey && (active === first || outside)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || outside)) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  // Success: move focus to "Done" and close after 2 s.
  useEffect(() => {
    if (status !== 'success') return undefined;
    doneRef.current?.focus();
    const t = setTimeout(onClose, AUTO_CLOSE_MS);
    return () => clearTimeout(t);
  }, [status, onClose]);

  const check = validateReportForm({
    reason,
    dontKnow,
    query,
    languages,
    predictedCode: result.language_code,
  });
  const showReasonError = attempted && check.errors.reason;
  const showLangError = attempted && check.errors.language;
  const langDisabled = dontKnow || unavailable;

  const toggleDontKnow = (checked) => {
    setDontKnow(checked);
    if (checked) setQuery(''); // "I don't know" clears the picked language
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (status === 'submitting') return;
    setAttempted(true);
    if (!check.valid) {
      if (check.errors.reason) firstReasonRef.current?.focus();
      else langInputRef.current?.focus();
      return;
    }
    const payload = {
      prediction_id: result.prediction_id,
      reason,
      suggested_language: check.suggestedLanguage,
    };
    if (comment.trim()) payload.comment = comment.trim();

    setStatus('submitting');
    setSubmitError('');
    try {
      await onSubmit(payload);
      setStatus('success');
    } catch (err) {
      setStatus('form');
      setSubmitError(err?.message || 'Could not send your report. Try again.');
    }
  };

  const isSubmitting = status === 'submitting';
  const langDescribedBy = [langHintId, showLangError ? langErrorId : null].filter(Boolean).join(' ');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4 backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="max-h-[calc(100dvh-2rem)] w-full max-w-lg overflow-y-auto rounded-2xl border border-border bg-surface p-6 shadow-xl outline-none"
      >
        <div className="flex items-start justify-between gap-3">
          <h2 id={titleId} className="text-xl font-bold text-ink">
            Report incorrect prediction
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-2 -mt-2 inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-subtle hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        {status === 'success' ? (
          <div role="status" className="flex flex-col items-center gap-3 py-8 text-center">
            <CheckCircle2 className="h-12 w-12 text-teal" aria-hidden="true" />
            <p className="text-base font-medium text-ink">Thank you! Your feedback has been recorded.</p>
            <button
              ref={doneRef}
              type="button"
              onClick={onClose}
              className="mt-2 inline-flex min-h-[44px] items-center justify-center rounded-xl bg-teal px-5 font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
            >
              Done
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} noValidate className="mt-4 space-y-5">
            {/* Read-only summary of what is being reported */}
            <div className="rounded-xl border border-border bg-subtle p-3 text-sm text-ink">
              <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="text-muted">Predicted:</span>
                <span className="font-medium">{result.language_name}</span>
                <span className="rounded-md bg-chip px-2 py-0.5 font-mono text-xs text-teal">
                  {result.language_code}
                </span>
                {result.model_version && (
                  <span className="rounded bg-surface px-2 py-0.5 font-mono text-xs text-muted">
                    Model: {result.model_version}
                  </span>
                )}
              </p>
              {text && <p className="mt-2 break-words text-muted">“{snippet(text)}”</p>}
            </div>

            {/* RP-2: mandatory reason */}
            <fieldset aria-describedby={showReasonError ? reasonErrorId : undefined}>
              <legend className="text-sm font-medium text-ink">
                What went wrong?
                <span className="text-red" aria-hidden="true">
                  {' '}
                  *
                </span>
              </legend>
              <div className="mt-2 space-y-2">
                {REPORT_REASONS.map((r, i) => (
                  <label key={r.value} className="block cursor-pointer">
                    <input
                      ref={i === 0 ? firstReasonRef : undefined}
                      type="radio"
                      name={`${uid}-reason`}
                      value={r.value}
                      checked={reason === r.value}
                      onChange={() => setReason(r.value)}
                      aria-required="true"
                      className="peer sr-only"
                    />
                    <span
                      className={`flex min-h-[44px] items-center rounded-xl border bg-surface px-3.5 py-2 text-sm text-ink transition-colors hover:bg-subtle peer-checked:border-teal peer-checked:bg-chip peer-checked:font-medium peer-checked:text-teal peer-focus-visible:ring-4 peer-focus-visible:ring-teal/30 ${
                        showReasonError ? 'border-red' : 'border-field'
                      }`}
                    >
                      {r.label}
                    </span>
                  </label>
                ))}
              </div>
              {showReasonError && (
                <p id={reasonErrorId} className="mt-1.5 flex items-start gap-1.5 text-sm text-red">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  {check.errors.reason}
                </p>
              )}
            </fieldset>

            {/* RP-1 / RP-3: optional correct language */}
            <div>
              <label htmlFor={langId} className="block text-sm font-medium text-ink">
                Correct language (optional)
              </label>

              <label className="mt-2 flex min-h-[44px] cursor-pointer items-center gap-3 text-sm text-ink">
                <input
                  type="checkbox"
                  checked={dontKnow}
                  onChange={(e) => toggleDontKnow(e.target.checked)}
                  className="h-5 w-5 rounded border-field text-teal focus:ring-teal"
                />
                I don&apos;t know the correct language
              </label>

              {suggestions.length > 0 && (
                <div
                  role="group"
                  aria-label="Other candidates from the prediction"
                  className="mb-2 flex flex-wrap gap-2"
                >
                  {suggestions.map((l) => {
                    const on = resolveLanguage(languages, query)?.code === l.code;
                    return (
                      <button
                        key={l.code}
                        type="button"
                        aria-pressed={on}
                        disabled={dontKnow}
                        onClick={() => setQuery(languageLabel(l))}
                        className={`inline-flex min-h-[44px] items-center gap-2 rounded-xl border px-3.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 ${
                          on
                            ? 'border-teal bg-chip font-medium text-teal'
                            : 'border-field bg-surface text-ink hover:bg-subtle'
                        }`}
                      >
                        {l.name}{' '}
                        <span className="font-mono text-xs text-muted">{l.code}</span>
                      </button>
                    );
                  })}
                </div>
              )}

              <LanguageCombobox
                id={langId}
                inputRef={langInputRef}
                languages={languages}
                query={query}
                onQueryChange={setQuery}
                onPick={(lang) => setQuery(languageLabel(lang))}
                disabled={langDisabled}
                invalid={Boolean(showLangError)}
                describedBy={langDescribedBy}
              />

              <p id={langHintId} className="mt-1.5 flex items-start gap-1.5 text-xs text-muted">
                {unavailable ? (
                  <>
                    <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    The language list is unavailable offline. You can still submit without a suggestion.
                  </>
                ) : dontKnow ? (
                  'No language will be suggested.'
                ) : loading ? (
                  'Loading languages…'
                ) : source === 'fallback' ? (
                  'Offline: using the built-in list of 235 languages.'
                ) : (
                  'Optional. Pick one of the 235 supported languages.'
                )}
              </p>

              {showLangError && (
                <p id={langErrorId} className="mt-1.5 flex items-start gap-1.5 text-sm text-red">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  {check.errors.language}
                </p>
              )}
            </div>

            {/* RP-3: comment */}
            <div>
              <label htmlFor={commentId} className="block text-sm font-medium text-muted">
                Comment (optional)
              </label>
              <textarea
                id={commentId}
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                maxLength={COMMENT_MAX}
                aria-describedby={commentCountId}
                placeholder="e.g., Contains mixed dialects, transliteration, or slang..."
                className="mt-2 block min-h-[80px] w-full resize-y rounded-xl border border-field bg-surface p-3 font-sans text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:border-teal focus:ring-4 focus:ring-teal/25 sm:text-sm"
              />
              <p id={commentCountId} className="mt-1 text-right font-mono text-xs text-muted">
                {comment.length} / {COMMENT_MAX}
              </p>
            </div>

            {submitError && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-xl border border-red/30 bg-red/10 p-3 text-sm text-red"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <p>{submitError}</p>
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={onClose}
                className="min-h-[44px] rounded-xl px-4 text-muted transition-colors hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                aria-busy={isSubmitting}
                className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-xl bg-teal px-5 font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-wait"
              >
                {isSubmitting && (
                  <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                )}
                {isSubmitting ? 'Submitting…' : 'Submit report'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

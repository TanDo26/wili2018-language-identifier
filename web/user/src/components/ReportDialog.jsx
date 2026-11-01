import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';
import { LANGUAGES } from '../data/languages.js';

const SNIPPET_MAX = 100;
const NOTES_MAX = 500;
const AUTO_CLOSE_MS = 2000;
const FOCUSABLE = 'button, input, textarea, select, a[href], [tabindex]:not([tabindex="-1"])';

// ---------------------------------------------------------------- helpers

/** Lowercase + strip diacritics so "norwegian bokmal" finds "Norwegian Bokmål". */
const normalize = (s) =>
  s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();

function snippet(text) {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > SNIPPET_MAX ? `${flat.slice(0, SNIPPET_MAX).trimEnd()}…` : flat;
}

/** Up to 6 languages whose name or ISO code matches, best matches first. */
function searchLanguages(query) {
  const q = normalize(query);
  if (!q) return [];
  const scored = [];
  for (const lang of LANGUAGES) {
    const name = normalize(lang.name);
    let score = -1;
    if (lang.code === q) score = 0;
    else if (name === q) score = 1;
    else if (name.startsWith(q)) score = 2;
    else if (lang.code.startsWith(q)) score = 3;
    else if (name.includes(q)) score = 4;
    if (score >= 0) scored.push({ lang, score });
  }
  scored.sort((a, b) => a.score - b.score || a.lang.name.localeCompare(b.lang.name));
  return scored.slice(0, 6).map((s) => s.lang);
}

/**
 * Turns whatever is typed into { code, name } or null:
 * "English (eng)", "eng", "English", or any other 3-letter code (the starter list is not exhaustive).
 */
function resolveQuery(query) {
  const raw = query.trim();
  if (!raw) return null;
  const inParens = raw.match(/\(([a-z]{3})\)\s*$/i);
  const q = normalize(inParens ? inParens[1] : raw);
  const hit = LANGUAGES.find((l) => l.code === q || normalize(l.name) === q);
  if (hit) return hit;
  if (/^[a-z]{3}$/.test(q)) return { code: q, name: q };
  return null;
}

// -------------------------------------------------------------- component

/**
 * Misclassification report dialog (EU-06, FR5).
 *
 * Props
 * @param {boolean} open  Whether the dialog is shown. Form state resets every time it opens.
 * @param {() => void} onClose  Called on Esc, backdrop click, close/cancel/done, and after the success auto-close.
 * @param {string} text  The text that was analysed (not whatever is in the input box now).
 * @param {object} result  PredictResponse the user is disputing.
 * @param {(payload: object) => Promise<unknown>} onSubmit  Sends the ReportRequest to POST /api/v1/report;
 *   must resolve on 201 and reject with an Error whose message is shown to the user.
 */
export default function ReportDialog({ open, ...props }) {
  if (!open || !props.result) return null;
  return createPortal(<ReportDialogInner {...props} />, document.body);
}

function ReportDialogInner({ onClose, text, result, onSubmit }) {
  const uid = useId();
  const titleId = `${uid}-title`;
  const inputId = `${uid}-lang`;
  const listId = `${uid}-list`;
  const errorId = `${uid}-lang-error`;
  const hintId = `${uid}-lang-hint`;
  const notesId = `${uid}-notes`;

  const dialogRef = useRef(null);
  const inputRef = useRef(null);
  const doneRef = useRef(null);

  const [selected, setSelected] = useState(null); // { code, name } | null
  const [query, setQuery] = useState('');
  const [listOpen, setListOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [notes, setNotes] = useState('');
  const [attempted, setAttempted] = useState(false);
  const [status, setStatus] = useState('form'); // 'form' | 'submitting' | 'success'
  const [submitError, setSubmitError] = useState('');

  const suggestions = (result.top_candidates ?? []).filter((c) => c.code !== result.language_code);
  const matches = listOpen ? searchLanguages(query) : [];
  const showList = listOpen && matches.length > 0;
  const noMatchHint =
    query.trim().length >= 2 && !selected && searchLanguages(query).length === 0 && !resolveQuery(query);

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
        if (e.defaultPrevented) return; // the open suggestion list already used this Esc
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

  // Success: move focus to "Done" (the submit button is gone) and close after 2 s.
  useEffect(() => {
    if (status !== 'success') return undefined;
    doneRef.current?.focus();
    const t = setTimeout(onClose, AUTO_CLOSE_MS);
    return () => clearTimeout(t);
  }, [status, onClose]);

  // ---- validation (errors appear after the first submit attempt, then update live)
  const choice = selected ?? resolveQuery(query);
  let langError = '';
  if (!query.trim() && !selected) langError = 'Select the correct language.';
  else if (!choice) langError = 'Choose a language from the list or enter a 3-letter ISO 639-3 code.';
  else if (choice.code === result.language_code)
    langError = "That's the language the model predicted. Choose a different one.";
  const showLangError = attempted && langError !== '';

  const pick = (lang) => {
    setSelected(lang);
    setQuery(`${lang.name} (${lang.code})`);
    setListOpen(false);
    setActiveIdx(-1);
  };

  const onQueryChange = (e) => {
    setQuery(e.target.value);
    setSelected(null);
    setListOpen(true);
    setActiveIdx(-1);
  };

  const onInputKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!listOpen) setListOpen(true);
      setActiveIdx((i) => (matches.length ? (i + 1) % matches.length : -1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((i) => (matches.length ? (i <= 0 ? matches.length - 1 : i - 1) : -1));
    } else if (e.key === 'Enter' && showList && activeIdx >= 0) {
      e.preventDefault();
      pick(matches[activeIdx]);
    } else if (e.key === 'Escape' && listOpen) {
      e.preventDefault(); // closes the list only, not the dialog
      setListOpen(false);
      setActiveIdx(-1);
    }
  };

  const onInputBlur = () => {
    setListOpen(false);
    setActiveIdx(-1);
    if (!selected) {
      const hit = resolveQuery(query);
      if (hit) setSelected(hit);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (status === 'submitting') return;
    setAttempted(true);
    if (langError) {
      inputRef.current?.focus();
      return;
    }
    const payload = {
      text,
      predicted_language: result.language_code,
      correct_language: choice.code,
    };
    if (notes.trim()) payload.user_feedback = notes.trim();

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
  const describedBy = [hintId, showLangError ? errorId : null].filter(Boolean).join(' ');

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
              </p>
              <p className="mt-2 break-words text-muted">“{snippet(text)}”</p>
            </div>

            {/* Correct language: quick chips + search */}
            <div>
              <label htmlFor={inputId} className="block text-sm font-medium text-ink">
                What is the correct language?
                <span className="text-red" aria-hidden="true">
                  {' '}
                  *
                </span>
              </label>

              {suggestions.length > 0 && (
                <div
                  role="group"
                  aria-label="Other candidates from the prediction"
                  className="mt-2 flex flex-wrap gap-2"
                >
                  {suggestions.map((c) => {
                    const on = selected?.code === c.code;
                    return (
                      <button
                        key={c.code}
                        type="button"
                        aria-pressed={on}
                        onClick={() => pick({ code: c.code, name: c.name })}
                        className={`inline-flex min-h-[44px] items-center gap-2 rounded-xl border px-3.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 ${
                          on
                            ? 'border-teal bg-chip font-medium text-teal'
                            : 'border-field bg-surface text-ink hover:bg-subtle'
                        }`}
                      >
                        {c.name}{' '}
                        <span className="font-mono text-xs text-muted">{c.code}</span>
                      </button>
                    );
                  })}
                </div>
              )}

              <input
                id={inputId}
                ref={inputRef}
                type="text"
                role="combobox"
                aria-expanded={showList}
                aria-controls={listId}
                aria-autocomplete="list"
                aria-activedescendant={showList && activeIdx >= 0 ? `${listId}-${activeIdx}` : undefined}
                aria-required="true"
                aria-invalid={showLangError}
                aria-describedby={describedBy}
                autoComplete="off"
                autoCapitalize="off"
                spellCheck={false}
                value={query}
                onChange={onQueryChange}
                onKeyDown={onInputKeyDown}
                onBlur={onInputBlur}
                placeholder="Search by name or ISO code, e.g. English or eng"
                className={`mt-2 block min-h-[44px] w-full rounded-xl border bg-surface px-3 text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:ring-4 sm:text-sm ${
                  showLangError
                    ? 'border-red focus:ring-red/25'
                    : 'border-field focus:border-teal focus:ring-teal/25'
                }`}
              />

              <ul
                id={listId}
                role="listbox"
                aria-label="Matching languages"
                hidden={!showList}
                className="mt-1 overflow-hidden rounded-xl border border-border bg-surface"
              >
                {matches.map((lang, i) => (
                  <li
                    key={lang.code}
                    id={`${listId}-${i}`}
                    role="option"
                    aria-selected={i === activeIdx}
                    onMouseDown={(e) => e.preventDefault()} // keep focus in the input
                    onClick={() => pick(lang)}
                    onMouseEnter={() => setActiveIdx(i)}
                    className={`flex min-h-[44px] cursor-pointer items-center justify-between px-3 text-sm text-ink ${
                      i === activeIdx ? 'bg-subtle' : ''
                    }`}
                  >
                    <span>{lang.name}</span>{' '}
                    <span className="font-mono text-xs text-muted">{lang.code}</span>
                  </li>
                ))}
              </ul>

              <p id={hintId} className="mt-1.5 text-xs text-muted">
                {noMatchHint
                  ? 'No match in the list. You can still enter a 3-letter ISO 639-3 code.'
                  : 'Pick a suggestion above or search any language.'}
              </p>

              {showLangError && (
                <p id={errorId} className="mt-1.5 flex items-start gap-1.5 text-sm text-red">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  {langError}
                </p>
              )}
            </div>

            {/* Optional notes */}
            <div>
              <label htmlFor={notesId} className="block text-sm font-medium text-muted">
                Additional notes (optional)
              </label>
              <textarea
                id={notesId}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                maxLength={NOTES_MAX}
                placeholder="e.g., Contains mixed dialects, transliteration, or slang..."
                className="mt-2 block min-h-[80px] w-full resize-y rounded-xl border border-field bg-surface p-3 font-sans text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:border-teal focus:ring-4 focus:ring-teal/25 sm:text-sm"
              />
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

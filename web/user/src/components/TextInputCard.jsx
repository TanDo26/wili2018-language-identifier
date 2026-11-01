import { useId, useRef, useState } from 'react';
import { AlertCircle, AlertTriangle, Loader2 } from 'lucide-react';

export const MIN_CHARS = 20;
export const MAX_CHARS = 5000;

const fmt = (n) => n.toLocaleString('en-US');

/**
 * Validation status of the current text.
 *  empty      – nothing typed yet (no message, submit disabled)
 *  emptyError – field was touched and left empty (EU-04, red text)
 *  tooShort   – 1..19 characters (EU-03, amber warning, submit disabled)
 *  valid      – 20..5,000 characters
 *  tooLong    – more than 5,000 characters (EU-04, red border + message, submit disabled)
 */
function getStatus(value, touched) {
  if (value.length > MAX_CHARS) return 'tooLong';
  if (value.trim().length === 0) return touched ? 'emptyError' : 'empty';
  if (value.length < MIN_CHARS) return 'tooShort';
  return 'valid';
}

/**
 * Text input card for EU-01 / EU-03 / EU-04.
 *
 * Props
 * @param {string}   value      Controlled textarea value.
 * @param {(next: string) => void} onChange  Called with the new text on every edit and on Clear.
 * @param {() => void} onSubmit  Called when the user presses "Identify language" and the text is valid.
 * @param {boolean}  [isLoading=false]  Shows a spinner and blocks submit/clear while a request is in flight.
 */
export default function TextInputCard({ value, onChange, onSubmit, isLoading = false }) {
  const [touched, setTouched] = useState(false);
  const textareaRef = useRef(null);
  const uid = useId();
  const counterId = `${uid}-counter`;
  const messageId = `${uid}-message`;

  const status = getStatus(value, touched);
  const isError = status === 'emptyError' || status === 'tooLong';
  const canSubmit = status === 'valid' && !isLoading;
  const hasMessage = status === 'emptyError' || status === 'tooShort' || status === 'tooLong';

  const borderClass = isError
    ? 'border-red focus:ring-red/25'
    : 'border-field focus:border-teal focus:ring-teal/25';

  const handleSubmit = (e) => {
    e.preventDefault();
    if (canSubmit) onSubmit();
  };

  const handleClear = () => {
    onChange('');
    setTouched(false);
    textareaRef.current?.focus();
  };

  return (
    <form
      onSubmit={handleSubmit}
      noValidate
      className="rounded-2xl border border-border bg-surface p-4 sm:p-5"
    >
      <label htmlFor={`${uid}-text`} className="sr-only">
        Text to identify
      </label>
      <textarea
        id={`${uid}-text`}
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={() => setTouched(true)}
        placeholder="Paste text or write something in any language..."
        aria-invalid={isError}
        aria-describedby={hasMessage ? `${counterId} ${messageId}` : counterId}
        className={`block min-h-[140px] w-full resize-y rounded-xl border bg-surface p-3.5 font-sans text-base leading-relaxed text-ink outline-none transition-colors placeholder:text-muted/80 focus:ring-4 ${borderClass}`}
      />

      <p
        id={counterId}
        className={`mt-2 font-mono text-[13px] ${status === 'tooLong' ? 'font-medium text-red' : 'text-muted'}`}
      >
        {fmt(value.length)} / {fmt(MAX_CHARS)} characters
      </p>

      <div id={hasMessage ? messageId : undefined} aria-live="polite">
        {status === 'emptyError' && (
          <p className="mt-2 flex items-start gap-1.5 text-sm text-red">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            Enter some text to identify its language.
          </p>
        )}

        {status === 'tooShort' && (
          <p className="mt-2 flex items-start gap-2 rounded-xl border border-amber/40 bg-amber/10 px-3 py-2 text-sm text-ink">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber" aria-hidden="true" />
            Warning: Text is under {MIN_CHARS} characters (minimum required for reliable
            classification)
          </p>
        )}

        {status === 'tooLong' && (
          <p className="mt-2 flex items-start gap-1.5 text-sm text-red">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            Text exceeds maximum limit of {fmt(MAX_CHARS)} characters
          </p>
        )}
      </div>

      <div className="mt-4 flex items-center gap-2">
        <button
          type="submit"
          disabled={!canSubmit}
          aria-busy={isLoading}
          className={`inline-flex min-h-[52px] flex-1 items-center justify-center gap-2 rounded-xl bg-teal px-5 text-base font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 ${
            isLoading ? 'cursor-wait' : 'disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-teal'
          }`}
        >
          {isLoading && <Loader2 className="h-5 w-5 animate-spin motion-reduce:animate-none" aria-hidden="true" />}
          {isLoading ? 'Identifying…' : 'Identify language'}
        </button>

        <button
          type="button"
          onClick={handleClear}
          disabled={isLoading || (value.length === 0 && !touched)}
          className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-xl px-4 text-base font-medium text-muted transition-colors hover:bg-subtle hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
        >
          Clear
        </button>
      </div>
    </form>
  );
}

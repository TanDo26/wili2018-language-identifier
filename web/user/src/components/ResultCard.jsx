import { useEffect, useId, useRef } from 'react';
import { AlertTriangle } from 'lucide-react';

/** 0..1 -> whole percent label. Tiny non-zero values read "<1%" instead of a misleading "0%". */
function formatPercent(confidence) {
  const pct = confidence * 100;
  if (pct > 0 && pct < 0.5) return '<1%';
  return `${Math.round(pct)}%`;
}

/** Bar width in %, clamped so bad data can never overflow the track. */
function barWidth(confidence) {
  return `${Math.min(100, Math.max(0, confidence * 100))}%`;
}

/**
 * Finds quoted n-grams such as 'thi' or "ngư" inside an explanation sentence.
 * An opening quote must follow the start/space/punctuation and a closing quote must
 * precede the end/space/punctuation, so apostrophes in words ("Vietnamese's") are ignored.
 * (Lookahead only: lookbehind is not supported by older Safari versions.)
 */
const NGRAM_RE = /(^|[\s(,:;])(['"‘“])([^'"‘’“”]{1,12}?)(['"’”])(?=$|[\s),.:;!?])/g;

/** Splits text into plain and highlighted (n-gram) parts. */
function splitExplanation(text) {
  const parts = [];
  let last = 0;
  let m;
  NGRAM_RE.lastIndex = 0;
  while ((m = NGRAM_RE.exec(text)) !== null) {
    const [, lead, open, gram] = m;
    const gramStart = m.index + lead.length + open.length;
    parts.push({ text: text.slice(last, gramStart), highlight: false });
    parts.push({ text: gram, highlight: true });
    last = gramStart + gram.length; // closing quote stays in the next plain part
  }
  if (last < text.length) parts.push({ text: text.slice(last), highlight: false });
  return parts;
}

function ExplanationText({ text }) {
  return splitExplanation(text).map((part, i) =>
    part.highlight ? (
      <strong key={i} className="font-bold text-teal">
        {part.text}
      </strong>
    ) : (
      <span key={i}>{part.text}</span>
    ),
  );
}

/**
 * Prediction result (EU-02) with low-confidence alert (EU-05, FR3) and model explanation.
 *
 * Props
 * @param {object|null} result  PredictResponse from openapi.yaml:
 *   { language_code, language_name, confidence, is_unsure, explanation, top_candidates }.
 *   Renders nothing when null/undefined.
 * @param {() => void} onOpenReport  Called when the user presses "Report incorrect result" (EU-06 trigger).
 */
export default function ResultCard({ result, onOpenReport }) {
  const uid = useId();
  const headingId = `${uid}-lang`;
  const sectionRef = useRef(null);

  // On small screens the result can land below the fold: bring it into view for each new result.
  useEffect(() => {
    if (!result) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    sectionRef.current?.scrollIntoView?.({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' });
  }, [result]);

  if (!result) return null;

  const { language_code, language_name, confidence, is_unsure, top_candidates = [] } = result;

  const explanation = (result.explanation ?? []).filter(
    (line) => typeof line === 'string' && line.trim().length > 0,
  );

  // top_candidates includes the best match; keep original ranks (#2, #3) for the rest.
  const topIndex = top_candidates.findIndex((c) => c.code === language_code);
  const others = top_candidates
    .map((c, i) => ({ ...c, rank: i + 1 }))
    .filter((_, i) => i !== topIndex);

  return (
    <section
      ref={sectionRef}
      aria-labelledby={headingId}
      className="rounded-2xl border border-border bg-surface p-5 sm:p-6"
    >
      <p className="text-[13px] font-medium text-muted">Best match</p>
      <h2 id={headingId} className="mt-1 break-words text-[32px] font-bold leading-tight text-teal">
        {language_name}
      </h2>

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="rounded-md bg-chip px-2.5 py-1 font-mono text-xs text-teal">
          {language_code}
        </span>
        <span className="text-sm font-medium text-ink">{formatPercent(confidence)} confidence</span>
      </div>

      <div
        role="progressbar"
        aria-label={`Confidence for ${language_name}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(confidence * 100)}
        className="mt-3 h-2.5 w-full overflow-hidden rounded-full bg-subtle"
      >
        <div className="h-full rounded-full bg-teal" style={{ width: barWidth(confidence) }} />
      </div>

      {/* EU-05 / FR3: low-confidence alert */}
      {is_unsure === true && (
        <div
          role="status"
          className="mt-4 flex items-start gap-2.5 rounded-xl border border-amber/40 bg-amber/10 p-3 text-sm text-ink"
        >
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber" aria-hidden="true" />
          <div className="min-w-0">
            <span className="mb-1 inline-block rounded-full bg-amber/20 px-2 py-0.5 text-xs font-medium text-amber">
              Low confidence
            </span>
            <p>
              Low confidence prediction. The text may be too short, out-of-distribution, or contain
              mixed languages.
            </p>
          </div>
        </div>
      )}

      {/* Model explanation: n-gram triggers returned by the model */}
      {explanation.length > 0 && (
        <>
          <h3 className="mb-2 mt-4 text-[13px] font-medium text-muted">Model explanation</h3>
          <ul className="space-y-2">
            {explanation.map((line, i) => (
              <li
                key={i}
                className="break-words rounded-lg border border-border bg-subtle p-2.5 font-mono text-xs leading-relaxed text-ink"
              >
                <ExplanationText text={line} />
              </li>
            ))}
          </ul>
        </>
      )}

      {others.length > 0 && (
        <>
          <h3 className="mb-2 mt-4 text-[13px] font-medium text-muted">Other candidates</h3>
          <ul className="space-y-2">
            {others.map((c) => (
              <li
                key={`${c.code}-${c.rank}`}
                className="flex items-center justify-between gap-3 rounded-xl bg-subtle p-3 text-sm text-ink"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <span className="shrink-0 rounded bg-surface px-1.5 py-0.5 font-mono text-xs text-muted">
                    #{c.rank}
                  </span>
                  <span className="truncate">{c.name}</span>
                  <span className="shrink-0 font-mono text-xs text-muted">{c.code}</span>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <div
                    role="progressbar"
                    aria-label={`Confidence for ${c.name}`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(c.confidence * 100)}
                    className="h-1.5 w-14 overflow-hidden rounded-full bg-border sm:w-24"
                  >
                    <div className="h-full rounded-full bg-teal" style={{ width: barWidth(c.confidence) }} />
                  </div>
                  <span className="w-9 text-right tabular-nums">{formatPercent(c.confidence)}</span>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      <div className="mt-4 border-t border-border pt-1">
        <button
          type="button"
          onClick={onOpenReport}
          className="min-h-[44px] min-w-[44px] rounded-md text-sm text-muted underline underline-offset-4 transition-colors hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
        >
          Report incorrect result
        </button>
      </div>
    </section>
  );
}

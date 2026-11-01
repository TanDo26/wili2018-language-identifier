import { useCallback, useState } from 'react';
import { AlertCircle, Loader2, RotateCw } from 'lucide-react';
import TextInputCard from './components/TextInputCard.jsx';
import ResultCard from './components/ResultCard.jsx';
import ReportDialog from './components/ReportDialog.jsx';

// '' in dev (Vite proxies /api, see vite.config.js); set VITE_API_BASE_URL in production.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

/** Error with a user-facing message and a flag telling the UI whether "Retry" makes sense. */
class ApiError extends Error {
  constructor(message, { status = null, retryable = false } = {}) {
    super(message);
    this.status = status;
    this.retryable = retryable;
  }
}

/** Remedies per HTTP status for POST /api/v1/predict (see openapi.yaml). */
function predictErrorForStatus(status) {
  switch (status) {
    case 400:
    case 422:
      return new ApiError('Input text must be between 20 and 5,000 characters.', { status });
    case 413:
      return new ApiError('Payload too large: Text exceeds maximum limit of 5,000 characters.', { status });
    case 429:
      return new ApiError(
        'Rate limit exceeded: You have made too many requests. Please wait a moment.',
        { status },
      );
    case 503:
      return new ApiError(
        'Service Unavailable: ML model engine is currently loading or uninitialized.',
        { status, retryable: true },
      );
    default:
      return new ApiError('Something went wrong. Try again.', { status });
  }
}

/** Messages for POST /api/v1/report, shown inside the report dialog. */
function reportErrorForStatus(status) {
  switch (status) {
    case 400:
    case 422:
      return new ApiError('The report was rejected. Check the correct language and try again.', { status });
    case 429:
      return new ApiError('Too many requests. Wait a moment and try again.', { status });
    case 503:
      return new ApiError('The service is unavailable right now. Try again shortly.', { status });
    default:
      return new ApiError('Could not send your report. Try again.', { status });
  }
}

/** POST JSON; a network failure becomes a retryable ApiError. */
async function postJson(path, body) {
  try {
    return await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Can't reach the server. Check your connection and try again.", {
      retryable: true,
    });
  }
}

async function predictLanguage(text) {
  const res = await postJson('/api/v1/predict', { text });
  if (!res.ok) throw predictErrorForStatus(res.status);
  return res.json(); // PredictResponse
}

/** payload = ReportRequest; resolves with { message, report_id } on 201. */
async function submitReport(payload) {
  const res = await postJson('/api/v1/report', payload);
  if (!res.ok) throw reportErrorForStatus(res.status);
  return res.json();
}

export default function App() {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null); // PredictResponse
  // The text the current result belongs to. Reports must use this, not the live input,
  // because the user may keep editing the box after a prediction.
  const [analyzedText, setAnalyzedText] = useState('');
  const [apiError, setApiError] = useState(null); // ApiError | null
  const [isReportOpen, setIsReportOpen] = useState(false);

  const handleChange = useCallback((next) => {
    setText(next);
    setApiError(null);
  }, []);

  // Also used by "Retry". The error stays visible while retrying so the button keeps focus;
  // it is replaced on success or on the next failure.
  const handleSubmit = useCallback(async () => {
    setIsLoading(true);
    setIsReportOpen(false);
    try {
      const data = await predictLanguage(text);
      setResult(data);
      setAnalyzedText(text);
      setApiError(null);
    } catch (err) {
      setResult(null);
      setApiError(err instanceof ApiError ? err : predictErrorForStatus(null));
    } finally {
      setIsLoading(false);
    }
  }, [text]);

  const closeReport = useCallback(() => setIsReportOpen(false), []);

  return (
    <main className="mx-auto flex w-full max-w-xl flex-col gap-4 px-4 pb-12 pt-8 sm:max-w-2xl sm:px-6 sm:pt-14">
      <header className="mb-2 sm:mb-4">
        <span className="inline-flex items-center rounded-full bg-chip px-3 py-1 font-mono text-xs font-medium text-teal">
          WiLI-2018
        </span>
        <h1 className="mt-4 text-[30px] font-bold leading-[1.15]">Which language is this?</h1>
        <p className="mt-2 text-[15px] text-muted">Detects among 235 languages.</p>
      </header>

      <TextInputCard
        value={text}
        onChange={handleChange}
        onSubmit={handleSubmit}
        isLoading={isLoading}
      />

      {apiError && (
        <div
          role="alert"
          className="flex flex-col gap-3 rounded-xl border border-red/30 bg-red/10 p-3.5 text-[15px] text-red sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
            <p>{apiError.message}</p>
          </div>
          {apiError.retryable && (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isLoading}
              className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center gap-2 rounded-xl border border-red/40 bg-surface px-4 text-sm font-medium text-red transition-colors hover:bg-red/10 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-red/25 disabled:cursor-wait disabled:opacity-70"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden="true" />
              ) : (
                <RotateCw className="h-4 w-4" aria-hidden="true" />
              )}
              {isLoading ? 'Retrying…' : 'Retry'}
            </button>
          )}
        </div>
      )}

      <ResultCard result={result} onOpenReport={() => setIsReportOpen(true)} />

      <ReportDialog
        open={isReportOpen}
        onClose={closeReport}
        text={analyzedText}
        result={result}
        onSubmit={submitReport}
      />
    </main>
  );
}

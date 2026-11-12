import { useEffect, useId, useRef, useState } from 'react';
import { KeyRound, ShieldAlert, ShieldCheck } from 'lucide-react';
import {
  clearAdminToken,
  getTokenSource,
  onUnauthorized,
  setAdminToken,
  useAdminToken,
  useTokenRejected,
} from '../api/client.js';

/**
 * Compact "Admin Token" pill for the header (MV-5). Opens a small form to enter, change or clear the
 * X-Admin-Token used by protected API calls, and opens itself when the API answers 401.
 */
export default function AdminTokenControl() {
  const uid = useId();
  const panelId = `${uid}-panel`;
  const inputId = `${uid}-input`;
  const token = useAdminToken();
  const rejected = useTokenRejected();

  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const [draft, setDraft] = useState('');
  const buttonRef = useRef(null);
  const inputRef = useRef(null);
  const rootRef = useRef(null);

  // A 401 anywhere in the console opens the prompt.
  useEffect(
    () =>
      onUnauthorized(({ hadToken }) => {
        setNotice(
          hadToken
            ? 'The admin token was rejected (401). Enter a valid admin token.'
            : 'This action needs an admin token. Enter one to continue.',
        );
        setOpen(true);
      }),
    [],
  );

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const close = (restoreFocus = true) => {
    setOpen(false);
    setNotice('');
    setDraft('');
    if (restoreFocus) buttonRef.current?.focus();
  };

  const save = (e) => {
    e.preventDefault();
    if (!draft.trim()) return;
    setAdminToken(draft);
    close();
  };

  const state = rejected ? 'rejected' : token ? 'set' : 'missing';
  const pill = {
    rejected: { label: 'Token rejected', Icon: ShieldAlert, cls: 'border-red/40 bg-red/10 text-red' },
    set: { label: 'Admin token set', Icon: ShieldCheck, cls: 'border-teal/30 bg-teal/10 text-teal' },
    missing: { label: 'Admin token required', Icon: KeyRound, cls: 'border-amber/40 bg-amber/10 text-ink' },
  }[state];
  const source = getTokenSource();

  return (
    <div
      ref={rootRef}
      className="relative"
      onKeyDown={(e) => {
        if (e.key === 'Escape' && open) {
          e.stopPropagation();
          close();
        }
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => (open ? close(false) : setOpen(true))}
        className={`inline-flex min-h-[44px] items-center gap-2 rounded-lg border px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 ${pill.cls}`}
      >
        <pill.Icon
          className={`h-4 w-4 ${state === 'missing' ? 'text-amber' : ''}`}
          aria-hidden="true"
        />
        {pill.label}
      </button>

      {open && (
        <form
          id={panelId}
          onSubmit={save}
          aria-label="Admin token"
          className="absolute right-0 top-full z-30 mt-2 w-[min(20rem,calc(100vw-2rem))] rounded-xl border border-border bg-surface p-4 shadow-xl"
        >
          {notice && (
            <p role="alert" className="mb-3 rounded-lg border border-red/30 bg-red/10 p-2.5 text-sm text-red">
              {notice}
            </p>
          )}
          <label htmlFor={inputId} className="block text-sm font-medium text-ink">
            Admin token <span className="font-mono text-xs text-muted">X-Admin-Token</span>
          </label>
          <input
            id={inputId}
            ref={inputRef}
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            autoComplete="off"
            spellCheck={false}
            placeholder={token ? 'Enter a new token to replace it' : 'Paste the admin token'}
            className="mt-2 block min-h-[44px] w-full rounded-xl border border-field bg-surface px-3 text-base text-ink outline-none placeholder:text-muted/80 focus:border-teal focus:ring-4 focus:ring-teal/25 sm:text-sm"
          />
          <p className="mt-2 text-xs text-muted">
            {source === 'env' && !draft
              ? 'Using the build-time VITE_ADMIN_TOKEN. A token entered here overrides it.'
              : 'Kept for this browser tab only (sessionStorage), never in the page source.'}
          </p>
          <div className="mt-3 flex items-center justify-end gap-2">
            {source === 'session' && (
              <button
                type="button"
                onClick={() => {
                  clearAdminToken();
                  close();
                }}
                className="min-h-[44px] rounded-lg px-3 text-sm text-muted hover:text-ink focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
              >
                Clear
              </button>
            )}
            <button
              type="submit"
              disabled={!draft.trim()}
              className="min-h-[44px] rounded-lg bg-teal px-4 text-sm font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-teal"
            >
              Save token
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

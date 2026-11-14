import { useEffect, useState } from 'react';
import stats from '../../../../data/processed/stats.json'; // A's pipeline output, same names as labels.csv

// '' in dev (Vite proxies /api); set VITE_API_BASE_URL in production.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
const CACHE_KEY = 'wili.languages.v1';
const EMPTY = [];

let memory = null; // { languages, source: 'api' | 'cache' }
let inflight = null;

/** Validates a languages payload; returns a de-duplicated [{ code, name }] or null if malformed. */
export function cleanLanguages(data) {
  if (!Array.isArray(data)) return null;
  const seen = new Set();
  const out = [];
  for (const item of data) {
    if (!item || typeof item.code !== 'string' || typeof item.name !== 'string') return null;
    const code = item.code.trim();
    const name = item.name.trim();
    if (!code || !name) return null;
    if (seen.has(code.toLowerCase())) continue;
    seen.add(code.toLowerCase());
    out.push({ code, name });
  }
  return out.length > 0 ? out : null;
}

function readSession() {
  try {
    return cleanLanguages(JSON.parse(sessionStorage.getItem(CACHE_KEY)));
  } catch {
    return null;
  }
}

function writeSession(list) {
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(list));
  } catch {
    /* storage unavailable: the in-memory cache still works */
  }
}

/** Test helper: forget cached languages. */
export function resetLanguagesCache() {
  memory = null;
  inflight = null;
  try {
    sessionStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
}

/**
 * Resolves the 235 WiLI languages. Order: memory -> sessionStorage -> GET /api/v1/languages ->
 * data/processed/stats.json (bundled at build time). A fallback result is NOT cached, so the
 * next call retries the API.
 */
export function loadLanguages() {
  if (memory) return Promise.resolve(memory);
  if (inflight) return inflight;

  const cached = readSession();
  if (cached) {
    memory = { languages: cached, source: 'cache' };
    return Promise.resolve(memory);
  }

  inflight = (async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/languages`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list = cleanLanguages(await res.json());
      if (!list) throw new Error('Unexpected languages payload');
      writeSession(list);
      memory = { languages: list, source: 'api' };
      return memory;
    } catch {
      return { languages: cleanLanguages(stats.languages) ?? EMPTY, source: 'fallback' };
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

/**
 * React hook around loadLanguages().
 * source: 'loading' | 'api' | 'cache' | 'fallback'; unavailable = finished loading but nothing to offer.
 */
export function useLanguages() {
  const [state, setState] = useState(memory);

  useEffect(() => {
    if (state) return undefined;
    let cancelled = false;
    loadLanguages().then((result) => {
      if (!cancelled) setState(result);
    });
    return () => {
      cancelled = true;
    };
  }, [state]);

  const languages = state?.languages ?? EMPTY;
  return {
    languages,
    source: state?.source ?? 'loading',
    loading: state === null,
    unavailable: state !== null && languages.length === 0,
  };
}

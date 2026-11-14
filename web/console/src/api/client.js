import { useSyncExternalStore } from 'react';

// '' in dev (Vite proxies /api to FastAPI); set VITE_API_BASE_URL in production.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
const TOKEN_KEY = 'wili.adminToken';

/** Error thrown for any non-2xx response (status) or a network failure (status 0). */
export class ApiError extends Error {
  constructor(status, message, { data = null, network = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
    this.network = network;
  }
}

// ------------------------------------------------------------------ token store
// The token the Data Scientist types in the header lives in sessionStorage (this tab only).
// VITE_ADMIN_TOKEN is a build-time fallback for local demos. A token typed by hand wins.
let memoryToken = ''; // used if sessionStorage is unavailable
let rejected = false; // the last protected request was answered with 401
const subscribers = new Set();
const unauthorizedHandlers = new Set();

const notify = () => subscribers.forEach((cb) => cb());

function readStored() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? memoryToken;
  } catch {
    return memoryToken;
  }
}

/** Current admin token ('' when none). */
export function getAdminToken() {
  return readStored() || (import.meta.env.VITE_ADMIN_TOKEN ?? '');
}

/** Where the active token comes from: 'session' (typed), 'env' (build time) or 'none'. */
export function getTokenSource() {
  if (readStored()) return 'session';
  return import.meta.env.VITE_ADMIN_TOKEN ? 'env' : 'none';
}

export function setAdminToken(token) {
  const value = token.trim();
  if (!value) {
    clearAdminToken();
    return;
  }
  memoryToken = value;
  try {
    sessionStorage.setItem(TOKEN_KEY, value);
  } catch {
    /* in-memory copy is enough */
  }
  rejected = false;
  notify();
}

export function clearAdminToken() {
  memoryToken = '';
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
  rejected = false;
  notify();
}

const subscribe = (cb) => {
  subscribers.add(cb);
  return () => subscribers.delete(cb);
};

/** Re-renders when the token changes. */
export const useAdminToken = () => useSyncExternalStore(subscribe, getAdminToken);
/** True after a protected request was rejected with 401 and until the token changes. */
export const useTokenRejected = () => useSyncExternalStore(subscribe, () => rejected);

/** Called with { hadToken } whenever a protected request returns 401 or code asks for a token. */
export function onUnauthorized(handler) {
  unauthorizedHandlers.add(handler);
  return () => unauthorizedHandlers.delete(handler);
}

/** Asks the UI to prompt for a token (used by "Enter admin token" buttons). */
export function requestAdminToken() {
  const hadToken = Boolean(getAdminToken());
  unauthorizedHandlers.forEach((h) => h({ hadToken }));
}

function handleUnauthorized() {
  const hadToken = Boolean(getAdminToken());
  rejected = hadToken;
  notify();
  unauthorizedHandlers.forEach((h) => h({ hadToken }));
}

// -------------------------------------------------------------------- fetching
function detailOf(data, fallback) {
  if (data && typeof data.detail === 'string') return data.detail;
  if (data && Array.isArray(data.detail)) return data.detail.map((d) => d.msg ?? String(d)).join('; ');
  return fallback;
}

/**
 * JSON request to the API. Set `auth: true` for protected endpoints: the X-Admin-Token header is
 * attached and a 401 triggers the token prompt. Resolves with the parsed body (null if empty).
 * Throws ApiError (status 0 = network failure). AbortError is re-thrown untouched.
 */
export async function apiFetch(path, { method = 'GET', body, auth = false, signal } = {}) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (auth) {
    const token = getAdminToken();
    if (token) headers['X-Admin-Token'] = token;
  }

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError(0, "Can't reach the API. Check that the backend is running and try again.", {
      network: true,
    });
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* empty or non-JSON body */
  }

  if (!res.ok) {
    if (res.status === 401 && auth) handleUnauthorized();
    throw new ApiError(res.status, detailOf(data, `Request failed (${res.status})`), { data });
  }
  return data;
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from '../App.jsx';
import ModelRegistryReports from '../components/ModelRegistryReports.jsx';
import { clearAdminToken, onUnauthorized, setAdminToken } from '../api/client.js';
import { resetLanguagesCache } from '../hooks/useLanguages.js';

const LANGS = [
  { code: 'eng', name: 'English' }, { code: 'hin', name: 'Hindi' }, { code: 'srp', name: 'Serbian' },
  { code: 'hrv', name: 'Croatian' }, { code: 'bos', name: 'Bosnian' }, { code: 'vie', name: 'Vietnamese' },
  { code: 'dan', name: 'Danish' }, { code: 'nob', name: 'Norwegian Bokmål' }, { code: 'zh-yue', name: 'Cantonese' },
];
const MODELS = [
  { version: 'v1', algorithm: 'Multinomial Naive Bayes', created_at: '2026-10-20T09:30:00Z', data_version: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', status: 'archived', metrics: { accuracy: 0.915, macro_f1: 0.908 } },
  { version: 'v2', algorithm: 'Linear SVM', created_at: '2026-11-01T10:00:00Z', data_version: 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', status: 'active', metrics: { accuracy: 0.932, macro_f1: 0.926 } },
  { version: 'v3', algorithm: 'Linear SVM (tuned)', created_at: '2026-11-05T08:00:00Z', data_version: 'sha256:cccccccccccccccccccccccccccccccc', status: 'candidate', metrics: { accuracy: 0.94, macro_f1: 0.935 } },
];
const REPORTS = () => [
  { report_id: 11, text: 'Kal meeting hai, please sab log time par aana.', predicted_language: 'eng', confidence: 0.71, model_version: 'v2', reason: 'wrong_language', suggested_language: 'hin', comment: 'Hinglish', status: 'new', verified_language: null, created_at: '2026-10-31T16:42:10Z' },
  { report_id: 12, text: 'Ovo je jedan od najstarijih gradova.', predicted_language: 'hrv', confidence: 0.54, model_version: 'v1', reason: 'wrong_language', suggested_language: null, comment: null, status: 'new', verified_language: null, created_at: '2026-10-30T11:05:37Z' },
  { report_id: 13, text: 'Hvala vam puno na pomoći.', predicted_language: 'hrv', confidence: 0.52, model_version: 'v2', reason: 'mixed_languages', suggested_language: 'bos', comment: 'short', status: 'accepted', verified_language: 'bos', created_at: '2026-10-29T19:20:02Z' },
];

const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });

/** Fake of the protected API: enforces X-Admin-Token on the protected routes. */
function fakeApi(opts = {}) {
  const st = { models: structuredClone(MODELS), reports: opts.reports ?? REPORTS(), calls: [], urls: [] };
  const token = opts.token ?? 'secret';
  const fn = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init = {}) => {
    const method = init.method ?? 'GET';
    st.calls.push({ method, url, headers: init.headers ?? {}, body: init.body ? JSON.parse(init.body) : undefined });
    if (opts.down) throw new TypeError('Failed to fetch');
    const protectedRoute = url.startsWith('/api/v1/reports') || url.startsWith('/api/v1/models/');
    if (protectedRoute && init.headers?.['X-Admin-Token'] !== token) return json({ detail: 'Invalid admin token' }, 401);

    if (url === '/api/v1/languages') return json(LANGS);
    if (url === '/api/v1/models') return json(st.models);
    if (url === '/api/v1/models/promote') {
      if (opts.promoteStatus) return json({ detail: 'x' }, opts.promoteStatus);
      const v = JSON.parse(init.body).version;
      const prev = st.models.find((m) => m.status === 'active')?.version ?? null;
      st.models = st.models.map((m) => (m.version === v ? { ...m, status: 'active' } : m.status === 'active' ? { ...m, status: 'archived' } : m));
      return json({ active_version: v, previous_version: prev });
    }
    if (url === '/api/v1/models/rollback') {
      if (opts.rollbackStatus) return json({ detail: 'x' }, opts.rollbackStatus);
      st.models = st.models.map((m) => (m.version === 'v1' ? { ...m, status: 'active' } : m.status === 'active' ? { ...m, status: 'archived' } : m));
      return json({ active_version: 'v1', previous_version: 'v2' });
    }
    if (url.startsWith('/api/v1/reports?') || url === '/api/v1/reports') {
      st.urls.push(url);
      const p = new URLSearchParams(url.split('?')[1] ?? '');
      return json(st.reports.filter((r) => (!p.get('status') || r.status === p.get('status')) && (!p.get('model_version') || r.model_version === p.get('model_version'))));
    }
    const patch = url.match(/^\/api\/v1\/reports\/(\d+)$/);
    if (patch && method === 'PATCH') {
      if (opts.patchStatus) return json({ detail: 'x' }, opts.patchStatus);
      const body = JSON.parse(init.body);
      if (body.status === 'accepted' && !body.verified_language) return json({ detail: 'verified_language required' }, 400);
      st.reports = st.reports.map((r) => (r.report_id === Number(patch[1]) ? { ...r, ...body } : r));
      return json(st.reports.find((r) => r.report_id === Number(patch[1])));
    }
    return json({ detail: 'Not Found' }, 404);
  });
  return { st, fn };
}

beforeEach(() => {
  window.location.hash = '';
  resetLanguagesCache();
  clearAdminToken();
  setAdminToken('secret');
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  clearAdminToken();
});

const registry = () => screen.getByRole('region', { name: 'Registered Production Models' });
const tableRows = () => within(within(registry()).getAllByRole('rowgroup')[1]).getAllByRole('row');
const promoteBtn = (v) => within(registry()).getAllByRole('button', { name: `Promote ${v} to Production` })[0];
const reportsSection = () => screen.getByRole('region', { name: /Logged User Misclassification Reports/ });
const cards = () => within(reportsSection()).getAllByRole('listitem').filter((li) => li.closest('ul[aria-label="Misclassification reports"]') && li.querySelector('dl'));
const ready = async () => { await screen.findByText('Linear SVM (tuned)', { selector: 'td' }); await screen.findByText('#11'); };

describe('registry: new schema (MV-2)', () => {
  it('reads metrics.*, shows all three lifecycle statuses, created_at and data_version', async () => {
    fakeApi(); render(<ModelRegistryReports />); await ready();
    expect(within(registry()).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Version Tag','Algorithm Name','Status','Created','Data Version','Test Accuracy','Macro-F1','Action']);
    const [v1, v2, v3] = tableRows();
    expect(v1.cells[2].textContent).toBe('Archived'); expect(v1.cells[2].firstChild.className).toContain('border-dashed');
    expect(v2.cells[2].textContent).toBe('Active Production'); expect(v2.cells[2].firstChild.className).toContain('bg-teal/10');
    expect(v3.cells[2].textContent).toBe('Candidate'); expect(v3.cells[2].firstChild.className).toContain('bg-subtle');
    expect([...v2.cells].slice(5, 7).map((c) => c.textContent)).toEqual(['93.2%', '0.926']);
    expect(v2.cells[3].textContent).toMatch(/^01 Nov 2026, \d{2}:\d{2}$/);
    expect(v2.cells[4].textContent).toBe(`${MODELS[1].data_version.slice(0, 22)}…`);
    expect(v2.cells[4].querySelector('span').getAttribute('title')).toBe(MODELS[1].data_version);
    expect(within(v2).queryByRole('button')).toBeNull();
    expect(within(v1).getByRole('button', { name: 'Promote v1 to Production' })).toBeTruthy(); // archived can be restored
    expect(screen.getByText(/Active production:/).textContent).toContain('v2 (Linear SVM)');
  });

  it('missing metrics render as an em dash', async () => {
    fakeApi(); render(<ModelRegistryReports />); await screen.findByText('Linear SVM (tuned)', { selector: 'td' });
    cleanup(); vi.restoreAllMocks();
    const f = fakeApi(); f.st.models = [{ ...MODELS[1], metrics: null }];
    render(<ModelRegistryReports />);
    await screen.findByText('Linear SVM', { selector: 'td' });
    expect([...tableRows()[0].cells].slice(5, 7).map((c) => c.textContent)).toEqual(['—', '—']);
  });
});

describe('promote / rollback (MV-3, MV-4, MV-5)', () => {
  it('promote sends the token + { version }, swaps badges at once and reports previous_version', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    fireEvent.click(promoteBtn('v3'));
    const call = await waitFor(() => { const c = st.calls.find((x) => x.url === '/api/v1/models/promote'); expect(c).toBeTruthy(); return c; });
    expect(call.method).toBe('POST'); expect(call.body).toEqual({ version: 'v3' }); expect(call.headers['X-Admin-Token']).toBe('secret');
    await waitFor(() => expect(tableRows()[2].cells[2].textContent).toBe('Active Production'));
    expect(tableRows()[1].cells[2].textContent).toBe('Archived'); // previous active
    expect(screen.getAllByRole('status')[0].textContent).toContain('Successfully promoted model v3 to active production engine (previous: v2). No container restart required.');
    await waitFor(() => expect(st.calls.filter((c) => c.url === '/api/v1/models').length).toBe(2)); // background re-sync
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss notification' }));
    expect(screen.queryByText(/Successfully promoted/)).toBeNull();
  });

  it('rollback posts without a body and shows the banner', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    fireEvent.click(screen.getByRole('button', { name: 'Rollback to Previous Version' }));
    await screen.findByText(/Successfully rolled back to model v1 \(replacing v2\)/);
    const call = st.calls.find((x) => x.url === '/api/v1/models/rollback');
    expect(call.method).toBe('POST'); expect(call.body).toBeUndefined(); expect(call.headers['X-Admin-Token']).toBe('secret');
    expect(tableRows()[0].cells[2].textContent).toBe('Active Production');
  });

  it('maps 401 / 404 / 409 to the agreed messages', async () => {
    const cases = [
      ['promote', 404, 'Model version not found in models/ directory.'],
      ['promote', 409, 'Conflict: Version is already active, or no previous version exists in history to roll back to.'],
      ['rollback', 409, 'Conflict: Version is already active, or no previous version exists in history to roll back to.'],
    ];
    for (const [kind, status, msg] of cases) {
      cleanup(); vi.restoreAllMocks(); resetLanguagesCache();
      fakeApi(kind === 'promote' ? { promoteStatus: status } : { rollbackStatus: status });
      render(<ModelRegistryReports />); await ready();
      fireEvent.click(kind === 'promote' ? promoteBtn('v3') : screen.getByRole('button', { name: 'Rollback to Previous Version' }));
      expect((await screen.findAllByRole('alert')).map((a) => a.textContent).join(' ')).toContain(msg);
      expect(tableRows()[1].cells[2].textContent).toBe('Active Production'); // nothing changed
    }
  });

  it('401 shows the unauthorized message and asks for a token', async () => {
    const asked = vi.fn(); const off = onUnauthorized(asked);
    fakeApi({ token: 'other-token' }); render(<ModelRegistryReports />); await screen.findByText('Linear SVM (tuned)', { selector: 'td' });
    fireEvent.click(promoteBtn('v3'));
    expect((await screen.findAllByRole('alert')).map((a) => a.textContent).join(' ')).toContain('Unauthorized: Invalid or missing X-Admin-Token.');
    expect(asked).toHaveBeenCalled(); off();
  });

  it('API down: offline sample registry with actions disabled; Retry recovers', async () => {
    fakeApi({ down: true }); render(<ModelRegistryReports />);
    expect(await screen.findByText(/API unreachable: showing offline sample data/)).toBeTruthy();
    expect(tableRows()).toHaveLength(2);
    expect(promoteBtn('v2').disabled).toBe(true);
    expect(screen.getByRole('button', { name: 'Rollback to Previous Version' }).disabled).toBe(true);
    vi.restoreAllMocks(); fakeApi();
    fireEvent.click(within(registry()).getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.queryByText(/API unreachable: showing offline sample data/)).toBeNull());
    expect(promoteBtn('v3').disabled).toBe(false);
  });
});

describe('reports: live data, filters, search (RV-1)', () => {
  it('fetches with the admin token and renders every ReportItem field', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    expect(st.calls.find((c) => c.url === '/api/v1/reports').headers['X-Admin-Token']).toBe('secret');
    expect(screen.getByText('3 reports logged')).toBeTruthy();
    const c = cards()[0];
    expect(c.textContent).toContain('#11'); expect(c.textContent).toContain('Model: v2');
    expect(c.textContent).toContain('Kal meeting hai, please sab log time par aana.');
    expect(c.textContent).toContain('English eng'); expect(c.textContent).toContain('71% confidence');
    expect(c.textContent).toContain('Wrong language'); expect(c.textContent).toContain('Hindi hin');
    expect(c.textContent).toContain('Comment: Hinglish'); expect(c.textContent).toMatch(/31 Oct 2026, \d{2}:\d{2}/);
    expect(within(c).getByText('new').className).toContain('bg-amber/10');
    const done = cards()[2];
    expect(within(done).getByText('accepted').className).toContain('bg-teal/10');
    expect(done.textContent).toContain('Bosnian bos'); expect(within(done).queryByRole('button', { name: /Accept/ })).toBeNull();
  });

  it('status and model_version filters become query params ("all" sends none)', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    expect(st.urls.at(-1)).toBe('/api/v1/reports');
    expect(within(reportsSection()).getAllByRole('option').map((o) => o.textContent)).toEqual(['All statuses', 'New', 'Accepted', 'Rejected', 'All versions', 'v1', 'v2', 'v3']);
    fireEvent.change(within(reportsSection()).getByLabelText('Status'), { target: { value: 'new' } });
    await waitFor(() => expect(st.urls.at(-1)).toBe('/api/v1/reports?status=new'));
    await waitFor(() => expect(cards()).toHaveLength(2));
    fireEvent.change(within(reportsSection()).getByLabelText('Model version'), { target: { value: 'v2' } });
    await waitFor(() => expect(st.urls.at(-1)).toBe('/api/v1/reports?status=new&model_version=v2'));
    await waitFor(() => expect(cards()).toHaveLength(1));
    expect(screen.getByText('1 reports match filters')).toBeTruthy();
  });

  it('search matches keywords, codes and names; empty state clears', async () => {
    fakeApi(); render(<ModelRegistryReports />); await ready();
    const box = screen.getByRole('searchbox');
    expect(box.getAttribute('placeholder')).toBe('Search by language code or keyword...');
    fireEvent.change(box, { target: { value: 'hinglish' } }); expect(cards()).toHaveLength(1);
    fireEvent.change(box, { target: { value: 'croatian' } }); expect(cards()).toHaveLength(2);
    fireEvent.change(box, { target: { value: 'bos' } }); expect(cards()).toHaveLength(1);
    fireEvent.change(box, { target: { value: 'zzzz' } });
    expect(screen.getByText(/No reports match “zzzz”/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Clear search' })); expect(cards()).toHaveLength(3);
  });

  it('no token / 401: shows unauthorized prompt, then loads after a token is entered', async () => {
    clearAdminToken();
    const asked = vi.fn(); const off = onUnauthorized(asked);
    fakeApi(); render(<ModelRegistryReports />);
    const alert = await within(reportsSection()).findByRole('alert');
    expect(alert.textContent).toContain('Unauthorized: Invalid or missing X-Admin-Token.');
    expect(asked).toHaveBeenCalledWith({ hadToken: false });
    expect(screen.queryByText(/offline sample reports/)).toBeNull(); // server is up: no sample fallback
    await act(async () => { setAdminToken('secret'); });
    await screen.findByText('#11'); expect(screen.getByText('3 reports logged')).toBeTruthy(); off();
  });

  it('server down: clearly labelled sample reports, filtered the same way, review disabled', async () => {
    fakeApi({ down: true }); render(<ModelRegistryReports />);
    expect(await screen.findByText(/API unreachable: showing offline sample reports/)).toBeTruthy();
    expect(screen.getByText('6 reports logged')).toBeTruthy();
    expect(within(cards()[0]).getByRole('button', { name: /Accept/ }).disabled).toBe(true);
    fireEvent.change(within(reportsSection()).getByLabelText('Status'), { target: { value: 'rejected' } });
    await waitFor(() => expect(cards()).toHaveLength(1));
  });

  it('endpoint missing (404) also falls back to samples and says why', async () => {
    const { st } = fakeApi(); const orig = globalThis.fetch;
    vi.spyOn(globalThis, 'fetch').mockImplementation((u, i) => (String(u).startsWith('/api/v1/reports') ? Promise.resolve(json({ detail: 'Not Found' }, 404)) : orig(u, i)));
    render(<ModelRegistryReports />);
    expect(await screen.findByText(/API answered HTTP 404: showing offline sample reports/)).toBeTruthy(); expect(st).toBeTruthy();
  });
});

describe('review actions (RV-2)', () => {
  const card = (id) => cards().find((c) => c.textContent.includes(`#${id}`));

  it('Accept: pre-filled with suggested_language, PATCHes verified_language, updates the badge', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    const c = card(11);
    expect(within(c).getByRole('combobox').value).toBe('Hindi (hin)');
    fireEvent.click(within(c).getByRole('button', { name: /Accept/ }));
    await waitFor(() => expect(within(card(11)).getByText('accepted')).toBeTruthy());
    const patch = st.calls.find((x) => x.method === 'PATCH');
    expect(patch.url).toBe('/api/v1/reports/11'); expect(patch.body).toEqual({ status: 'accepted', verified_language: 'hin' });
    expect(patch.headers['X-Admin-Token']).toBe('secret');
    expect(within(card(11)).queryByRole('combobox')).toBeNull(); // no longer reviewable
    expect(card(11).textContent).toContain('Hindi hin'); // verified badge
  });

  it('Accept lets the reviewer pick another language from the 235-list', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    const box = within(card(11)).getByRole('combobox');
    fireEvent.change(box, { target: { value: 'Serb' } });
    fireEvent.click(await within(card(11)).findByRole('option', { name: /Serbian/ }));
    fireEvent.click(within(card(11)).getByRole('button', { name: /Accept/ }));
    await waitFor(() => expect(st.calls.find((x) => x.method === 'PATCH')).toBeTruthy());
    expect(st.calls.find((x) => x.method === 'PATCH').body.verified_language).toBe('srp');
  });

  it('Accept without a verified language is blocked client-side (no PATCH)', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    const c = card(12); // no suggested_language
    fireEvent.click(within(c).getByRole('button', { name: /Accept/ }));
    expect((await within(c).findByRole('alert')).textContent).toBe('Select a verified language from the list before accepting.');
    expect(within(c).getByRole('combobox').getAttribute('aria-invalid')).toBe('true');
    expect(st.calls.some((x) => x.method === 'PATCH')).toBe(false);
    fireEvent.change(within(c).getByRole('combobox'), { target: { value: 'xyz' } }); // strict: not a WiLI language
    fireEvent.click(within(c).getByRole('button', { name: /Accept/ }));
    expect((await within(c).findByRole('alert')).textContent).toContain('Select a verified language');
    expect(st.calls.some((x) => x.method === 'PATCH')).toBe(false);
  });

  it('Reject: PATCHes only { status: "rejected" } and updates the badge', async () => {
    const { st } = fakeApi(); render(<ModelRegistryReports />); await ready();
    fireEvent.click(within(card(12)).getByRole('button', { name: /Reject/ }));
    await waitFor(() => expect(within(card(12)).getByText('rejected')).toBeTruthy());
    expect(st.calls.find((x) => x.method === 'PATCH').body).toEqual({ status: 'rejected' });
    expect(within(card(12)).getByText('rejected').className).toContain('bg-subtle');
  });

  it('shows inline alerts for 400, 401 and 404', async () => {
    const cases = [[400, 'a verified language is required'], [404, 'This report was not found'], [401, 'Unauthorized: Invalid or missing X-Admin-Token.']];
    for (const [status, msg] of cases) {
      cleanup(); vi.restoreAllMocks(); resetLanguagesCache(); setAdminToken('secret');
      fakeApi({ patchStatus: status }); render(<ModelRegistryReports />); await ready();
      fireEvent.click(within(card(11)).getByRole('button', { name: /Reject/ }));
      expect((await within(card(11)).findByRole('alert')).textContent).toContain(msg);
      expect(within(card(11)).getByText('new')).toBeTruthy(); // unchanged
    }
  });
});

describe('admin token control + App (MV-5)', () => {
  it('pill reflects token state; saving, clearing and Esc behave', async () => {
    clearAdminToken(); fakeApi(); render(<App />);
    const pill = screen.getByRole('button', { name: 'Admin token required' });
    expect(pill.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(pill);
    const input = screen.getByLabelText(/Admin token/, { selector: 'input' });
    expect(input.type).toBe('password'); expect(document.activeElement).toBe(input);
    expect(screen.getByRole('button', { name: 'Save token' }).disabled).toBe(true);
    fireEvent.change(input, { target: { value: ' topsecret ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save token' }));
    const set = screen.getByRole('button', { name: 'Admin token set' });
    expect(sessionStorage.getItem('wili.adminToken')).toBe('topsecret');
    expect(document.activeElement).toBe(set);
    fireEvent.click(set); fireEvent.keyDown(screen.getByRole('form', { name: 'Admin token' }), { key: 'Escape' });
    expect(screen.queryByRole('form', { name: 'Admin token' })).toBeNull(); expect(document.activeElement).toBe(set);
    fireEvent.click(set); fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(sessionStorage.getItem('wili.adminToken')).toBeNull();
    expect(screen.getByRole('button', { name: 'Admin token required' })).toBeTruthy();
  });

  it('a 401 from the API opens the prompt with a clear message and flips the pill to "rejected"', async () => {
    setAdminToken('stale'); fakeApi({ token: 'fresh' }); render(<App />);
    fireEvent.click(screen.getByRole('tab', { name: 'Model Registry & Reports' }));
    const form = await screen.findByRole('form', { name: 'Admin token' });
    expect(within(form).getByRole('alert').textContent).toContain('The admin token was rejected (401). Enter a valid admin token.');
    expect(screen.getByRole('button', { name: 'Token rejected' })).toBeTruthy();
    fireEvent.change(within(form).getByLabelText(/Admin token/, { selector: 'input' }), { target: { value: 'fresh' } });
    fireEvent.click(within(form).getByRole('button', { name: 'Save token' }));
    await screen.findByText('#11'); // reports refetched with the new token
    expect(screen.getByRole('button', { name: 'Admin token set' })).toBeTruthy();
  });

  it('registry tab mounts in the app', async () => {
    fakeApi(); render(<App />);
    fireEvent.click(screen.getByRole('tab', { name: 'Model Registry & Reports' }));
    expect(await screen.findByRole('heading', { name: 'Model Registry & User Feedback Audit' })).toBeTruthy(); // hashchange is async
    expect(window.location.hash).toBe('#registry');
  });
});

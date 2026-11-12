// Pure helpers for the 235 WiLI language labels ({ code, name }). Codes include hyphenated
// Wikipedia-style labels such as "zh-yue" or "be-tarask", so never assume 3 letters.

/** Case- and accent-insensitive form used for matching ("volapuk" finds "Volapük"). */
export const normalize = (s) =>
  s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').toLowerCase().trim();

export const languageLabel = (lang) => `${lang.name} (${lang.code})`;

export const findByCode = (languages, code) =>
  code ? languages.find((l) => normalize(l.code) === normalize(code)) : undefined;

/** Up to `limit` languages whose name or code matches, best matches first. */
export function searchLanguages(languages, query, limit = 6) {
  const q = normalize(query);
  if (!q) return [];
  const scored = [];
  for (const lang of languages) {
    const code = normalize(lang.code);
    const name = normalize(lang.name);
    let score = -1;
    if (code === q) score = 0;
    else if (name === q) score = 1;
    else if (name.startsWith(q)) score = 2;
    else if (code.startsWith(q)) score = 3;
    else if (name.includes(q)) score = 4;
    if (score >= 0) scored.push({ lang, score });
  }
  scored.sort((a, b) => a.score - b.score || a.lang.name.localeCompare(b.lang.name));
  return scored.slice(0, limit).map((s) => s.lang);
}

/**
 * Strict lookup: returns the language only when `query` is exactly one of the known codes, one of
 * the known names, or the "Name (code)" label the picker produces. There is NO "any 3-letter code"
 * escape hatch: anything else returns null.
 */
export function resolveLanguage(languages, query) {
  const raw = query.trim();
  if (!raw) return null;
  const paren = raw.match(/\(([^()\s]+)\)\s*$/);
  const candidates = [normalize(raw), paren ? normalize(paren[1]) : null].filter(Boolean);
  for (const q of candidates) {
    const hit =
      languages.find((l) => normalize(l.code) === q) ?? languages.find((l) => normalize(l.name) === q);
    if (hit) return hit;
  }
  return null;
}

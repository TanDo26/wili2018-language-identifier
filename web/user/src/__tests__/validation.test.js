import { describe, expect, it } from 'vitest';
import { countChars, getStatus } from '../components/TextInputCard.jsx';
import { REPORT_REASONS, validateReportForm } from '../components/ReportDialog.jsx';
import fallback from '../data/languages.fallback.json';

describe('getStatus (TextInputCard): trimmed Unicode code points, 20..5,000', () => {
  it('empty, and "touched" empty', () => {
    expect(getStatus('', false)).toBe('empty');
    expect(getStatus('', true)).toBe('emptyError');
  });

  it('whitespace-only counts as empty', () => {
    expect(getStatus('   \n\t  ', false)).toBe('empty');
    expect(getStatus('   \n\t  ', true)).toBe('emptyError');
  });

  it('trims leading and trailing whitespace before counting', () => {
    expect(countChars('   hello   ')).toBe(5);
    expect(getStatus(`${' '.repeat(30)}short text${' '.repeat(30)}`, false)).toBe('tooShort');
    expect(getStatus(`  ${'a'.repeat(20)}  `, false)).toBe('valid');
  });

  it('boundaries: 19 short, 20 valid, 5000 valid, 5001 too long', () => {
    expect(getStatus('a'.repeat(19), false)).toBe('tooShort');
    expect(getStatus('a'.repeat(20), false)).toBe('valid');
    expect(getStatus('a'.repeat(5000), false)).toBe('valid');
    expect(getStatus('a'.repeat(5001), false)).toBe('tooLong');
  });

  it('counts emoji / surrogate pairs as ONE character, like Python len()', () => {
    // 10 emoji = 10 code points (but .length is 20, which the old check wrongly accepted)
    expect('😀'.repeat(10).length).toBe(20);
    expect(countChars('😀'.repeat(10))).toBe(10);
    expect(getStatus('😀'.repeat(10), false)).toBe('tooShort');
    expect(getStatus('😀'.repeat(20), false)).toBe('valid');
    // 5,000 emoji = 5,000 code points = valid (.length would be 10,000 -> wrongly "too long")
    expect(getStatus('😀'.repeat(5000), false)).toBe('valid');
    expect(getStatus('😀'.repeat(5001), false)).toBe('tooLong');
  });

  it('a ZWJ family emoji is 5 code points (Python len() agrees)', () => {
    expect(countChars('👨‍👩‍👧')).toBe(5);
  });

  it('Vietnamese with combining marks counts code points, not glyphs', () => {
    const composed = 'Việt Nam'.normalize('NFC');
    const decomposed = 'Việt Nam'.normalize('NFD');
    expect(countChars(composed)).toBe(8);
    expect(countChars(decomposed)).toBeGreaterThan(8); // same as Python len() on NFD text
  });
});

describe('validateReportForm (ReportDialog)', () => {
  const languages = [
    { code: 'eng', name: 'English' },
    { code: 'vie', name: 'Vietnamese' },
    { code: 'zh-yue', name: 'Cantonese' },
    { code: 'be-tarask', name: 'Belarusian (Taraškievica)' },
  ];
  const base = { reason: 'wrong_language', dontKnow: false, query: '', languages, predictedCode: 'vie' };
  const run = (over) => validateReportForm({ ...base, ...over });

  it('exposes exactly the four fixed reason values', () => {
    expect(REPORT_REASONS.map((r) => r.value)).toEqual([
      'wrong_language',
      'mixed_languages',
      'not_a_language',
      'other',
    ]);
  });

  it('reason is mandatory and must be one of the enum values', () => {
    expect(run({ reason: '' }).errors.reason).toBeTruthy();
    expect(run({ reason: 'nonsense' }).errors.reason).toBeTruthy();
    for (const r of REPORT_REASONS) expect(run({ reason: r.value }).valid).toBe(true);
  });

  it('suggested language is optional: empty -> null', () => {
    const r = run({ query: '' });
    expect(r.valid).toBe(true);
    expect(r.suggestedLanguage).toBeNull();
  });

  it('"I don\'t know" -> null even if text is left in the box', () => {
    const r = run({ dontKnow: true, query: 'garbage that is not a language' });
    expect(r.valid).toBe(true);
    expect(r.suggestedLanguage).toBeNull();
  });

  it('accepts a language by code, by name, and by the "Name (code)" label', () => {
    expect(run({ query: 'eng' }).suggestedLanguage).toBe('eng');
    expect(run({ query: 'English' }).suggestedLanguage).toBe('eng');
    expect(run({ query: 'english' }).suggestedLanguage).toBe('eng');
    expect(run({ query: 'English (eng)' }).suggestedLanguage).toBe('eng');
  });

  it('accepts hyphenated WiLI codes (zh-yue, be-tarask)', () => {
    expect(run({ query: 'zh-yue' }).suggestedLanguage).toBe('zh-yue');
    expect(run({ query: 'Cantonese (zh-yue)' }).suggestedLanguage).toBe('zh-yue');
    expect(run({ query: 'Belarusian (Taraškievica) (be-tarask)' }).suggestedLanguage).toBe('be-tarask');
  });

  it('is strict: a well-formed 3-letter code that is not in the list is rejected', () => {
    expect(run({ query: 'xyz' }).errors.language).toBeTruthy();
    expect(run({ query: 'yue' }).errors.language).toBeTruthy(); // WiLI uses zh-yue
    expect(run({ query: 'zul' }).errors.language).toBeTruthy(); // not one of the 235 labels
    expect(run({ query: 'xyz' }).valid).toBe(false);
  });

  it('rejects the language the model already predicted', () => {
    const r = run({ query: 'Vietnamese' });
    expect(r.valid).toBe(false);
    expect(r.errors.language).toMatch(/predicted/);
    expect(run({ query: 'VIE' }).errors.language).toMatch(/predicted/);
  });

  it('with the built-in fallback list: all 235 codes validate, others do not', () => {
    expect(fallback).toHaveLength(235);
    for (const l of fallback) {
      const r = validateReportForm({ ...base, languages: fallback, predictedCode: 'none', query: l.code });
      expect(r.suggestedLanguage).toBe(l.code);
    }
    expect(validateReportForm({ ...base, languages: fallback, query: 'yue' }).valid).toBe(false);
  });
});

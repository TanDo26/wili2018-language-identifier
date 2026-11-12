import { useId, useState } from 'react';
import { ExternalLink, Search, X } from 'lucide-react';
import { DATASET_STATS, SAMPLES } from '../data/sampleDataset.js';

const DATASET_URL = 'https://huggingface.co/datasets/MartinThoma/wili_2018';

/** Case- and accent-insensitive ("ha noi" finds "Hà Nội"). */
const normalize = (s) =>
  s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').toLowerCase().trim();

function filterRows(rows, query) {
  const q = normalize(query);
  if (!q) return rows;
  return rows.filter(
    (r) =>
      normalize(r.sentence).includes(q) ||
      normalize(r.name).includes(q) ||
      r.code.startsWith(q),
  );
}

const SCHEMA = [
  { field: 'sentence', type: 'string' },
  { field: 'label', type: `ClassLabel (${DATASET_STATS.languages} classes)` },
  { field: 'split_id', type: 'integer' },
];

/**
 * Dataset Explorer (DS-01, FR6): browse sample paragraphs per split, search by language or text,
 * with the schema and a link to the official dataset alongside.
 * Rows come from src/data/sampleDataset.js until the API provides them.
 */
export default function DatasetExplorer() {
  const [splitId, setSplitId] = useState('train');
  const [query, setQuery] = useState('');
  const searchId = useId();

  const split = DATASET_STATS.splits.find((s) => s.id === splitId);
  const allRows = SAMPLES[splitId];
  const rows = filterRows(allRows, query);

  return (
    <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      {/* ---------------------------------------------------- sample browser */}
      <section aria-labelledby="browser-title" className="min-w-0 rounded-xl border border-border bg-surface">
        <div className="space-y-3 border-b border-border p-4 sm:p-5">
          <h2 id="browser-title" className="text-lg font-bold text-ink">
            Sample browser
          </h2>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div
              role="group"
              aria-label="Dataset split"
              className="flex shrink-0 rounded-xl bg-subtle p-1"
            >
              {DATASET_STATS.splits.map((s) => {
                const active = s.id === splitId;
                return (
                  <button
                    key={s.id}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setSplitId(s.id)}
                    className={`min-h-[44px] flex-1 rounded-lg px-4 text-sm transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30 sm:flex-none ${
                      active ? 'bg-surface font-medium text-ink shadow-sm' : 'text-muted hover:text-ink'
                    }`}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>

            <div className="relative min-w-0 flex-1">
              <label htmlFor={searchId} className="sr-only">
                Search language or text
              </label>
              <Search
                className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
                aria-hidden="true"
              />
              <input
                id={searchId}
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search language or text..."
                autoComplete="off"
                className="block min-h-[44px] w-full rounded-full border border-transparent bg-subtle pl-10 pr-4 text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:border-teal focus:bg-surface focus:ring-4 focus:ring-teal/25 sm:text-sm"
              />
            </div>
          </div>

          <p aria-live="polite" className="text-xs text-muted">
            Showing {rows.length} of {allRows.length} sample rows from the {split.label} split (
            {split.size.toLocaleString('en-US')} paragraphs in total).
          </p>
        </div>

        {rows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <caption className="sr-only">
                Sample paragraphs from the {split.label} split with their language labels
              </caption>
              <thead>
                <tr className="border-b border-border text-xs text-muted">
                  <th scope="col" className="px-4 py-3 font-medium sm:px-5">
                    Sentence (Wikipedia Paragraph)
                  </th>
                  <th scope="col" className="w-32 px-4 py-3 font-medium sm:w-44 sm:px-5">
                    Language Label
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {rows.map((r) => (
                  <tr key={r.id} className="align-top">
                    <td className="px-4 py-3.5 text-ink sm:px-5">
                      <span lang={r.lang} dir={r.dir} className="block break-words leading-relaxed">
                        {r.sentence}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 sm:px-5">
                      <span className="inline-block rounded-full bg-chip px-2.5 py-1 text-xs font-medium text-[#2B4442]">
                        {r.name}
                      </span>
                      <span className="ml-2 font-mono text-xs text-muted">{r.code}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center">
            <p className="text-sm text-ink">
              No samples match “{query.trim()}” in the {split.label} split.
            </p>
            <button
              type="button"
              onClick={() => setQuery('')}
              className="mt-3 inline-flex min-h-[44px] items-center gap-1.5 rounded-lg px-4 text-sm font-medium text-teal hover:bg-subtle focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
            >
              <X className="h-4 w-4" aria-hidden="true" />
              Clear search
            </button>
          </div>
        )}

        <p className="border-t border-border px-4 py-3 text-xs text-muted sm:px-5">
          Rows are illustrative mock samples, not the real dataset. The live data will replace them
          once the API exposes it.
        </p>
      </section>

      {/* ------------------------------------------------------------ sidebar */}
      <aside className="space-y-4" aria-label="Dataset information">
        <section aria-labelledby="schema-title" className="rounded-xl border border-border bg-surface p-5">
          <h2 id="schema-title" className="text-lg font-bold text-ink">
            Schema
          </h2>
          <ul className="mt-3 space-y-2">
            {SCHEMA.map((f) => (
              <li key={f.field} className="rounded-lg bg-subtle px-3 py-2 font-mono text-sm text-ink">
                <span className="text-teal">{f.field}</span>: {f.type}
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="docs-title" className="rounded-xl bg-ink p-5 text-white">
          <h2 id="docs-title" className="text-lg font-bold">
            WiLI-2018 Dataset
          </h2>
          <p className="mt-2 text-sm text-white/80">
            Official 235-language benchmark curated by Martin Thoma (2018).
          </p>
          <a
            href={DATASET_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 inline-flex min-h-[44px] items-center gap-2 rounded-lg bg-surface px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-chip focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-white/40"
          >
            View on Hugging Face
            <ExternalLink className="h-4 w-4" aria-hidden="true" />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        </section>
      </aside>
    </div>
  );
}

import { DATASET_STATS, TOTAL_PARAGRAPHS } from '../data/sampleDataset.js';

const fmt = (n) => n.toLocaleString('en-US');

/** Three headline numbers. Stacked on phones (40px figures do not fit three-across at 390px). */
export default function MetricCards() {
  const splitSummary = DATASET_STATS.splits.map((s) => `${s.short} (${s.size / 1000}k)`).join(', ');

  const metrics = [
    { value: fmt(DATASET_STATS.languages), label: 'languages' },
    { value: fmt(TOTAL_PARAGRAPHS), label: 'paragraphs' },
    { value: `${DATASET_STATS.splits.length} splits`, label: splitSummary },
  ];

  return (
    <ul aria-label="Dataset summary" className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {metrics.map((m) => (
        <li key={m.label} className="rounded-xl border border-border bg-surface p-5">
          <p className="text-[40px] font-bold leading-none text-teal">{m.value}</p>
          <p className="mt-2 text-sm text-muted">{m.label}</p>
        </li>
      ))}
    </ul>
  );
}

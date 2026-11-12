/** Stand-in for console sections that are not built yet. */
export default function TabPlaceholder({ ticket, title }) {
  return (
    <div className="rounded-xl border border-dashed border-field bg-surface p-8 text-center">
      <p className="font-mono text-xs text-muted">{ticket}</p>
      <h2 className="mt-2 text-lg font-bold text-ink">{title}</h2>
      <p className="mt-1 text-sm text-muted">This section has not been built yet.</p>
    </div>
  );
}

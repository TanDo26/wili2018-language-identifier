import { useEffect, useRef, useState } from 'react';
import Hero from './components/Hero.jsx';
import MetricCards from './components/MetricCards.jsx';
import DatasetExplorer from './components/DatasetExplorer.jsx';
import PipelineLeakage from './components/PipelineLeakage.jsx';
import ModelEvaluation from './components/ModelEvaluation.jsx';
import ModelRegistryReports from './components/ModelRegistryReports.jsx';
import TabPlaceholder from './components/TabPlaceholder.jsx';
import AdminTokenControl from './components/AdminTokenControl.jsx';

// End-user app: port 3000 in docker-compose.yml. Override with VITE_USER_APP_URL.
const USER_APP_URL = import.meta.env.VITE_USER_APP_URL ?? 'http://localhost:3000';

const TABS = [
  { id: 'dataset', label: 'Dataset Explorer', ticket: 'DS-01' },
  { id: 'pipeline', label: 'Data Pipeline & Leakage', ticket: 'DS-02' },
  { id: 'evaluation', label: 'Model Evaluation', ticket: 'DS-03 & DS-04' },
  { id: 'registry', label: 'Model Registry & Reports', ticket: 'DS-05' },
];

// Tabs that already have a real view; the others show a placeholder.
const IMPLEMENTED = ['dataset', 'pipeline', 'evaluation', 'registry'];

/** The active tab lives in the URL hash (#pipeline), so reloads and shared links keep the section. */
function tabFromHash() {
  const id = window.location.hash.replace('#', '');
  return TABS.some((t) => t.id === id) ? id : TABS[0].id;
}

export default function App() {
  const [activeId, setActiveId] = useState(tabFromHash);
  const tabRefs = useRef({});

  useEffect(() => {
    const onHashChange = () => setActiveId(tabFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const select = (id) => {
    window.location.hash = id; // triggers hashchange -> state
  };

  // WAI-ARIA tabs: arrows / Home / End move between tabs (roving tabindex).
  const onTabKeyDown = (e, index) => {
    const last = TABS.length - 1;
    let next = null;
    if (e.key === 'ArrowRight') next = index === last ? 0 : index + 1;
    else if (e.key === 'ArrowLeft') next = index === 0 ? last : index - 1;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = last;
    if (next === null) return;
    e.preventDefault();
    select(TABS[next].id);
    tabRefs.current[TABS[next].id]?.focus();
  };

  const active = TABS.find((t) => t.id === activeId);

  return (
    <>
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 px-4 sm:px-6 lg:flex-nowrap">
          <span className="order-1 py-3 text-xl font-bold text-ink">WiLI-2018</span>

          <div
            role="tablist"
            aria-label="Console sections"
            className="order-3 -mx-4 flex w-full overflow-x-auto px-4 sm:-mx-6 sm:px-6 lg:order-2 lg:mx-0 lg:w-auto lg:overflow-visible lg:px-0"
          >
            {TABS.map((t, i) => {
              const selected = t.id === activeId;
              return (
                <button
                  key={t.id}
                  ref={(el) => {
                    tabRefs.current[t.id] = el;
                  }}
                  type="button"
                  role="tab"
                  id={`tab-${t.id}`}
                  aria-selected={selected}
                  aria-controls={selected ? `panel-${t.id}` : undefined}
                  tabIndex={selected ? 0 : -1}
                  onClick={() => select(t.id)}
                  onKeyDown={(e) => onTabKeyDown(e, i)}
                  className={`min-h-[48px] shrink-0 whitespace-nowrap border-b-2 px-4 text-sm transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-inset focus-visible:ring-teal/30 ${
                    selected
                      ? 'border-teal font-medium text-teal'
                      : 'border-transparent text-muted hover:text-ink'
                  }`}
                >
                  {t.label}
                </button>
              );
            })}
          </div>

          <div className="order-2 flex items-center gap-2 lg:order-3">
            <AdminTokenControl />
            <a
              href={USER_APP_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-h-[44px] items-center rounded-lg bg-teal px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-teal-dark focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-teal/30"
            >
              Open End-User App
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6 sm:pt-12">
        <div
          role="tabpanel"
          id={`panel-${active.id}`}
          aria-labelledby={`tab-${active.id}`}
          tabIndex={IMPLEMENTED.includes(activeId) ? undefined : 0}
          className="space-y-8 outline-none"
        >
          {activeId === 'dataset' && (
            <>
              <Hero />
              <MetricCards />
              <DatasetExplorer />
            </>
          )}
          {activeId === 'pipeline' && <PipelineLeakage />}
          {activeId === 'evaluation' && <ModelEvaluation />}
          {activeId === 'registry' && <ModelRegistryReports />}
          {!IMPLEMENTED.includes(activeId) && <TabPlaceholder ticket={active.ticket} title={active.label} />}
        </div>
      </main>
    </>
  );
}

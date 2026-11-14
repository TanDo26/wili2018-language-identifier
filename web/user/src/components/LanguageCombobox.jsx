import { useId, useState } from 'react';
import { languageLabel, searchLanguages } from '../lib/languages.js';

export default function LanguageCombobox({
  id,
  inputRef,
  languages = [],
  query = '',
  onQueryChange,
  onPick,
  disabled = false,
  invalid = false,
  describedBy,
  placeholder = 'Search by name or WiLI code (e.g. Cantonese, zh-yue)...',
}) {
  const listId = useId();
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);

  const matches = open && !disabled ? searchLanguages(languages, query, 6) : [];
  const showList = open && !disabled && matches.length > 0;

  const selectItem = (lang) => {
    onPick(lang);
    setOpen(false);
    setActiveIdx(-1);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) setOpen(true);
      setActiveIdx((i) => (matches.length ? (i + 1) % matches.length : -1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((i) => (matches.length ? (i <= 0 ? matches.length - 1 : i - 1) : -1));
    } else if (e.key === 'Enter' && showList && activeIdx >= 0) {
      e.preventDefault();
      selectItem(matches[activeIdx]);
    } else if (e.key === 'Escape' && open) {
      e.preventDefault();
      setOpen(false);
      setActiveIdx(-1);
    }
  };

  return (
    <div className="relative mt-2">
      <input
        id={id}
        ref={inputRef}
        type="text"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={showList && activeIdx >= 0 ? `${listId}-${activeIdx}` : undefined}
        aria-invalid={invalid}
        aria-describedby={describedBy}
        autoComplete="off"
        autoCapitalize="off"
        spellCheck={false}
        disabled={disabled}
        value={query}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          setOpen(false);
          setActiveIdx(-1);
        }}
        onChange={(e) => {
          onQueryChange(e.target.value);
          setOpen(true);
          setActiveIdx(-1);
        }}
        onKeyDown={handleKeyDown}
        placeholder={disabled ? 'Language selection disabled' : placeholder}
        className={`block min-h-[44px] w-full rounded-xl border bg-surface px-3 text-base text-ink outline-none transition-colors placeholder:text-muted/80 focus:ring-4 disabled:cursor-not-allowed disabled:bg-subtle disabled:opacity-60 sm:text-sm ${
          invalid
            ? 'border-red focus:ring-red/25'
            : 'border-field focus:border-teal focus:ring-teal/25'
        }`}
      />

      <ul
        id={listId}
        role="listbox"
        aria-label="Matching languages"
        hidden={!showList}
        className="mt-1 max-h-48 overflow-y-auto rounded-xl border border-border bg-surface shadow-sm"
      >
        {matches.map((lang, i) => (
          <li
            key={lang.code}
            id={`${listId}-${i}`}
            role="option"
            aria-selected={i === activeIdx}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => selectItem(lang)}
            onMouseEnter={() => setActiveIdx(i)}
            className={`flex min-h-[44px] cursor-pointer items-center justify-between px-3 text-sm text-ink ${
              i === activeIdx ? 'bg-subtle font-medium' : ''
            }`}
          >
            <span>{languageLabel(lang)}</span>
            <span className="font-mono text-xs text-muted">{lang.code}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
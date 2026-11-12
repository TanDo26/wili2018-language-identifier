# WiLI-2018 End-User App (`web/user/`)

React + Vite + Tailwind CSS front end for the language identification API: paste text, get the
detected language (with confidence, explanation and candidates), and report wrong predictions.

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api to the backend
npm test           # Vitest unit + component tests
npm run lint       # ESLint (incl. eslint-plugin-react-hooks)
npm run build      # production build in dist/
```

Node 18+ is required. Fonts (DM Sans, DM Mono) are bundled through `@fontsource`, so the app renders
the same offline; there is no Google Fonts request.

## Configuration

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `VITE_API_TARGET` | `npm run dev` (Vite proxy) | `http://localhost:8000` | Where the dev server forwards `/api/*` and `/health`. Point it at a Prism mock with `VITE_API_TARGET=http://localhost:4010 npm run dev`. |
| `VITE_API_BASE_URL` | the built app (`fetch`) | `''` (same origin) | Absolute API origin for production builds, e.g. `https://api.example.com`. Leave empty when a reverse proxy serves app and API from one origin. |

Put them in `.env.local` (git-ignored) or pass them inline.

## API contract used

| Call | Purpose |
| --- | --- |
| `POST /api/v1/predict` `{ text }` | Returns `PredictResponse` including `prediction_id` and `model_version`. |
| `GET /api/v1/languages` | `[{ code, name }]` for all 235 WiLI labels (single source of truth for the language picker). |
| `POST /api/v1/report` | `{ prediction_id, reason, suggested_language, comment? }`. The server looks up the text and predicted language from `prediction_id`, so they are not sent. |

`reason` is one of `wrong_language`, `mixed_languages`, `not_a_language`, `other`.
`suggested_language` is a WiLI code or `null` ("I don't know"). A `404` on report means the
`prediction_id` is unknown or expired and the dialog asks the user to run a new identification.

### Input length

Text is validated as **trimmed Unicode code points** (`[...value.trim()].length`, 20 to 5,000),
which matches Python's `len(text.strip())` on the server. Emoji count as one character.

### Language list

`useLanguages()` loads `GET /api/v1/languages` once, caches it in memory and `sessionStorage`, and
falls back to `src/data/languages.fallback.json` when the API is unreachable. Picks are validated
strictly against this list (hyphenated labels such as `zh-yue` and `be-tarask` are valid; arbitrary
3-letter codes are not).

To regenerate the fallback from the dataset's labels file:

```bash
npm run gen:languages                  # reads ../../data/processed/labels.csv
npm run gen:languages path/to/labels.csv
```

It reads the `Label` (code) and `English` (name) columns and keeps the existing file if the CSV is
missing. The committed fallback holds the 235 WiLI codes; its names come from the ISO 639-3
registry, so regenerate it from `labels.csv` to use the dataset's own names.

## Project layout

```
src/
  App.jsx                    page shell, API calls, error messages
  components/
    TextInputCard.jsx        input + live validation (getStatus, countChars)
    ResultCard.jsx           prediction, model version badge, explanation, candidates
    ReportDialog.jsx         accessible report dialog (validateReportForm)
    LanguageCombobox.jsx     searchable language picker
  hooks/useLanguages.js      GET /languages + cache + fallback
  lib/languages.js           pure search / strict-resolve helpers
  data/languages.fallback.json
  __tests__/                 Vitest tests
```

# app

Expo (React Native + TypeScript) front-end for the
[risk-return-analysis](../risk-return-analysis) simulation pipeline.

The app does not run any model itself. It reads pre-generated JSON files in
[`assets/data/`](assets/data) and renders them. To refresh the data, re-run
the Python pipeline (see [Regenerating data](#regenerating-data) below).

## Running

```bash
npm install
npx expo start
```

Scan the QR code with the Expo Go app on your phone, or press `i` / `a` for
the iOS / Android simulator. Designed and tested in Expo Go — no custom
native code.

### If `expo start` fails with `TypeError: fetch failed`

Expo CLI runs a preflight that fetches a native-module compatibility
table from Expo's API (`getNativeModuleVersionsAsync`). If your laptop
can't reach that API (captive portal, VPN, flaky DNS, registry hiccup),
the CLI errors out before Metro even starts. The dev server itself does
not need network — the app reads bundled JSON from `assets/data/`. Start
in offline mode to skip the preflight:

```bash
npx expo start --offline
# or
npm run start:offline
```

Append `--clear` to either command to flush the Metro bundler cache
(e.g. `npm run start:offline -- --clear`). Once Metro is up, Expo Go on
your phone connects over the LAN as usual.

## What the app shows

1. **Onboarding** — pick a risk profile (Conservative / Balanced /
   Aggressive), enter a starting capital, and choose whether to enable the
   AI tilt. Default is Balanced + tilt off.
2. **Dashboard** tab — portfolio value chart vs SPY buy-and-hold,
   headline metrics (annualized return, max drawdown, Sharpe, win rate),
   current allocation bar.
3. **Decisions** tab — list of all eight quarterly rebalance events. Tap
   any row to drill into that day.
4. **Date detail** — for the selected rebalance, shows the human-readable
   rationale, asset-class allocation, AI tilt detail (if enabled), and the
   list of trades (scaled to your selected capital).
5. **About** tab — describes what the app does, lists the active
   simulation parameters, exposes a switch for the AI tilt, links back to
   onboarding to change the profile or capital.

A "Simulation only — not investment advice" banner appears throughout.

## Architecture

```
app/                       Expo Router file-based routes
  _layout.tsx              wraps everything in <AppProvider>; declares the
                           Stack with (tabs), onboarding, event/[date]
  onboarding.tsx           profile + capital + tilt onboarding
  event/[date].tsx         rebalance detail screen
  (tabs)/
    _layout.tsx            three tabs: Dashboard, Decisions, About
    index.tsx              Dashboard
    timeline.tsx           Decisions
    about.tsx              About / settings

components/
  AllocationBar.tsx        horizontal asset-class breakdown
  DisclaimerBanner.tsx     "not investment advice" banner
  MetricCard.tsx           reusable metric tile + grid
  PortfolioChart.tsx       react-native-gifted-charts line chart with benchmark
  ProfileOption.tsx        selectable profile card for onboarding
  RebalanceEventRow.tsx    timeline list row
  themed-text.tsx          (template) light/dark-aware Text
  themed-view.tsx          (template) light/dark-aware View
  ui/icon-symbol.tsx       SF Symbols on iOS, Material Icons elsewhere

data/
  types.ts                 TypeScript mirror of the JSON schema
  loadData.ts              static import of summary + 6 detail files

state/
  AppContext.tsx           profile / capital / tilt + AsyncStorage persistence

lib/
  format.ts                currency, percent, date helpers; `scaleFromBase`
                           for $1,000 → user-capital scaling
  palette.ts               profile colors + per-asset-class colors

assets/
  data/                    JSON produced by ../risk-return-analysis/main.py
  images/                  app icon, splash, adaptive icons
  fonts/                   bundled fonts
```

## Data flow

```
risk-return-analysis/main.py
   ↓ writes JSON
app/assets/data/{summary.json, balanced_no_tilt.json, ...}
   ↓ static `import` in data/loadData.ts
   ↓ Metro bundles at build time
loadProfileDetail(profileKey, useTilt)  →  ProfileDetail object used by screens
```

All dollar values in the JSON are stored against a $1,000 starting-capital
base. The app multiplies by `(userCapital / 1000)` everywhere it displays a
dollar — see `scaleFromBase` in `lib/format.ts`. Returns and percentages are
already relative so they need no scaling.

## Regenerating data

If you change the strategy, profiles, or universe in `risk-return-analysis/`,
re-run the pipeline to refresh the JSON the app reads:

```bash
cd ../risk-return-analysis
python main.py
```

The pipeline writes directly to `app/assets/data/`. Restart the Expo dev
server (or `r` to reload) to pick up the new bundle.

## TypeScript notes

- `tsconfig.json` configures the `@/` path alias relative to `app/`.
- Expo Router's `typedRoutes: true` autogenerates `.expo/types/router.d.ts`
  on dev-server startup. If you see TS errors about route names being
  unknown, restart `expo start` so the file is regenerated.
- JSON imports are typed via `as unknown as Summary | ProfileDetail` casts
  in `data/loadData.ts` — the schema contract lives in `data/types.ts` and
  must stay in sync with the Python export module.

## Dependencies worth knowing about

- **`react-native-gifted-charts`** — chart library, requires `react-native-svg` (already in Expo) and `expo-linear-gradient` (for area-fill rendering)
- **`expo-linear-gradient`** — used implicitly by gifted-charts when `areaChart` is on
- **`@react-native-async-storage/async-storage`** — persists onboarding choice
- **`expo-router`** — file-based routing
- **`react-native-safe-area-context`** — top/bottom inset handling

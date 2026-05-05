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
   Aggressive), enter a starting capital, optionally configure recurring
   contributions (amount + period of 1/2/3/6/12 months + day-of-month
   1/15/EOM), and choose whether to enable the AI tilt. Default is
   Balanced + no recurring contribution + tilt off.
2. **Dashboard** tab — portfolio value chart vs SPY buy-and-hold,
   headline metrics (annualized return, max drawdown, Sharpe, win rate),
   current allocation bar. When recurring contributions are on, the
   dashboard also shows total contributed alongside final value.
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
  AppContext.tsx           profile / capital / tilt / deposit schedule +
                           AsyncStorage persistence (storage key v2)

lib/
  format.ts                currency, percent, date helpers; `scaleFromBase`
                           for $1,000 → user-capital scaling (lump-sum mode)
  deposits.ts              TS port of get_deposit_dates: 1st/15th snap forward,
                           EOM snaps backward to the last trading day
  reconstructPortfolio.ts  walks the canonical daily series + deposit schedule
                           to produce a deposit-aware dollar trajectory; also
                           exports `twrTotalReturn` and `scaleAtDate`
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

All dollar values in the JSON are stored against a **$1,000 starting-capital,
no-deposits canonical baseline**. The app derives display dollars from it two
ways:

- **Lump-sum mode** (no recurring contributions): linearly scales each value
  by `(userCapital / 1000)` — see `scaleFromBase` in `lib/format.ts`.
- **Deposit mode**: `lib/reconstructPortfolio.ts` walks the daily series,
  applying the cashflow-adjusted `daily_return` then injecting the
  configured deposit on each scheduled trading day. Per-trade dollar values
  are scaled to the user's portfolio value at the trade date via
  `scaleAtDate`. The headline return is time-weighted (`twrTotalReturn`),
  identical to the lump-sum total return when no deposits are active.

Honest approximation: the canonical baseline assumes 100% deployment, so
deposit-mode reconstructed returns slightly overstate the cash-drag between
quarterly rebalances. For exact numbers, run the Python pipeline with the
`--deposit-amount` flag and read the printed metrics.

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

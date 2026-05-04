# ai-trading-portfolio

A two-part project that explores how a beginner-friendly investment app could
sit on top of a real ML-driven trading model.

## What's in here

```
ai-trading-portfolio/
├── risk-return-analysis/   Python research pipeline:
│                           ML training, profile-based simulation, JSON export
└── app/                    Expo (React Native) app that visualises the
                            simulation on your phone
```

The two halves are loosely coupled: the Python pipeline runs end-to-end and
writes JSON files into `app/assets/data/`. The app reads those JSON files at
build time — no backend, no network calls, no live model inference.

## What it does

Three risk profiles — **Conservative**, **Balanced**, **Aggressive** — each
defined by a fixed cross-asset allocation (US stocks, long-term Treasuries,
S&P 500 ETF, gold, international equities). The portfolio rebalances back to
those targets every quarter. Optionally, an "AI tilt" nudges individual
stock weights inside the equity sleeve by up to ±5% based on a machine-
learning return signal.

The app simulates **what would have happened** to your chosen capital under
each profile from 2024-01-02 through 2025-12-31, scaled linearly from a
$1,000 base simulation.

### Backtest results (2024-01-02 → 2025-12-31)

| Profile      | AI tilt | Total return | Annualized | Sharpe | Max drawdown |
|--------------|:-------:|-------------:|-----------:|-------:|-------------:|
| Conservative | off     | 27.77%       | 13.09%     | 1.17   | -10.37%      |
| Conservative | on      | 28.83%       | 13.56%     | 1.17   | -12.04%      |
| Balanced     | off     | 59.65%       | 26.47%     | 1.58   | -16.37%      |
| Balanced     | on      | 60.44%       | 26.78%     | 1.58   | -17.05%      |
| Aggressive   | off     | 85.13%       | 36.23%     | 1.60   | -20.87%      |
| Aggressive   | on      | 85.68%       | 36.43%     | 1.61   | -21.53%      |
| _SPY buy & hold (benchmark)_ | — | _47.84%_ | _21.69%_ | — | _-19.43%_ |

Read those numbers honestly: the test window (2024–2025) was a strong bull
market, and a single such period does not validate a strategy. The AI tilt
adds ~0.5–1% return but also slightly increases drawdown — its risk-adjusted
contribution is essentially neutral. The honest story here is what the
profiles do (diversification, discipline, quarterly rebalancing), not what
the ML model adds.

## Quick start

### 1. Generate the simulation data

```bash
cd risk-return-analysis
pip install -r ../requirements.txt
python main.py
```

This trains the ML models, runs all six profile/tilt combinations, and
writes JSON into `../app/assets/data/`. Runtime is ~5–10 minutes, mostly
spent training the models.

### 2. Run the app

```bash
cd ../app
npm install
npx expo start
```

Open the resulting QR code in Expo Go on your phone, or press `i` / `a` for
the iOS / Android simulator.

For more detail on each half:
- [`risk-return-analysis/README.md`](risk-return-analysis/README.md) — pipeline internals, model selection, profile definitions
- [`app/README.md`](app/README.md) — app structure, screens, data flow

## Disclaimer

**This is a simulation, not investment advice.** Past performance does not
predict future results. The results above come from a single backtest window
on a small ticker universe. Do not treat any number in this repo as a
forecast of your real portfolio's behavior.

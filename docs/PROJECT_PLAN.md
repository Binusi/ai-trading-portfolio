# Project Plan: From Simulation to Live Advice App

> **Living document.** Update this each session. The version in `/Users/user1/.claude/plans/` was the initial draft; this file is now the source of truth.
>
> Last updated: 2026-05-13 (session 1 — initial scaffolding).

---

## Context

The repo today is a **portfolio simulation**: a Python ML pipeline (`risk-return-analysis/`) trains models, runs quarterly-rebalance backtests across 3 risk profiles, and exports static JSON consumed by an Expo mobile app (`app/`). Everything is offline — no auth, no database, no live inference, no notifications. The `regular-payments-simulation` branch recently added recurring-contribution support (see `risk-return-analysis/src/deposits.py` and `app/lib/reconstructPortfolio.ts`).

We are turning this into a **real product**: a mobile app where users sign in, declare initial capital + recurring deposit cadence, and receive **daily buy/sell/hold advice with rationale**. **Not** an actual trading platform — advisory only, no broker, no real money movement in MVP. Target infrastructure: AWS, free-tier / minimal cost (< $20/mo while userbase is small). Notifications via push + email (SES).

This plan is the architectural and milestone source of truth. The companion file `HANDOVER.md` is the **session-state log** — what was done, what's pending, what's broken — and is updated continuously so work survives interrupted sessions.

---

## MVP Vision (Phase 1 launch)

A beginner-friendly daily-advice app. A user signs up, picks a risk profile, sets initial capital + recurring contribution, and each morning gets:

1. A short list: **what to buy, what to sell, what to hold** (with target dollar amounts).
2. Plain-English **rationale** ("Why NVDA today?") tied to model signals — no jargon by default, with optional "show me the math".
3. A **confidence indicator** (Strong / Moderate / Weak) per pick.
4. Their **simulated portfolio value** as if they had followed advice.
5. **Push notification** on iOS/Android and an **email digest** with the day's calls.

**Explicitly out of MVP:** live broker, real money, social features, alternative assets, custom universes.

---

## Beginner-Helpful Features

Low-cost product moves that turn a signals app into something a beginner can actually learn from. These should be considered through Phases 1–4.

- **"Why this pick?" card** — every recommendation explains its top 2-3 features in plain English ("Momentum is strong, valuation is reasonable, sector rotation favors tech").
- **Confidence + dissent** — show when the model is unsure or when the 7 model families disagreed. Teaches users that signals aren't certainty.
- **Learning nudges** — a rotating "Term of the day" card pulled from `LEARNING.md`.
- **Cool-off / loss-aversion guardrails** — if a user *manually overrides* advice (post-MVP), surface a "are you sure?" with a 24h reflection nudge.
- **Diary mode** — let the user note *why* they ignored or followed advice. Builds a personal log.
- **Backwards lens** — "If you had followed every call for the last 30 days, you'd be up $X."
- **Risk profile re-confirmation** every quarter — beginners often pick "Aggressive" then panic in drawdowns.
- **Plain-English disclaimers** at sign-up *and* on every advice card. Short and blunt: "This is not financial advice. You can lose money."

---

## Target Architecture (AWS, serverless-first)

```
┌──────────────────────────┐
│  Expo Mobile App         │   iOS + Android (Phase 1)
│  (existing app/)         │   + Expo Push Notifications
└────────────┬─────────────┘
             │ HTTPS (JWT from Cognito)
             ▼
┌──────────────────────────┐
│  Amazon Cognito          │   User pool: email/password + Apple/Google SSO later
│  (auth, JWT issuer)      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  API Gateway (HTTP API)  │   /me, /advice/today, /history, /profile
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Lambda (Python)         │   Per-route handlers, shared layer for model+util
│  - api_handler           │
│  - daily_inference (cron)│   EventBridge schedule: every weekday 06:00 ET
│  - notifier              │   Fires push + email after inference
└────┬────────────┬────────┘
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────┐
│DynamoDB │  │   S3         │   Model artifacts (.pkl), exported JSON snapshots
│(users,  │  │   ml-models/ │
│ advice, │  │   exports/   │
│ devices)│  └──────────────┘
└─────────┘
     ▲
     │
┌────┴───────────────────────────────┐
│  SES (email digest)                │   Free tier 62k/mo
│  Expo Push (push notifications)    │   Free, swap for APNs/FCM later if needed
└────────────────────────────────────┘
```

**Why this shape:** see [`decisions/0001-aws-serverless-stack.md`](decisions/0001-aws-serverless-stack.md). Component contracts: see [`ARCHITECTURE.md`](ARCHITECTURE.md).

**Market data:** Production cannot rely on `yfinance` long-term (unofficial scraping). Migrate to **Alpha Vantage free tier** or **Polygon.io free tier** during Phase 1. Keep `yfinance` as the dev-pipeline fallback.

---

## Repo Layout (target)

```
ai-trading-portfolio/
├── docs/                      # Living project docs (this directory)
│   ├── PROJECT_PLAN.md
│   ├── HANDOVER.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   └── decisions/
│       └── 0001-aws-serverless-stack.md
├── app/                       # Expo mobile (existing — may rename to mobile/ later)
├── backend/                   # NEW — AWS Lambda functions
├── infra/                     # NEW — AWS CDK (TypeScript)
├── risk-return-analysis/      # ML pipeline (existing — may rename to ml-pipeline/)
├── scripts/                   # NEW — developer utilities
├── .github/workflows/         # NEW — CI/CD
├── README.md
├── LEARNING.md
└── requirements.txt           # will split into per-component requirements later
```

Renames (`app/` → `mobile/`, `risk-return-analysis/` → `ml-pipeline/`) are deferred until Phase 0 implementation — they touch many imports and aren't worth the churn until we're ready to commit to the new structure.

---

## Phased Roadmap (summary)

| Phase | Goal | Duration |
|-------|------|----------|
| **0. Foundations** | AWS + repo ready to build on | 1-2 weeks |
| **1. Auth + onboarding** | Users can sign up and store profile | 1-2 weeks |
| **2. Daily advice engine** | Backend produces signals daily | 2-3 weeks |
| **3. Notifications** | Push + email reach users | 1 week |
| **4. Beta polish** | Ready for TestFlight / Play Store internal | 1-2 weeks |
| **5. Public soft launch** | Real users on internal tracks | — |
| **6+. Post-MVP** | Compounding features | ongoing |

Full milestone detail with checkboxes: [`ROADMAP.md`](ROADMAP.md).

---

## Critical Open Questions

Tracked here until resolved. When a decision is made, move it to an ADR under `decisions/`.

1. **AWS account & email domain** — which account hosts the app? SES requires a verified domain for the "from" address.
2. **Apple/Google developer accounts** — $99/yr (Apple) + $25 one-time (Google). Needed for TestFlight / Play Store.
3. **Market data licensing** — Polygon / Alpha Vantage free tiers are fine for dev; check ToS before commercial launch.
4. **Regulatory disclaimer wording** — in many jurisdictions, specific buy/sell advice can require registration. MVP framing: "Educational / model output, not financial advice." Worth a lawyer skim before public launch.
5. **Model retrain cadence** — MVP: frozen models with daily feature recompute only. Retrain pipeline is Phase 6.

---

## Key Files We Will Reuse

- `risk-return-analysis/main.py` — pipeline entry. Live inference will reuse data-fetch + predict stages.
- `risk-return-analysis/src/profile_strategy.py` — risk profile → target allocation. Server-side reuse.
- `risk-return-analysis/src/deposits.py` — `DepositSchedule`. Server-side reuse for projected value.
- `risk-return-analysis/src/export_app_data.py` — JSON output schema. Live API responses should mirror these shapes.
- `app/state/AppContext.tsx` — currently AsyncStorage-only; in Phase 1 becomes a thin wrapper around `/me`.
- `app/lib/reconstructPortfolio.ts` — keep client-side for now; eventually move server-side.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `yfinance` scraper breaks in prod | Migrate to Alpha Vantage or Polygon by Phase 2; cache last-known-good in S3 |
| Lambda cold-start for ML inference | Daily-cron Lambda has no UX impact; precompute & cache for API paths |
| Apple App Store rejects "stock advice" app | Strong "Educational / Not Financial Advice" framing; consult Apple guideline 5.0 before submit |
| Beginner follows bad advice and loses money | In-app disclaimers, confidence indicators, no auto-execution, encourage paper-tracking before real trades |
| AWS bill surprise | Billing alarms at $5/$10/$20; tag all resources; CDK makes teardown easy |
| Model drift (trained 2015-2023, deployed 2026) | Manual retrain job in Phase 6; monitor live signal quality via CloudWatch |

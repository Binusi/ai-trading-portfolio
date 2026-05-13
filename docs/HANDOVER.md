# Handover Log

> **Read this first** at the start of every session. This file tracks **state**, not plans. The plan lives in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
>
> **Updating rules:**
> 1. Update as you work, not just at the end — sessions can end abruptly.
> 2. The **top entry** is the current/most-recent session. Older sessions append below.
> 3. If you change a file, list it in "Files touched". If you tried something that didn't work, write it under "What failed" so the next session doesn't repeat it.
> 4. End every session by filling in "Next up" so the next session has a starting line.

---

## Session 1 — 2026-05-13 — Initial scaffolding

**Goal:** Produce a project plan for turning the simulation into a live AWS-hosted advice app, and create the scaffolding files the project will grow into.

**Branch:** `regular-payments-simulation` (working off the most recent merge of the recurring-contributions feature).

### Current state of the project
- Codebase is still 100% simulation: Python ML pipeline (`risk-return-analysis/`) + Expo mobile app (`app/`) reading static JSON.
- No backend, no auth, no DB, no live inference, no notifications.
- AWS infra: **not yet provisioned**. No AWS account work has been done in this session.

### Decisions made this session
- **Platform**: Mobile only (iOS + Android via Expo) for Phase 1.
- **MVP scope**: Advice-only (signals + notifications). No broker, no real money.
- **Notifications**: Push (Expo Push) + Email (SES).
- **Cost target**: Free-tier / minimal (< $20/mo while < 100 users) — drove the serverless-first architecture.
- **Architecture**: Cognito + API Gateway + Lambda + DynamoDB + EventBridge + S3 + SES + Expo Push. Detail in [`ARCHITECTURE.md`](ARCHITECTURE.md), rationale in [`decisions/0001-aws-serverless-stack.md`](decisions/0001-aws-serverless-stack.md).
- **Renames deferred**: `app/` → `mobile/` and `risk-return-analysis/` → `ml-pipeline/` not done yet; will happen in Phase 0 implementation when we're committed.

### Files touched this session
- `docs/PROJECT_PLAN.md` — created (living plan).
- `docs/HANDOVER.md` — created (this file).
- `docs/ARCHITECTURE.md` — created (AWS diagram + component contracts).
- `docs/ROADMAP.md` — created (phased milestone checklist).
- `docs/decisions/0001-aws-serverless-stack.md` — created (first ADR).
- `backend/README.md` — created (stub).
- `infra/README.md` — created (stub).
- `scripts/README.md` — created (stub).
- `.github/workflows/ci.yml` — created (CI stub — lint, type-check, tests; no deploy yet).
- `README.md` — updated with a "Going Live" section pointing at `docs/`.

### What worked
- AskUserQuestion to pin down platform / scope / notifications / budget before designing — gave clear constraints (mobile-only, advice-only, push+email, free tier) which collapsed the design space cleanly.
- Existing repo had enough structure that the scaffolding fits naturally beside it without disturbing the simulation pipeline or mobile app.

### What failed / pitfalls noted
- None this session — pure scaffolding, no code execution.
- **Pitfall to watch in next session**: `yfinance` is fragile for production. Don't build the live inference Lambda on top of it. Plan for Alpha Vantage / Polygon migration during Phase 2.

### Open questions still unresolved
- Which AWS account will host this?
- Does the user have an email domain for SES "from" address?
- Apple ($99/yr) + Google ($25) developer accounts not yet set up.
- Regulatory disclaimer wording needs a lawyer skim before public launch.

### Next up (next session starts here)
1. **Phase 0 step 1**: Set up the AWS account hardening checklist — root MFA, IAM admin user for dev work, billing alarms at $5/$10/$20.
2. **Phase 0 step 2**: Bootstrap CDK in `infra/`. Use TypeScript. First stack: `DataStack` with a single DynamoDB table (single-table design) and S3 bucket. Goal is `cdk synth` succeeds locally.
3. **Phase 0 step 3**: Make the GitHub Actions CI stub actually run (right now it's a stub — needs lint/test commands wired up).
4. **Phase 0 step 4**: Decide on the rename. Either commit to `app/` → `mobile/` + `risk-return-analysis/` → `ml-pipeline/` now (Phase 0 is the cheapest time to do it) or formally defer.
5. Update `HANDOVER.md` with a new "Session 2" block as soon as the next session starts.

### Notes for future-me
- This handover log is append-only at the top. Don't edit past sessions retroactively — if something was wrong, note it in the current session.
- When a decision is finalized that affects architecture, write an ADR under `docs/decisions/`. The handover log captures **what** changed; ADRs capture **why**.
- The plan file at `/Users/user1/.claude/plans/we-have-started-an-humble-sutton.md` was the initial draft. The canonical version lives at `docs/PROJECT_PLAN.md`.

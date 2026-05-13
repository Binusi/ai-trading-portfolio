# Roadmap

> Phased milestones for getting from simulation to live app. Check items off as they land.
> Approval to advance to the next phase requires the **verification** item to pass.
>
> Plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). Session state: [`HANDOVER.md`](HANDOVER.md).

---

## Phase 0 — Foundations (1-2 weeks)

**Goal:** AWS account and repo are ready to build on.

- [ ] AWS account hardening: root user MFA, billing alarms ($5/$10/$20), separate IAM admin user for dev
- [ ] Domain registered (or chosen) for the SES "from" address
- [ ] CDK bootstrap in `infra/` (TypeScript)
- [ ] First CDK stack: `DataStack` — DynamoDB table + S3 buckets — `cdk synth` succeeds
- [ ] GitHub Actions CI wired up: lint + tests run on PR
- [ ] Repo restructure decision: commit to `app/`→`mobile/`, `risk-return-analysis/`→`ml-pipeline/` (or formally defer)
- [ ] Apple Developer account ($99/yr) created
- [ ] Google Play Developer account ($25 one-time) created

**Verification:** `cdk synth` succeeds, CI passes on a PR, billing alarms visible in console.

---

## Phase 1 — Auth + Onboarding (1-2 weeks)

**Goal:** Users can sign up and store their profile server-side.

- [ ] `AuthStack` CDK: Cognito User Pool + App Client
- [ ] `ApiStack` CDK: HTTP API + `/me` route + JWT authorizer
- [ ] `api_handler` Lambda: `GET /me`, `PUT /me`
- [ ] Expo app: signup screen, login screen, JWT storage in `expo-secure-store`
- [ ] Expo app: onboarding values (risk, capital, deposits) write through `/me`
- [ ] `AppContext` refactored to call API; AsyncStorage demoted to cache

**Verification:** New email signs up via the app, logs in, sets profile, force-quits, reopens — profile is still there and pulled from the API.

---

## Phase 2 — Daily Advice Engine (2-3 weeks)

**Goal:** Backend produces signals daily and serves them to the app.

- [ ] `ml-pipeline/live_inference.py` — reuses data fetch + predict, skips training
- [ ] Models packaged and uploaded to `app-ml-models` S3 bucket
- [ ] `daily_inference` Lambda: load models, fetch latest data, write advice rows
- [ ] EventBridge schedule rule (weekday 06:00 ET)
- [ ] Migrate market data source from `yfinance` to **Alpha Vantage** or **Polygon** (free tier)
- [ ] API: `GET /advice/today`, `GET /advice/history`
- [ ] Expo app: advice card screen (buy/sell/hold list with rationale & confidence)
- [ ] Disclaimer banner on advice card

**Verification:** EventBridge fires, advice row appears in DynamoDB, `/advice/today` returns it, mobile app displays it.

---

## Phase 3 — Notifications (1 week)

**Goal:** Users get push and email when daily advice publishes.

- [ ] `POST /devices` to register Expo push tokens
- [ ] `notifier` Lambda: triggered after `daily_inference`; iterates user devices, calls Expo Push HTTP API
- [ ] SES sandbox setup; domain verified
- [ ] HTML + plain-text email template for daily digest
- [ ] User preferences: opt out of push / email separately
- [ ] Welcome email on signup

**Verification:** A test user receives both a push notification and an email within 60s of the scheduled job finishing.

---

## Phase 4 — Beta Polish (1-2 weeks)

**Goal:** Ready for TestFlight + Play Store internal testing.

- [ ] Sentry integrated for mobile crash reports
- [ ] CloudWatch dashboards: API health, daily inference success, push delivery
- [ ] Legal pages: Terms of Service, Privacy Policy, "Not Financial Advice" notice
- [ ] In-app disclaimers (signup, advice card)
- [ ] Beginner-helpful: "Why this pick?" expanded rationale; "Term of the day" card
- [ ] App icon, splash screen, store listing copy + screenshots

**Verification:** Submit build to TestFlight internal. A second user (not the dev) signs up cold and reports their experience.

---

## Phase 5 — Public Soft Launch

**Goal:** Real users on internal testing tracks.

- [ ] Apple App Store internal testing track live
- [ ] Google Play internal testing track live
- [ ] Feedback channel (in-app form → email)
- [ ] Observability dashboards reviewed weekly

**Verification:** 5-10 external testers using the app for at least one full week.

---

## Phase 6+ — Post-MVP

Ideas to consider once MVP is stable. Pull into a real phase when prioritized.

- [ ] Paper-portfolio tracking (DB-backed, follow-the-advice simulation)
- [ ] Manual trade logging + reconciliation
- [ ] Broker integration (Alpaca **paper** API first)
- [ ] Expanded ticker universe (sector ETFs, more individual names)
- [ ] Periodic model retraining pipeline + drift monitoring
- [ ] Weekly performance digest email
- [ ] Streaks / consistency gamification
- [ ] Social: anonymous leaderboard of best-performing risk profiles in the user base
- [ ] Web app (Expo Web or Next.js) — broader reach, no app-store friction

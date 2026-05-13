# Architecture

> Component-level contracts for the live advice app. Updated as decisions land.
> High-level plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Rationale: [`decisions/`](decisions/).

---

## System diagram

```
┌──────────────────────────┐
│  Expo Mobile App         │
│  (iOS + Android)         │
└────────────┬─────────────┘
             │ HTTPS + JWT
             ▼
┌──────────────────────────┐         ┌──────────────────────┐
│  Amazon Cognito          │         │  EventBridge         │
│  (User Pool)             │         │  weekday 06:00 ET    │
└────────────┬─────────────┘         └──────────┬───────────┘
             │                                  │
             ▼                                  ▼
┌──────────────────────────┐         ┌──────────────────────┐
│  API Gateway (HTTP API)  │         │  daily_inference λ   │
│  Cognito JWT authorizer  │         │  (model predict)     │
└────────────┬─────────────┘         └──────────┬───────────┘
             │                                  │
             ▼                                  │
┌──────────────────────────┐                    │
│  api_handler λ           │                    │
└────┬─────────────────────┘                    │
     │                                          │
     │   ┌──────────────────────────────────────┘
     │   │
     ▼   ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  DynamoDB (single table)│     │  S3                      │
│  pk / sk schema         │     │  - ml-models/*.pkl       │
│  - USER#               │     │  - exports/yyyy-mm-dd/   │
│  - ADVICE#yyyy-mm-dd    │     └──────────────────────────┘
│  - DEVICE#              │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│  notifier λ              │
│  - SES email             │
│  - Expo Push HTTP API    │
└──────────────────────────┘
```

---

## Components

### 1. Mobile app (Expo)
- **Where**: `app/` (current code, will migrate to use APIs instead of bundled JSON).
- **Auth**: Cognito hosted UI or `amazon-cognito-identity-js` SDK. JWTs stored in `expo-secure-store`.
- **Push registration**: `expo-notifications` → device token POSTed to `/devices`.
- **State**: `AppContext` becomes a thin wrapper around the API; AsyncStorage used only as offline cache.

### 2. Cognito User Pool
- **Sign-in**: email + password to start. Apple / Google federated identity added before public launch (App Store requires Apple Sign-In if any third-party sign-in is offered).
- **Custom attributes**: `custom:risk_profile`, `custom:initial_capital`. Or move these to DynamoDB and keep Cognito attributes minimal. **Decision: DynamoDB** — easier to evolve.
- **Email verification**: required.
- **Password policy**: 8+ chars, mixed case + digit. No symbol requirement (UX cost > security gain at this scale).

### 3. API Gateway (HTTP API)

| Route | Method | Auth | Lambda | Purpose |
|-------|--------|------|--------|---------|
| `/me` | GET | JWT | api_handler | Return user profile (risk, capital, deposits, prefs) |
| `/me` | PUT | JWT | api_handler | Update user profile |
| `/advice/today` | GET | JWT | api_handler | Today's advice for the user |
| `/advice/history` | GET | JWT | api_handler | Paginated advice history |
| `/devices` | POST | JWT | api_handler | Register push token |
| `/devices/{id}` | DELETE | JWT | api_handler | Unregister push token |
| `/health` | GET | none | api_handler | Liveness probe |

JWT authorizer attached at the route level. CORS open for now (mobile only; revisit if web added).

### 4. Lambda functions

| Function | Runtime | Trigger | Memory | Timeout |
|----------|---------|---------|--------|---------|
| `api_handler` | Python 3.12 | API Gateway | 512 MB | 10 s |
| `daily_inference` | Python 3.12 | EventBridge cron | 2048 MB | 5 min |
| `notifier` | Python 3.12 | DynamoDB stream OR SQS from `daily_inference` | 512 MB | 1 min |

Shared code (DB client, model loader, schemas) packaged as a **Lambda Layer**.

### 5. DynamoDB (single-table)

Table name: `app-data` (TBD).

| pk | sk | Attributes |
|----|----|------------|
| `USER#<userId>` | `PROFILE` | email, risk_profile, initial_capital, deposit schedule, created_at |
| `USER#<userId>` | `DEVICE#<token>` | platform (ios/android), registered_at, last_seen |
| `USER#<userId>` | `ADVICE#<date>#<universe>` | denormalized advice doc (per-user snapshot) |
| `ADVICE#<date>` | `UNIVERSE#<universe>` | source advice doc (one per date+universe, fanned out to users) |

GSI candidates added when query patterns force them. Initial access patterns:
- "Get user profile" → pk=USER#u, sk=PROFILE
- "Get today's advice for user" → pk=USER#u, sk begins_with ADVICE#today
- "List devices for user" → pk=USER#u, sk begins_with DEVICE#

### 6. S3 buckets

| Bucket | Contents | Lifecycle |
|--------|----------|-----------|
| `app-ml-models` | `.pkl` model artifacts (output of pipeline) | versioning on, no expiry |
| `app-exports` | daily JSON snapshots (rationale, feature contributions) | move to IA after 30d, expire after 365d |

### 7. SES (email)
- **Sandbox first** — can only send to verified addresses until graduated.
- **Templates**: HTML + plain text. One template for "Daily digest", one for "Welcome".
- **From address**: needs verified domain. Open question in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

### 8. Expo Push
- HTTP API: `POST https://exp.host/--/api/v2/push/send`.
- Batched — one call per ~100 tokens. The `notifier` Lambda iterates user device tokens.
- No per-message cost. Migration to APNs/FCM direct is possible later but not needed for MVP.

---

## Security & IAM

- **Least-privilege Lambda roles**: each function gets only the DynamoDB / S3 / SES actions it needs. No `*` actions outside CloudWatch Logs.
- **Secrets**: API keys (Polygon, Alpha Vantage) in AWS Secrets Manager, not env vars. Loaded at cold-start.
- **No long-lived AWS keys in mobile**: Cognito JWT → API Gateway → server-side AWS calls. The app never holds an AWS access key.
- **PII**: email is the only PII initially. Encrypted at rest (Cognito + DynamoDB default). No SSN / financial account info ever.

---

## Observability

- **CloudWatch Logs**: every Lambda. Structured JSON via `aws-lambda-powertools`.
- **CloudWatch Metrics**: API 5xx rate, daily-inference success/duration, push delivery success rate.
- **Alarms**: 5xx > 1% / 5min, daily inference failed, billing > $10/$20.
- **Sentry**: added in Phase 4 for mobile crash reporting.

---

## Environments

- **dev** — single CDK stack, deployed from feature branches. Cognito user pool with relaxed password rules.
- **prod** — deployed from `main` after CI passes. Stricter password policy, billing alarms tighter.

CDK contexts: `--context env=dev` vs `--context env=prod`. Stack names suffixed accordingly.

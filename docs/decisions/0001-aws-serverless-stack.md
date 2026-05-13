# ADR 0001 — AWS serverless stack for the live advice app

- **Status:** Accepted
- **Date:** 2026-05-13
- **Decision-makers:** Kumbirai (project owner)

## Context

We are turning the existing simulation repo into a live, mobile-only advice app. Constraints established with the user this session:
- Mobile-only (Expo / iOS + Android) for Phase 1.
- Advice-only MVP — no broker, no real money.
- Notifications via push + email.
- Cost target: free-tier / minimal (< $20/mo while userbase is small).
- Solo developer; ops simplicity is a hard requirement.

We need an architecture that minimizes idle cost, keeps the surface small enough for a solo developer, and can scale to the first few hundred users without re-platforming.

## Decision

Adopt a **serverless-first AWS stack**:

- **Auth:** Amazon Cognito User Pool (free tier 50k MAU).
- **API:** API Gateway HTTP API + Lambda (Python).
- **Database:** DynamoDB single-table design (free tier 25 GB, on-demand pricing).
- **Object storage:** S3 (model artifacts, daily exports).
- **Scheduled jobs:** EventBridge → Lambda.
- **Email:** SES (free tier 62k/mo when sent from EC2/Lambda).
- **Push:** Expo Push (free, vendor-managed).
- **Infrastructure-as-code:** AWS CDK (TypeScript).

## Alternatives considered

### 1. RDS PostgreSQL + ECS Fargate
- **Pros:** Familiar SQL, no single-table-design learning curve, mature ecosystem.
- **Cons:** Minimum ~$15/mo for the smallest RDS instance even when idle; Fargate adds another $10+/mo. Exceeds our "minimal" target before the app even has a user.
- **Verdict:** Rejected for MVP. Migration path stays open if we outgrow DynamoDB later.

### 2. Auth0 / Clerk for auth instead of Cognito
- **Pros:** Better developer experience; cleaner SDKs.
- **Cons:** $25-100/mo above their free tiers. Cognito's UX is rough but its price is zero and it integrates natively with API Gateway authorizers.
- **Verdict:** Rejected on cost grounds. Re-evaluate if Cognito UX becomes a blocker.

### 3. Supabase / Firebase
- **Pros:** Batteries-included, fast to ship MVP.
- **Cons:** Vendor lock-in stronger than AWS; pricing climbs quickly past free tier; user wants AWS specifically.
- **Verdict:** Rejected — user constraint is AWS.

### 4. Pure-Lambda monolith (one function for all routes)
- **Pros:** Slightly simpler deployment.
- **Cons:** All routes share cold-start and timeout config; one bad route can affect others.
- **Verdict:** Rejected. Three purpose-built Lambdas (`api_handler`, `daily_inference`, `notifier`) is the right granularity for this scale.

### 5. SageMaker endpoint for model inference
- **Pros:** Managed, autoscaling inference.
- **Cons:** Minimum cost ~$50/mo for the smallest endpoint, kept warm 24/7. Our use case (one inference per day) doesn't need an always-on endpoint.
- **Verdict:** Rejected. A scheduled Lambda is the right shape; revisit when we need sub-second per-request inference.

## Consequences

### Positive
- Idle cost ≈ $0. Pay-per-use across the board.
- Stack scales to thousands of users without changes — only Lambda concurrency, DynamoDB throughput, and SES sending limits need attention.
- CDK lets us tear down and recreate the whole environment in a single command.
- IAM least-privilege boundaries are natural in this architecture.

### Negative
- **DynamoDB single-table design has a learning curve.** Modeling access patterns up-front is harder than writing SQL ad hoc. Documented in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
- **Cognito UX is rough.** Custom signup flows take more work than with Auth0/Clerk. Acceptable for MVP.
- **Lambda cold starts.** Mitigated by: (a) daily inference Lambda has no UX impact; (b) API Lambda is small (< 10 MB package) and Python 3.12 cold-starts in ~300ms.
- **Migration risk if requirements change.** If the app needs sub-second inference or transactional SQL, parts of this stack would need replacement. Not on the MVP path.

## Notes for future ADRs

Things this ADR does **not** decide and that should get their own ADRs when they come up:
- Market data provider (Alpha Vantage vs Polygon vs paid tier).
- Mobile push approach beyond Expo Push (direct APNs / FCM).
- Model retraining cadence and pipeline.
- When/whether to add a web frontend.

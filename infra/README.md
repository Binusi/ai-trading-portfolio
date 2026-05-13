# infra/

AWS infrastructure-as-code via **AWS CDK** (TypeScript). **Stub** — bootstrap happens in Phase 0.

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the components this provisions, and [`../docs/decisions/0001-aws-serverless-stack.md`](../docs/decisions/0001-aws-serverless-stack.md) for why CDK over Terraform / SAM / raw CloudFormation.

## Planned layout

```
infra/
├── bin/
│   └── app.ts            # CDK app entry point
├── lib/
│   ├── data-stack.ts     # DynamoDB single-table, S3 buckets
│   ├── auth-stack.ts     # Cognito User Pool + App Client
│   ├── api-stack.ts      # API Gateway HTTP API + api_handler λ
│   └── jobs-stack.ts     # EventBridge schedule, daily_inference λ, notifier λ
├── test/
├── cdk.json
├── package.json
└── tsconfig.json
```

## Bootstrap (first time, per AWS account/region)

```bash
cd infra
npm install
npx cdk bootstrap aws://<account-id>/<region>
```

## Deploy

```bash
# dev
npx cdk deploy --all --context env=dev

# prod (from main after CI)
npx cdk deploy --all --context env=prod
```

Stack names will be suffixed with `-dev` / `-prod`.

## Inspect without deploying

```bash
npx cdk synth        # render CloudFormation
npx cdk diff         # diff vs deployed
```

## Teardown

```bash
npx cdk destroy --all --context env=dev
```

S3 buckets default to **retain on stack delete** — wipe them manually if you really want them gone.

## Conventions

- One stack per concern; cross-stack references via stack exports, not by importing classes.
- Resources tagged: `Project=ai-trading-portfolio`, `Env=dev|prod`, `ManagedBy=cdk`.
- Lambda code lives in [`../backend/`](../backend/); CDK packages it via `lambda.Code.fromAsset`.
- All secrets pulled from AWS Secrets Manager, never embedded in CDK code.

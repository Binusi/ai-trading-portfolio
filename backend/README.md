# backend/

AWS Lambda functions for the live advice app. **Stub** — implementation begins in Phase 1.

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the contracts each function must satisfy.

## Planned layout

```
backend/
├── lambdas/
│   ├── api_handler/        # Routed via API Gateway
│   │   ├── handler.py
│   │   ├── routes/
│   │   └── requirements.txt
│   ├── daily_inference/    # EventBridge cron-triggered
│   │   ├── handler.py
│   │   └── requirements.txt
│   └── notifier/           # Fires after daily_inference
│       ├── handler.py
│       └── requirements.txt
├── shared/                 # Packaged as a Lambda Layer
│   ├── db/                 # DynamoDB client + single-table access patterns
│   ├── models/             # Pydantic models for advice / user / device
│   └── observability/      # aws-lambda-powertools setup
└── tests/                  # pytest, unit + integration
```

## Local development (planned)

- **AWS SAM CLI** for local Lambda invocation: `sam local invoke api_handler -e events/get-me.json`
- **moto** for mocking DynamoDB and S3 in unit tests
- **localstack** optional for end-to-end local AWS

## Conventions

- Python 3.12.
- One module per Lambda; shared code lives only in `shared/`.
- `aws-lambda-powertools` for logging (JSON), tracing (X-Ray), metrics.
- Type-hinted everywhere; `mypy --strict` in CI.
- No business logic in `handler.py` — handlers parse/validate, delegate to functions in `routes/` or `shared/`.

## Deployment

The CDK app in [`../infra/`](../infra/) packages and deploys these Lambdas. There is no separate "deploy backend" command — `cdk deploy` does it.

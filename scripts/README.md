# scripts/

Developer utility scripts. **Stub** — populated as needs arise.

Anything more involved than a one-liner that gets re-run more than twice belongs here. Think: seed data, model upload, manual triggers, log tailing.

## Planned scripts

- `seed_dev_user.sh` — create a test Cognito user + sample DynamoDB rows in the dev environment.
- `upload_models.sh` — push freshly trained `.pkl` files to the `app-ml-models` S3 bucket.
- `trigger_daily_inference.sh` — manually invoke the daily inference Lambda for testing.
- `tail_logs.sh` — convenience wrapper around `aws logs tail` for a given Lambda.

## Conventions

- Bash by default; reach for Python if argument parsing or AWS SDK usage gets gnarly.
- Each script starts with a one-line description comment.
- Every script accepts `--help` and prints usage.
- Scripts that mutate AWS prompt for confirmation unless `--yes` is passed.

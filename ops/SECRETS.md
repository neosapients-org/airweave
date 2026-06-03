# Airweave Service — AWS Secrets Manager Keys

All keys below must be populated in the AWS Secrets Manager secret created by Terraform:

- **Dev**: `airweave-svc-dev-use1-shared1-secrets`

Terraform creates the secret shell with only `AWS_REGION` seeded. Populate the rest manually via the AWS Console or the CLI snippet in the [Quick Start](README.md).

## Required Keys

| Key | Description | Example / How to generate |
|-----|-------------|--------------------------|
| `AWS_REGION` | AWS region — seeded by Terraform | `us-east-1` |
| `POSTGRES_PASSWORD` | RDS `airweave` user password | From neo-platform datastore secrets |
| `REDIS_PASSWORD` | Redis AUTH password | `openssl rand -base64 32` |
| `ENCRYPTION_KEY` | AES-256 key for encrypting sensitive fields | `openssl rand -base64 32` |
| `STATE_SECRET` | CSRF / OAuth state secret | `openssl rand -base64 32` |
| `AUTH0_DOMAIN` | Auth0 tenant domain | `neosapients.us.auth0.com` |
| `AUTH0_AUDIENCE` | Auth0 API identifier | e.g. `https://api.airweave.ai` |
| `AUTH0_CLIENT_ID` | M2M client ID for backend → Auth0 Management API | From Auth0 dashboard |
| `AUTH0_CLIENT_SECRET` | M2M client secret | From Auth0 dashboard |
| `OPENAI_API_KEY` | OpenAI API key — used for embeddings (`text-embedding-3-small`) and classification (`gpt-4o-mini`) | From OpenAI platform |
| `SVIX_JWT_SECRET` | Svix webhook JWT secret (min 32 chars) | `openssl rand -base64 32` |
| `SVIX_DB_DSN` | Svix PostgreSQL connection string | `postgresql://airweave:<POSTGRES_PASSWORD>@postgres:5432/svix` |
| `FIRST_SUPERUSER` | Bootstrap admin email address | e.g. `admin@neosapients.ai` |
| `FIRST_SUPERUSER_PASSWORD` | Bootstrap admin password | Secure password, min 12 chars |
| `STORAGE_AWS_BUCKET` | S3 bucket for raw file storage | `airweave-dev-use1-shared1` |
| `POSTHOG_API_KEY` | PostHog project write key | From PostHog project settings |
| `MCP_AUTH0_CLIENT_ID` | Auth0 client ID for the MCP server OAuth flow | From Auth0 dashboard |
| `MCP_AUTH0_CLIENT_SECRET` | Auth0 client secret for the MCP server OAuth flow | From Auth0 dashboard |

## Notes

- All keys are injected into the pod via the `ExternalSecret` resource (sync wave `-1`) before the deployment starts.
- `refreshInterval: 0` means secrets are fetched once at pod start — restart pods after rotating a secret.
- The secret is managed with `ignore_secret_changes = true` in Terraform, so values set manually in the Console are never overwritten by `terraform apply`.

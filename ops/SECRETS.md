# Airweave Service — AWS Secrets Manager Keys

All keys must be populated in the AWS Secrets Manager secret created by Terraform:
- **Dev**: `airweave-svc-dev-use1-shared1-secrets`

The Helm `ExternalSecret` pulls the entire secret via `dataFrom.extract` and injects every key as an env var into the pod. Keys must match the backend env var names in `backend/airweave/core/config/settings.py`.

Non-sensitive config (hosts, ports, feature flags) lives in the Helm ConfigMap (`values.yaml → env:`) — do not duplicate here.

## How to populate

```bash
aws secretsmanager put-secret-value \
  --secret-id airweave-svc-dev-use1-shared1-secrets \
  --secret-string file://secret-values.json
```

## Required secrets

| Key | Description | How to obtain / generate |
|-----|-------------|--------------------------|
| `POSTGRES_PASSWORD` | Password for the `airweave` Postgres user | From the Postgres StatefulSet DBA / datastore secret |
| `ENCRYPTION_KEY` | Fernet key for stored OAuth/source credentials | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `STATE_SECRET` | HMAC secret for OAuth state tokens (CSRF). ≥ 32 chars | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `SVIX_JWT_SECRET` | HMAC secret for Svix webhook JWTs. ≥ 32 chars | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `SVIX_DB_DSN` | Svix Postgres connection string | `postgresql://airweave:<POSTGRES_PASSWORD>@postgres:5432/svix` |
| `REDIS_PASSWORD` | Redis AUTH password | `openssl rand -base64 32` |
| `FIRST_SUPERUSER` | Email of the bootstrap admin user | e.g. `admin@neosapients.ai` |
| `FIRST_SUPERUSER_PASSWORD` | Password for the bootstrap admin | Strong password, min 12 chars |
| `OPENAI_API_KEY` | OpenAI key — embeddings (`text-embedding-3-small`) and classification (`gpt-4o-mini`) | OpenAI dashboard |

## Auth0 (required — `AUTH_ENABLED=true` in values.yaml)

The config validator raises if any of these are empty while `AUTH_ENABLED=true`.

| Key | Description | Where to obtain |
|-----|-------------|-----------------|
| `AUTH0_DOMAIN` | Auth0 tenant domain | Auth0 → Application settings |
| `AUTH0_AUDIENCE` | API identifier / audience | Auth0 → APIs |
| `AUTH0_CLIENT_ID` | SPA application client ID | Auth0 → Applications (SPA) |
| `AUTH0_M2M_CLIENT_ID` | Machine-to-Machine client ID for Management API | Auth0 → Applications (M2M) |
| `AUTH0_M2M_CLIENT_SECRET` | Machine-to-Machine client secret | Auth0 → Applications (M2M) |

> `AUTH0_RULE_NAMESPACE` is already in the ConfigMap (`https://airweave.ai`) — do not add here.

## Conditionally required

| Key | When needed | Notes |
|-----|-------------|-------|
| `ANTHROPIC_API_KEY` | If any Anthropic model is used | Optional |
| `MISTRAL_API_KEY` | Only if `DENSE_EMBEDDER` is switched to `mistral_embed` | Not needed with current OpenAI embedder |
| `FIRECRAWL_API_KEY` | If web-fetch / crawling sources are enabled | Optional |
| `SVIX_AUTH_TOKEN` | Only for hosted Svix — leave empty for embedded Svix | Optional |

## Source-connector OAuth credentials (BYOC)

Add a pair only for the connectors being deployed. Redirect URI must use `API_FULL_URL`.

| Key | Connector | Where to create |
|-----|-----------|-----------------|
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google Drive | console.cloud.google.com → APIs & Services → Credentials |
| `ATLASSIAN_CLIENT_ID` / `ATLASSIAN_CLIENT_SECRET` | Confluence / Atlassian | developer.atlassian.com → OAuth 2.0 (3LO) |
| `SLACK_CLIENT_ID` / `SLACK_CLIENT_SECRET` | Slack | api.slack.com/apps → OAuth & Permissions |

## Generate random secrets in one shot

```bash
python - <<'PY'
import secrets
from cryptography.fernet import Fernet
print("ENCRYPTION_KEY  =", Fernet.generate_key().decode())
print("STATE_SECRET    =", secrets.token_urlsafe(32))
print("SVIX_JWT_SECRET =", secrets.token_urlsafe(32))
PY
```

## Ready-to-fill template (secret-values.json)

```json
{
  "POSTGRES_PASSWORD": "",
  "ENCRYPTION_KEY": "",
  "STATE_SECRET": "",
  "SVIX_JWT_SECRET": "",
  "SVIX_DB_DSN": "postgresql://airweave:<POSTGRES_PASSWORD>@postgres:5432/svix",
  "REDIS_PASSWORD": "",
  "FIRST_SUPERUSER": "",
  "FIRST_SUPERUSER_PASSWORD": "",
  "OPENAI_API_KEY": "",
  "AUTH0_DOMAIN": "",
  "AUTH0_AUDIENCE": "",
  "AUTH0_CLIENT_ID": "",
  "AUTH0_M2M_CLIENT_ID": "",
  "AUTH0_M2M_CLIENT_SECRET": "",
  "ANTHROPIC_API_KEY": "",
  "GOOGLE_CLIENT_ID": "",
  "GOOGLE_CLIENT_SECRET": "",
  "ATLASSIAN_CLIENT_ID": "",
  "ATLASSIAN_CLIENT_SECRET": "",
  "SLACK_CLIENT_ID": "",
  "SLACK_CLIENT_SECRET": ""
}
```

> Remove any key you are not using rather than leaving it blank. Never commit the filled-in `secret-values.json`.

## Do NOT put these in the secret

Already handled elsewhere:

| Value | Where it comes from |
|-------|---------------------|
| `POSTGRES_HOST/PORT/DB/USER/SSLMODE`, `REDIS_HOST/PORT/DB` | ConfigMap |
| `VESPA_*`, `TEMPORAL_*`, `STORAGE_BACKEND`, `STORAGE_AWS_REGION`, `STORAGE_AWS_BUCKET` | ConfigMap |
| `DENSE_EMBEDDER`, `EMBEDDING_DIMENSIONS`, `SPARSE_EMBEDDER`, `CLASSIFICATION_*` | ConfigMap |
| `AUTH_ENABLED`, `AUTH0_RULE_NAMESPACE`, `ENVIRONMENT`, `LOG_LEVEL`, `OTEL_*` | ConfigMap |
| `API_FULL_URL`, `APP_FULL_URL` | ConfigMap |
| `AWS_REGION` | Seeded by Terraform |
| AWS S3 access credentials | Provided via IRSA on the ServiceAccount — no static keys |
| `POSTHOG_API_KEY` | Public key — add to ConfigMap if needed, not a secret |

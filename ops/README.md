# Airweave Service — EKS Operations

Infrastructure-as-code and Kubernetes manifests for deploying Airweave to EKS on Neo Platform.

## Structure

```
ops/
├── SECRETS.md              # All secret keys that must be populated in AWS Secrets Manager
├── argocd/
│   ├── application.yaml            # ArgoCD Application — dev
│   └── application-staging.yaml    # ArgoCD Application — staging
├── helm/
│   └── airweave-svc/
│       ├── Chart.yaml
│       ├── values.yaml                 # Shared defaults (DEV-flavoured — see below)
│       ├── dev-use1-shared1.yaml       # Dev env overrides
│       ├── staging-use1-shared1.yaml   # Staging env overrides
│       └── templates/
│           ├── _helpers.tpl
│           ├── configmap.yaml
│           ├── deployment.yaml
│           ├── external-secrets.yaml
│           ├── hpa.yaml
│           ├── ingress.yaml
│           ├── service.yaml
│           └── serviceaccount.yaml
└── terraform/
    └── aws/
        ├── backend.tf
        ├── main.tf             # ECR + Secrets Manager + S3 + IRSA
        ├── outputs.tf
        ├── providers.tf
        ├── variables.tf
        └── variables/
            ├── dev-use1-shared1.tfvars
            └── staging-use1-shared1.tfvars
```

> ⚠️ **`values.yaml` holds dev-flavoured defaults**, not neutral ones: the dev ECR
> repository, the dev Secrets Manager secret name, the dev S3 bucket and the dev
> Karpenter nodepool. Each env overlay overrides all of them. If you add a new
> environment-specific default to `values.yaml`, mirror it into **every** overlay
> or that environment silently points at dev.

## Environments

| | dev | staging |
|---|---|---|
| EKS cluster | `dev-use1-shared1` | `dev-use1-shared1` (same cluster) |
| Namespace | `airweave-svc` | `staging-airweave-svc` |
| Helm release / Deployment | `neo-airweave-svc-dev-use1-shared1` | `neo-airweave-svc-staging-use1-shared1` |
| Karpenter nodepool | `dev-use1-shared1-generic-arm64-w50` | `staging-use1-shared1-generic-arm64-w50` |
| Node taints to tolerate | `kubernetes.io/arch=arm64` | `kubernetes.io/arch=arm64` **and** `neo/env=staging` |
| ECR repository | `airweave-svc-dev-use1-shared1` | `airweave-svc-staging-use1-shared1` |
| Image tag | `latest` | `staging-latest` |
| Secrets Manager secret | `neo-airweave-svc-dev-use1-shared1-secrets` | `neo-airweave-svc-staging-use1-shared1-secrets` |
| S3 raw-file bucket | `airweave-dev-use1-shared1` | `airweave-staging-use1-shared1` |
| IRSA role | `airweave-svc-dev-use1-shared1-irsa` | `airweave-svc-staging-use1-shared1-irsa` |
| Public entry point (via gateway) | `https://dev-dp.neosapientai.com/v1/meridian` | `https://staging-dp.neosapientai.com/v1/meridian` |
| Post-OAuth frontend redirect | `https://neo-sapients.neosapientai.com/...` | `https://staging.neosapientai.com/...` |
| `ENVIRONMENT` | `dev` | `dev` (see below) |

> **Why staging sets `ENVIRONMENT=dev`.** `platform/auth/settings.py` resolves
> `yaml/<ENVIRONMENT>.integrations.yaml` at import time, and only `dev`, `prd`
> and `self-hosted` ship a file — `staging` raises `FileNotFoundError` and the
> pod crashloops before serving a request. `prd` is also wrong: it routes OAuth
> client secrets through Azure Key Vault. Everything environment-specific
> (bucket, URLs, secret, trace tagging via `OTEL_RESOURCE_ATTRIBUTES`) is set
> independently, so the only side effect is that PostHog analytics tags staging
> events as `dev`. Supporting a real `staging` value needs a backend change to
> fall back to `dev.integrations.yaml`.

Both environments share the cluster's SigNoz collector and the `aws-secrets-manager`
ClusterSecretStore. Everything else — Postgres, Redis, Vespa, Temporal, Svix,
Docling — is deployed per-namespace by this chart, so the two never share state.

## Consuming Services

Airweave is consumed by multiple services in the `neo-platform` repo. There is **no
ingress** on Airweave in either environment: the only path in is
`neo-gateway-svc`'s `/v1/meridian` proxy, which presents `SOURCE_CONNECTOR_API_KEY`
and applies tenant/workspace scoping.

| Service | Integration | Env var it reads |
|---------|-------------|------------------|
| `neo-gateway-svc` | REST proxy for the whole connector UI + OAuth callback + Svix webhook receiver | `SOURCE_CONNECTOR_BACKEND_URL`, `SOURCE_CONNECTOR_PUBLIC_URL`, `MERIDIAN_OAUTH_REDIRECT_URL`, `SYNC_WEBHOOK_CALLBACK_URL`, `SVIX_WEBHOOK_SECRET` |
| `neo-execution-engine-svc` | `DOC_SEARCH` Restate service calls `/collections/{id}/search/instant` | `SOURCE_CONNECTOR_BACKEND_URL` |
| `neo-graph-sync-svc` | Queries Vespa `file_entity.doc_categories` for T2 enrichment | `VESPA_URL` |
| `neo-dag-planner-svc` | Maps `DOC_SEARCH` plan steps | `DISABLE_DOC_SEARCH` |
| `neo-agent-orchestrator-svc` | Phase-2 T2 discovery + synthesis over Airweave results | `DISABLE_DOC_SEARCH` |

Internal K8s DNS:

- dev — `http://neo-airweave-svc-dev-use1-shared1.airweave-svc.svc.cluster.local:8001`
- staging — `http://neo-airweave-svc-staging-use1-shared1.staging-airweave-svc.svc.cluster.local:8001`

Vespa (queried directly by `neo-graph-sync-svc`):

- dev — `http://vespa.airweave-svc.svc.cluster.local:8081`
- staging — `http://vespa.staging-airweave-svc.svc.cluster.local:8081`

## Quick Start

Replace `<env>` with `dev-use1-shared1` or `staging-use1-shared1`.

```bash
# 1. Terraform — provision ECR + Secrets Manager + S3 + IRSA.
#    One workspace per environment: the S3 backend key is shared, so without a
#    workspace a second env would overwrite the first one's state.
cd ops/terraform/aws
terraform init
terraform workspace new <env>        # or: terraform workspace select <env>
terraform plan  -var-file="variables/<env>.tfvars"
terraform apply -var-file="variables/<env>.tfvars"

# 2. Populate the Secrets Manager secret by hand (see SECRETS.md).
#    Terraform only creates the shell and then ignores value drift.

# 3. Build and push the image for that environment.
#    build-and-push.sh only knows how to restart the DEV deployment, so skip its
#    restart step for staging and roll out by hand.
./ops/build-and-push.sh --suffix staging-use1-shared1 --tag staging-latest --no-restart
kubectl rollout restart deployment/neo-airweave-svc-staging-use1-shared1 -n staging-airweave-svc

# 4. Register the ArgoCD Application.
kubectl apply -f ops/argocd/application.yaml          -n argocd   # dev
kubectl apply -f ops/argocd/application-staging.yaml  -n argocd   # staging
```

### After the first staging sync

1. **Register the OAuth redirect URI** with every provider in use (Google Cloud
   Console, Atlassian developer console, Slack app config):
   `https://staging-dp.neosapientai.com/v1/meridian/source-connections/callback`.
   Airweave derives it from `API_FULL_URL`; the provider rejects the flow if the
   exact URI is not registered.
2. **Sync the Svix webhook secret.** `neo-gateway-svc` auto-registers its
   webhook subscription on boot, then verifies deliveries with
   `SVIX_WEBHOOK_SECRET`. Fetch the freshly-created endpoint's signing secret and
   put it in `neo-gateway-svc-staging-use1-shared1-secrets`:

   ```bash
   kubectl exec -n staging-neo-gateway-svc deploy/neo-gateway-svc-staging-use1-shared1 -- \
     curl -s "http://neo-airweave-svc-staging-use1-shared1.staging-airweave-svc.svc.cluster.local:8001/webhooks/subscriptions"
   # then GET /webhooks/subscriptions/<ep_id>?include_secret=true
   ```

   Until it matches, `sync.completed` deliveries fail signature verification and
   graph-sync never runs — the sync itself will look successful in the UI.

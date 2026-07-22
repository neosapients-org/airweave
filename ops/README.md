# Airweave Service — EKS Operations

Infrastructure-as-code and Kubernetes manifests for deploying Airweave to EKS on Neo Platform.

## Structure

```
ops/
├── SECRETS.md              # All secret keys that must be populated in AWS Secrets Manager
├── argocd/
│   └── application.yaml    # ArgoCD Application manifest (one per env)
├── helm/
│   └── airweave-svc/
│       ├── Chart.yaml
│       ├── values.yaml             # Defaults shared across all envs
│       ├── dev-use1-shared1.yaml   # Dev env overrides
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
        ├── main.tf             # ECR + Secrets Manager + S3
        ├── outputs.tf
        ├── providers.tf
        ├── variables.tf
        └── variables/
            └── dev-use1-shared1.tfvars
```

## Consuming Services

Airweave is consumed by multiple services in the `neo-platform` repo:

| Service | Integration |
|---------|-------------|
| `neo-execution-engine-svc` | Direct HTTP calls to `/search/instant` and `/search/classic` |
| `neo-agent-orchestrator-svc` | Synthesis layer processes Airweave search results |
| `neo-dag-planner-svc` | Maps `DOC_SEARCH` plan steps targeting Airweave |
| `neo-gateway-svc` | Fronts the MCP pipeline triggering doc search |

Internal K8s DNS: `http://airweave-svc.airweave-svc.svc.cluster.local:8001`

## Quick Start

```bash
# 1. Terraform — provision ECR + Secrets Manager + S3
cd ops/terraform/aws
terraform init
terraform workspace new dev-use1-shared1
terraform plan -var-file="variables/dev-use1-shared1.tfvars"
terraform apply -var-file="variables/dev-use1-shared1.tfvars"

# 2. Populate secrets (see SECRETS.md)

# 3. Deploy via ArgoCD
kubectl apply -f ops/argocd/application.yaml -n argocd
```

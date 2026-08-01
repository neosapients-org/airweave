# Staging Environment Configuration for airweave-svc
# Region: us-east-1 — same account and EKS cluster as dev (dev-use1-shared1),
# separated by namespace, IAM role, S3 bucket and Secrets Manager secret.
#
# Apply into its own Terraform workspace so the state never collides with dev:
#   terraform workspace new staging-use1-shared1
#   terraform apply -var-file="variables/staging-use1-shared1.tfvars"

aws_region = "us-east-1"
name       = "airweave-svc-staging-use1-shared1"

# Dedicated ECR repository for staging images (pushed as :staging-latest).
ecr_create = true

# Creates the secret SHELL only — values are populated by hand, see SECRETS.md.
# main.tf sets ignore_secret_changes, so a later apply will not clobber them.
secrets_manager_create = true

# Separate bucket: staging must never read or write dev's ingested raw files.
s3_storage_bucket_create = true
s3_storage_bucket_name   = "airweave-staging-use1-shared1"

# Same cluster as dev, so the same OIDC provider ARN. The trust policy is scoped
# to <namespace>:<service account>, which is what keeps the staging role from
# being assumable by the dev pods.
# k8s_service_account_name must equal the Helm release name
# (neo-airweave-svc-staging-use1-shared1 in ops/argocd/application-staging.yaml),
# because that is what airweave-svc.serviceAccountName renders to.
irsa_create              = true
eks_oidc_provider_arn    = "arn:aws:iam::679451892000:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/E2F1B9CCBC353D0581CF9F5CCE54E9DF"
k8s_namespace            = "staging-airweave-svc"
k8s_service_account_name = "neo-airweave-svc-staging-use1-shared1"

additional_tags = {
  Project     = "airweave-svc"
  Environment = "staging"
  ManagedBy   = "Terraform"
  Repository  = "airweave"
  Service     = "airweave-svc"
}

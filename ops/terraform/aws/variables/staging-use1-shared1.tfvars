# Staging Environment Configuration for airweave-svc
# Region: us-east-1 — same account and EKS cluster as dev, separated by
# namespace, IAM role, S3 bucket and Secrets Manager secret.
#
#   terraform workspace new staging-use1-shared1
#   terraform apply -var-file="variables/staging-use1-shared1.tfvars"

aws_region = "us-east-1"
name       = "airweave-svc-staging-use1-shared1"

ecr_create             = true
secrets_manager_create = true

s3_storage_bucket_create = true
s3_storage_bucket_name   = "airweave-staging-use1-shared1"

# Same cluster as dev, so the same OIDC provider. k8s_service_account_name must
# equal the Helm release name from ops/argocd/application-staging.yaml.
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

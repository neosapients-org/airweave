# Development Environment Configuration for airweave-svc
# Region: us-east-1

aws_region             = "us-east-1"
name                   = "airweave-svc-dev-use1-shared1"
ecr_create             = true
secrets_manager_create = true
s3_storage_bucket_create = true
s3_storage_bucket_name   = "airweave-dev-use1-shared1"

irsa_create              = true
eks_oidc_provider_arn    = "arn:aws:iam::679451892000:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/E2F1B9CCBC353D0581CF9F5CCE54E9DF"
k8s_namespace            = "airweave-svc"
k8s_service_account_name = "neo-airweave-svc-dev-use1-shared1"

additional_tags = {
  Project     = "airweave-svc"
  Environment = "development"
  ManagedBy   = "Terraform"
  Repository  = "airweave"
  Service     = "airweave-svc"
}

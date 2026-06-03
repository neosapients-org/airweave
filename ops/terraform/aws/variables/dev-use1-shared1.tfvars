# Development Environment Configuration for airweave-svc
# Region: us-east-1

aws_region             = "us-east-1"
name                   = "airweave-svc-dev-use1-shared1"
ecr_create             = true
secrets_manager_create = true
s3_storage_bucket_create = true
s3_storage_bucket_name   = "airweave-dev-use1-shared1"

additional_tags = {
  Project     = "airweave-svc"
  Environment = "development"
  ManagedBy   = "Terraform"
  Repository  = "airweave"
  Service     = "airweave-svc"
}

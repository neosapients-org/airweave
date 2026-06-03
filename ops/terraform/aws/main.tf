# ECR Repository for airweave-svc
module "ecr" {
  source  = "terraform-aws-modules/ecr/aws"
  version = "~> 2.0"

  create = var.ecr_create

  repository_name                 = var.name
  repository_image_tag_mutability = "MUTABLE"
  repository_image_scan_on_push   = false
  repository_encryption_type      = var.repository_encryption_type

  create_lifecycle_policy = false
}

# ===============================================================================
# AWS Secrets Manager
# ===============================================================================
# Creates the secret shell for airweave-svc.
# Values are managed manually via AWS Console — see SECRETS.md for required keys.
# ===============================================================================

module "secrets_manager" {
  source  = "terraform-aws-modules/secrets-manager/aws"
  version = "~> 1.0"

  create = var.secrets_manager_create

  name        = "${var.name}-secrets"
  description = "Secrets for ${var.name} service"

  # Placeholder — populate actual values manually in AWS Console after apply
  secret_string = jsonencode({
    AWS_REGION = var.aws_region
  })

  # Ignore changes to secret value — values will be added manually in AWS Console
  ignore_secret_changes = true
}

# ===============================================================================
# S3 Bucket for raw file storage
# ===============================================================================

module "s3_storage" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 4.0"

  create_bucket = var.s3_storage_bucket_create

  bucket = var.s3_storage_bucket_name

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }
}

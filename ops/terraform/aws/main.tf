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

  name        = "neo-${var.name}-secrets"
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

# ===============================================================================
# IRSA — IAM Role for Service Account (Airweave S3 access)
# ===============================================================================
# Allows the Airweave Kubernetes service account to read/write S3 without
# static credentials. Requires OIDC provider to be registered in IAM first.
# ===============================================================================

module "irsa_policy" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-policy"
  version = "~> 5.0"

  create_policy = var.irsa_create

  name        = "${var.name}-s3-policy"
  description = "S3 read/write access for Airweave service account"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_storage_bucket_name}",
          "arn:aws:s3:::${var.s3_storage_bucket_name}/*",
        ]
      }
    ]
  })

  tags = var.additional_tags
}

module "irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  create_role = var.irsa_create

  role_name = "${var.name}-irsa"

  oidc_providers = {
    main = {
      provider_arn               = var.eks_oidc_provider_arn
      namespace_service_accounts = ["${var.k8s_namespace}:${var.k8s_service_account_name}"]
    }
  }

  role_policy_arns = var.irsa_create ? {
    s3 = module.irsa_policy.arn
  } : {}

  tags = var.additional_tags
}

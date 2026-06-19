variable "aws_region" {
  description = "AWS region for airweave-svc infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "ecr_create" {
  description = "Whether to create ECR repository"
  type        = bool
  default     = false
}

variable "name" {
  description = "Resource name prefix (e.g. airweave-svc-<env>)"
  type        = string
  default     = ""

  validation {
    condition     = var.name != ""
    error_message = "name must be set (e.g. airweave-svc-dev-use1-shared1)."
  }
}

variable "repository_encryption_type" {
  description = "Type of encryption for the ECR repository"
  type        = string
  default     = "AES256"
}

variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "secrets_manager_create" {
  description = "Whether to create AWS Secrets Manager secret"
  type        = bool
  default     = true
}

variable "s3_storage_bucket_create" {
  description = "Whether to create S3 bucket for raw file storage"
  type        = bool
  default     = true
}

variable "s3_storage_bucket_name" {
  description = "S3 bucket name for Airweave raw file storage"
  type        = string
  default     = ""

  validation {
    condition     = !var.s3_storage_bucket_create || var.s3_storage_bucket_name != ""
    error_message = "s3_storage_bucket_name must be set when s3_storage_bucket_create is true."
  }
}

variable "irsa_create" {
  description = "Whether to create the IRSA IAM role for Airweave S3 access"
  type        = bool
  default     = false
}

variable "eks_oidc_provider_arn" {
  description = "OIDC provider ARN for the EKS cluster — sourced from infra-core stack output"
  type        = string
  default     = ""
}

variable "k8s_namespace" {
  description = "Kubernetes namespace where Airweave runs"
  type        = string
  default     = "airweave-svc"
}

variable "k8s_service_account_name" {
  description = "Kubernetes service account name for Airweave"
  type        = string
  default     = ""
}

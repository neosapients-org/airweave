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
}

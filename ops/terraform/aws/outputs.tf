output "repository_url" {
  description = "ECR repository URL for airweave-svc"
  value       = var.ecr_create ? module.ecr.repository_url : null
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = var.ecr_create ? module.ecr.repository_arn : null
}

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN"
  value       = module.secrets_manager.secret_arn
}

output "s3_storage_bucket_arn" {
  description = "S3 storage bucket ARN"
  value       = module.s3_storage.s3_bucket_arn
}

output "s3_storage_bucket_name" {
  description = "S3 storage bucket name"
  value       = module.s3_storage.s3_bucket_id
}

output "irsa_role_arn" {
  description = "ARN of the IRSA IAM role for Airweave S3 access (use in serviceAccount.annotations)"
  value       = var.irsa_create ? module.irsa.iam_role_arn : null
}

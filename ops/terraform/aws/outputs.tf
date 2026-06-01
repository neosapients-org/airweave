output "repository_url" {
  description = "ECR repository URL for airweave-svc"
  value       = module.ecr.repository_url
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = module.ecr.repository_arn
}

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN"
  value       = module.secrets_manager.secret_arn
}

output "s3_storage_bucket_arn" {
  description = "S3 storage bucket ARN"
  value       = var.s3_storage_bucket_create ? aws_s3_bucket.storage[0].arn : ""
}

output "s3_storage_bucket_name" {
  description = "S3 storage bucket name"
  value       = var.s3_storage_bucket_create ? aws_s3_bucket.storage[0].id : ""
}

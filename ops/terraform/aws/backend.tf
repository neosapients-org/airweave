# Backend configuration for airweave-svc
terraform {
  backend "s3" {
    bucket       = "neo-platform-terraform-state"
    key          = "services/airweave-svc/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

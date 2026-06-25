variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy resources"
  type        = string
  default     = "us-central1"
}

variable "db_password" {
  description = "The database password for PostgreSQL user"
  type        = string
  sensitive   = true
}

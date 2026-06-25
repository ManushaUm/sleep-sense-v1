output "db_connection_name" {
  description = "The Cloud SQL connection name to connect via Cloud Run"
  value       = google_sql_database_instance.postgres.connection_name
}

output "db_public_ip" {
  description = "The database public IP address"
  value       = google_sql_database_instance.postgres.public_ip_address
}

output "artifact_registry_endpoint" {
  description = "The Artifact Registry Docker registry endpoint"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.backend_repo.repository_id}"
}

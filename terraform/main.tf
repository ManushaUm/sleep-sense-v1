provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required APIs
locals {
  apis = [
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com"
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.apis)
  service            = each.key
  disable_on_destroy = false
}

# 2. Artifact Registry
resource "google_artifact_registry_repository" "backend_repo" {
  depends_on    = [google_project_service.enabled_apis["artifactregistry.googleapis.com"]]
  location      = var.region
  repository_id = "sleepsense-backend"
  description   = "Docker repository for SleepSense FastAPI backend"
  format        = "DOCKER"
}

# 3. VPC and VPC Access Connector for Cloud Run to DB Private IP
resource "google_compute_network" "vpc_network" {
  depends_on              = [google_project_service.enabled_apis["compute.googleapis.com"]]
  name                    = "sleepsense-vpc"
  auto_create_subnetworks = true
}

resource "google_vpc_access_connector" "connector" {
  depends_on    = [google_project_service.enabled_apis["vpcaccess.googleapis.com"]]
  name          = "sleepsense-vpc-conn"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc_network.name
  min_instances = 2
  max_instances = 3
}

# 4. Cloud SQL Database (PostgreSQL)
resource "google_sql_database_instance" "postgres" {
  depends_on       = [google_project_service.enabled_apis["sqladmin.googleapis.com"]]
  name             = "sleepsense-db"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-f1-micro" # Small, inexpensive instance for testing
    ip_configuration {
      ipv4_enabled = true # Enabled for initial migration; can be restricted to private IP later
      authorized_networks {
        name  = "Allow All for Init"
        value = "0.0.0.0/0"
      }
    }
  }
}

resource "google_sql_database" "database" {
  name     = "sleepsense"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "db_user" {
  name     = "sleepsense_user"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

# 5. Cloud Run Service Account
resource "google_service_account" "cloud_run_sa" {
  account_id   = "sleepsense-cloudrun-sa"
  display_name = "SleepSense Cloud Run Service Account"
}

# Grant Cloud IAM Permission to read Secrets & Cloud SQL Client role
resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

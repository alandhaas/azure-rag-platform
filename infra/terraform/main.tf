locals {
  normalized_project_name = replace(lower(var.project_name), "/[^a-z0-9-]/", "")
  compact_project_name    = replace(local.normalized_project_name, "-", "")
  resource_group_name     = coalesce(var.resource_group_name, "rg-${local.normalized_project_name}")

  storage_account_name = substr("${local.compact_project_name}${random_string.suffix.result}", 0, 24)
  acr_name             = substr("${local.compact_project_name}${random_string.suffix.result}", 0, 50)

  documents_container_name      = "documents"
  ingestion_queue_name          = "documents-to-ingest"
  document_metadata_table_name  = "DocumentMetadata"
  qdrant_storage_share_name     = "qdrant"
  qdrant_environment_storage    = "qdrant-storage"
  qdrant_environment_volume     = "qdrant-data"
  qdrant_environment_mount_path = "/qdrant/storage"

  common_tags = {
    project = var.project_name
  }
}

resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

resource "azurerm_resource_group" "main" {
  name     = local.resource_group_name
  location = var.location
  tags     = local.common_tags
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${local.normalized_project_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.normalized_project_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.common_tags
}

resource "azurerm_container_registry" "acr" {
  name                = local.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = local.common_tags
}

resource "azurerm_user_assigned_identity" "container_apps" {
  name                = "id-${local.normalized_project_name}-apps"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.common_tags
}

resource "azurerm_role_assignment" "container_apps_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

resource "azurerm_storage_account" "main" {
  name                     = local.storage_account_name
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"
  tags                     = local.common_tags
}

resource "azurerm_storage_container" "documents" {
  name                  = local.documents_container_name
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_queue" "ingestion" {
  name                 = local.ingestion_queue_name
  storage_account_name = azurerm_storage_account.main.name
}

resource "azurerm_storage_table" "document_metadata" {
  name                 = local.document_metadata_table_name
  storage_account_name = azurerm_storage_account.main.name
}

resource "azurerm_storage_share" "qdrant" {
  name               = local.qdrant_storage_share_name
  storage_account_id = azurerm_storage_account.main.id
  quota              = 5
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${local.normalized_project_name}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.common_tags
}

resource "azurerm_container_app_environment_storage" "qdrant" {
  name                         = local.qdrant_environment_storage
  container_app_environment_id = azurerm_container_app_environment.main.id
  account_name                 = azurerm_storage_account.main.name
  share_name                   = azurerm_storage_share.qdrant.name
  access_key                   = azurerm_storage_account.main.primary_access_key
  access_mode                  = "ReadWrite"
}

resource "azurerm_container_app" "qdrant" {
  name                         = "ca-${local.normalized_project_name}-qdrant"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.common_tags

  ingress {
    external_enabled = false
    target_port      = 6333
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  template {
    min_replicas = 1
    max_replicas = 1

    volume {
      name         = local.qdrant_environment_volume
      storage_name = azurerm_container_app_environment_storage.qdrant.name
      storage_type = "AzureFile"
    }

    container {
      name   = "qdrant"
      image  = var.qdrant_image
      cpu    = 0.5
      memory = "1Gi"

      volume_mounts {
        name = local.qdrant_environment_volume
        path = local.qdrant_environment_mount_path
      }
    }
  }
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${local.normalized_project_name}-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.common_tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  registry {
    server   = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name  = "storage-connection-string"
    value = azurerm_storage_account.main.primary_connection_string
  }

  secret {
    name  = "gemini-api-key"
    value = var.gemini_api_key
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  template {
    min_replicas = 0
    max_replicas = 3

    http_scale_rule {
      name                = "http"
      concurrent_requests = 20
    }

    container {
      name   = "api"
      image  = var.api_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "APP_ENV"
        value = "production"
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      env {
        name  = "CORS_ALLOWED_ORIGINS"
        value = var.cors_allowed_origins
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
      env {
        name  = "EMBEDDING_PROVIDER"
        value = "google"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "google"
      }
      env {
        name        = "GEMINI_API_KEY"
        secret_name = "gemini-api-key"
      }
      env {
        name  = "GEMINI_EMBEDDING_MODEL"
        value = var.gemini_embedding_model
      }
      env {
        name  = "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY"
        value = tostring(var.gemini_embedding_output_dimensionality)
      }
      env {
        name  = "GEMINI_LLM_MODEL"
        value = var.gemini_llm_model
      }
      env {
        name  = "QDRANT_URL"
        value = "http://${azurerm_container_app.qdrant.name}"
      }
      env {
        name  = "QDRANT_COLLECTION_NAME"
        value = var.qdrant_collection_name
      }
      env {
        name  = "QDRANT_VECTOR_SIZE"
        value = tostring(var.gemini_embedding_output_dimensionality)
      }
      env {
        name        = "AzureWebJobsStorage"
        secret_name = "storage-connection-string"
      }
      env {
        name  = "INGESTION_QUEUE_NAME"
        value = azurerm_storage_queue.ingestion.name
      }
      env {
        name  = "DOCUMENTS_CONTAINER_NAME"
        value = azurerm_storage_container.documents.name
      }
      env {
        name  = "DOCUMENT_METADATA_TABLE_NAME"
        value = azurerm_storage_table.document_metadata.name
      }
    }
  }
}

resource "azurerm_container_app" "worker" {
  name                         = "ca-${local.normalized_project_name}-worker"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.common_tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  registry {
    server   = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name  = "storage-connection-string"
    value = azurerm_storage_account.main.primary_connection_string
  }

  secret {
    name  = "gemini-api-key"
    value = var.gemini_api_key
  }

  template {
    min_replicas = 0
    max_replicas = 3

    azure_queue_scale_rule {
      name         = "ingestion-queue"
      queue_name   = azurerm_storage_queue.ingestion.name
      queue_length = 1

      authentication {
        secret_name       = "storage-connection-string"
        trigger_parameter = "connection"
      }
    }

    container {
      name   = "worker"
      image  = var.worker_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "APP_ENV"
        value = "production"
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
      env {
        name  = "EMBEDDING_PROVIDER"
        value = "google"
      }
      env {
        name  = "LLM_PROVIDER"
        value = "google"
      }
      env {
        name        = "GEMINI_API_KEY"
        secret_name = "gemini-api-key"
      }
      env {
        name  = "GEMINI_EMBEDDING_MODEL"
        value = var.gemini_embedding_model
      }
      env {
        name  = "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY"
        value = tostring(var.gemini_embedding_output_dimensionality)
      }
      env {
        name  = "GEMINI_LLM_MODEL"
        value = var.gemini_llm_model
      }
      env {
        name        = "AzureWebJobsStorage"
        secret_name = "storage-connection-string"
      }
      env {
        name  = "INGESTION_QUEUE_NAME"
        value = azurerm_storage_queue.ingestion.name
      }
      env {
        name  = "DOCUMENTS_CONTAINER_NAME"
        value = azurerm_storage_container.documents.name
      }
      env {
        name  = "DOCUMENT_METADATA_TABLE_NAME"
        value = azurerm_storage_table.document_metadata.name
      }
      env {
        name  = "DOCUMENT_CHUNK_MAX_CHARS"
        value = tostring(var.document_chunk_max_chars)
      }
      env {
        name  = "DOCUMENT_CHUNK_OVERLAP_CHARS"
        value = tostring(var.document_chunk_overlap_chars)
      }
      env {
        name  = "QDRANT_URL"
        value = "http://${azurerm_container_app.qdrant.name}"
      }
      env {
        name  = "QDRANT_COLLECTION_NAME"
        value = var.qdrant_collection_name
      }
      env {
        name  = "QDRANT_VECTOR_SIZE"
        value = tostring(var.gemini_embedding_output_dimensionality)
      }
    }
  }
}

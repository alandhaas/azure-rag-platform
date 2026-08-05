variable "project_name" {
  type    = string
  default = "azure-rag-platform"
}

variable "location" {
  type    = string
  default = "westeurope"
}

variable "resource_group_name" {
  type    = string
  default = null
}

variable "api_image" {
  type    = string
  default = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
}

variable "worker_image" {
  type    = string
  default = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
}

variable "qdrant_image" {
  type    = string
  default = "qdrant/qdrant:v1.18.3-unprivileged"
}

variable "gemini_api_key" {
  type      = string
  sensitive = true
}

variable "gemini_embedding_model" {
  type    = string
  default = "gemini-embedding-001"
}

variable "gemini_embedding_output_dimensionality" {
  type    = number
  default = 768
}

variable "gemini_llm_model" {
  type    = string
  default = "gemini-3.6-flash"
}

variable "qdrant_collection_name" {
  type    = string
  default = "documents"
}

variable "document_chunk_max_chars" {
  type    = number
  default = 1200
}

variable "document_chunk_overlap_chars" {
  type    = number
  default = 200
}

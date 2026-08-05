#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bootstrap-azure-github.sh OWNER/REPO

Example:
  scripts/bootstrap-azure-github.sh Alandhaas/azure-rag-platform

Optional environment variables:
  AZURE_LOCATION=westeurope
  AZURE_ROLE=Owner
  AZURE_APP_NAME=azure-rag-platform-github
  TF_STATE_STORAGE_ACCOUNT=<globally-unique-storage-account-name>
  GEMINI_API_KEY=<google-ai-studio-api-key>

If the GitHub CLI is installed and authenticated, the script also writes the
repository secrets for you. Otherwise it prints the values to add manually.
USAGE
}

repo="${1:-}"
if [[ -z "$repo" || "$repo" != */* ]]; then
  usage
  exit 1
fi

location="${AZURE_LOCATION:-westeurope}"
role="${AZURE_ROLE:-Owner}"
app_name="${AZURE_APP_NAME:-azure-rag-platform-github}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command az

if ! az account show >/dev/null 2>&1; then
  echo "Azure CLI is not logged in. Run: az login" >&2
  exit 1
fi

subscription_id="$(az account show --query id --output tsv)"
tenant_id="$(az account show --query tenantId --output tsv)"
scope="/subscriptions/${subscription_id}"

echo "Using Azure subscription: ${subscription_id}"
echo "Using Azure tenant:       ${tenant_id}"
echo "Using GitHub repo:        ${repo}"

client_id="$(az ad app list --display-name "$app_name" --query '[0].appId' --output tsv)"
if [[ -z "$client_id" ]]; then
  echo "Creating Azure app registration: ${app_name}"
  client_id="$(az ad app create --display-name "$app_name" --query appId --output tsv)"
else
  echo "Reusing Azure app registration: ${app_name}"
fi

if ! az ad sp show --id "$client_id" >/dev/null 2>&1; then
  echo "Creating service principal"
  az ad sp create --id "$client_id" --output none
else
  echo "Reusing service principal"
fi

principal_id="$(az ad sp show --id "$client_id" --query id --output tsv)"

existing_role_assignment="$(
  az role assignment list \
    --assignee "$principal_id" \
    --scope "$scope" \
    --query "[?roleDefinitionName=='${role}'].id | [0]" \
    --output tsv
)"

if [[ -z "$existing_role_assignment" ]]; then
  echo "Assigning ${role} on subscription"
  az role assignment create \
    --assignee-object-id "$principal_id" \
    --assignee-principal-type ServicePrincipal \
    --role "$role" \
    --scope "$scope" \
    --output none
else
  echo "Role assignment already exists"
fi

federated_name="github-main"
federated_subject="repo:${repo}:ref:refs/heads/main"
existing_federated="$(
  az ad app federated-credential list \
    --id "$client_id" \
    --query "[?name=='${federated_name}'].name | [0]" \
    --output tsv
)"

if [[ -z "$existing_federated" ]]; then
  echo "Creating GitHub OIDC federated credential"
  federated_parameters="$(printf '{"name":"%s","issuer":"https://token.actions.githubusercontent.com","subject":"%s","audiences":["api://AzureADTokenExchange"]}' "$federated_name" "$federated_subject")"
  az ad app federated-credential create \
    --id "$client_id" \
    --parameters "$federated_parameters" \
    --output none
else
  echo "Federated credential already exists"
fi

tf_state_storage_account="${TF_STATE_STORAGE_ACCOUNT:-}"
if [[ -z "$tf_state_storage_account" ]]; then
  suffix="$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-' | cut -c 1-10)"
  tf_state_storage_account="aragtf${suffix}"
fi

if [[ ! "$tf_state_storage_account" =~ ^[a-z0-9]{3,24}$ ]]; then
  echo "TF_STATE_STORAGE_ACCOUNT must be 3-24 lowercase letters/numbers." >&2
  exit 1
fi

name_available="$(
  az storage account check-name \
    --name "$tf_state_storage_account" \
    --query nameAvailable \
    --output tsv
)"

if [[ "$name_available" != "true" ]]; then
  echo "Storage account name is not available: ${tf_state_storage_account}" >&2
  echo "Rerun with a different value, for example:" >&2
  echo "  TF_STATE_STORAGE_ACCOUNT=aragtf$(date +%s) scripts/bootstrap-azure-github.sh ${repo}" >&2
  exit 1
fi

echo
echo "GitHub secrets:"
echo "AZURE_CLIENT_ID=${client_id}"
echo "AZURE_TENANT_ID=${tenant_id}"
echo "AZURE_SUBSCRIPTION_ID=${subscription_id}"
echo "TF_STATE_STORAGE_ACCOUNT=${tf_state_storage_account}"
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY=<provided from environment>"
else
  echo "GEMINI_API_KEY=<paste your Google AI Studio key>"
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo
  echo "Writing GitHub repository secrets with gh"
  gh secret set AZURE_CLIENT_ID --repo "$repo" --body "$client_id"
  gh secret set AZURE_TENANT_ID --repo "$repo" --body "$tenant_id"
  gh secret set AZURE_SUBSCRIPTION_ID --repo "$repo" --body "$subscription_id"
  gh secret set TF_STATE_STORAGE_ACCOUNT --repo "$repo" --body "$tf_state_storage_account"
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    gh secret set GEMINI_API_KEY --repo "$repo" --body "$GEMINI_API_KEY"
  else
    echo "GEMINI_API_KEY was not set because GEMINI_API_KEY env var is empty."
  fi
else
  echo
  echo "GitHub CLI is not available/authenticated, so add the secrets manually in GitHub."
fi

echo
echo "Done."

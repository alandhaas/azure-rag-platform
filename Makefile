.PHONY: install lint typecheck test check api worker worker-functions services-up compose-up compose-down compose-build compose-logs compose-ps bootstrap-azure-github

install:
	uv sync

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

check: lint typecheck test

api:
	uv run uvicorn rag_api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	cd app/worker && PYTHONPATH=src:../../packages/rag_core/src uv run --project ../.. python -m rag_worker.local_worker

worker-functions:
	cd app/worker && PYTHONPATH=src:../../packages/rag_core/src uv run --project ../.. func start

services-up:
	docker compose up -d qdrant qdrant-init azurite azurite-init

compose-up:
	docker compose up -d qdrant qdrant-init azurite azurite-init api worker

compose-down:
	docker compose down

compose-build:
	docker compose build api worker

compose-logs:
	docker compose logs -f api worker qdrant azurite

compose-ps:
	docker compose ps

bootstrap-azure-github:
	@test -n "$(REPO)" || (echo "Usage: make bootstrap-azure-github REPO=OWNER/REPO" && exit 1)
	bash scripts/bootstrap-azure-github.sh "$(REPO)"

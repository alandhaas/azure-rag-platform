.PHONY: install lint typecheck test check api worker compose-up compose-down compose-build compose-logs compose-ps

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
	cd app/worker && func start

compose-up:
	docker compose up -d qdrant qdrant-init azurite api worker

compose-down:
	docker compose down

compose-build:
	docker compose build api worker

compose-logs:
	docker compose logs -f api worker qdrant azurite

compose-ps:
	docker compose ps

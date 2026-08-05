ARG FUNCTIONS_BASE_PLATFORM=linux/amd64

FROM --platform=${FUNCTIONS_BASE_PLATFORM} ghcr.io/astral-sh/uv:0.11.29 AS uv
FROM --platform=${FUNCTIONS_BASE_PLATFORM} mcr.microsoft.com/azure-functions/python:4-python3.14 AS runtime

COPY --from=uv /uv /uvx /bin/

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    FUNCTIONS_WORKER_RUNTIME=python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/home/site/wwwroot/.venv \
    PATH="/home/site/wwwroot/.venv/bin:${PATH}" \
    PYTHONPATH="/home/site/wwwroot/.venv/lib/python3.14/site-packages:/home/site/wwwroot"

WORKDIR /home/site/wwwroot

COPY pyproject.toml uv.lock README.md ./
COPY app/worker/pyproject.toml app/worker/README.md ./app/worker/
COPY app/worker/src ./app/worker/src
COPY packages/rag_core/pyproject.toml packages/rag_core/README.md ./packages/rag_core/
COPY packages/rag_core/src ./packages/rag_core/src

RUN uv sync --frozen --no-dev --package rag-worker

COPY app/worker/function_app.py ./function_app.py
COPY app/worker/host.json ./host.json

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1/api/health/live', timeout=3).read()"

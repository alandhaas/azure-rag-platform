"""Application factory for the RAG API."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from rag_api.observability.logging import configure_logging
from rag_api.observability.middleware import RequestLoggingMiddleware
from rag_api.routes.health import router as health_router


async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="Azure RAG Platform API",
        version="0.1.0",
        summary="HTTP API for document ingestion, retrieval, and RAG workflows.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "filter": True,
        },
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health_router)
    app.add_api_route("/", redirect_to_docs, include_in_schema=False)

    return app


app = create_app()

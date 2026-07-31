"""Application factory for the RAG API."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from rag_api.config import ApiSettings, get_settings
from rag_api.observability.logging import configure_logging
from rag_api.observability.middleware import RequestLoggingMiddleware
from rag_api.routes.health import router as health_router


async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        summary="HTTP API for document ingestion, retrieval, and RAG workflows.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "filter": True,
        },
    )
    app.state.settings = settings
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health_router)
    app.add_api_route("/", redirect_to_docs, include_in_schema=False)

    return app


app = create_app()

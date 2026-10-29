"""FastAPI application for CIC-IDS2017 model inference."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.predictions import router as predictions_router
from app.core.config import Settings
from app.ml.model_loader import load_xgboost_bundle
from app.schemas.prediction import HealthResponse, ReadinessResponse
from app.services.prediction_service import PredictionService

logger = logging.getLogger("cicids.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = configured
        application.state.prediction_service = None
        application.state.model_load_error = None
        try:
            bundle = load_xgboost_bundle(configured.artifact_dir)
            application.state.prediction_service = PredictionService(bundle)
            logger.info("Loaded XGBoost inference artifact version %s", bundle.get("model_version"))
        except Exception as exc:
            application.state.model_load_error = exc
            logger.error("Model artifact could not be loaded (%s); API will remain not ready", type(exc).__name__)
        yield
        application.state.prediction_service = None

    application = FastAPI(
        title="CIC-IDS2017 Inference API",
        description="Single and bounded-batch predictions from the saved CIC-IDS2017 XGBoost model.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = configured
    application.state.prediction_service = None
    application.state.model_load_error = None
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(health_router)
    application.include_router(predictions_router)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "CIC-IDS2017 Inference API", "health": "/health", "readiness": "/ready"}

    return application


app = create_app()

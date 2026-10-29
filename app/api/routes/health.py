"""Health, readiness, and model information routes."""

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.prediction import HealthResponse, ModelMetadataResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse, tags=["health"])
def readiness(request: Request) -> ReadinessResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return ReadinessResponse(status="ready", model_loaded=True)


@router.get("/models", response_model=ModelMetadataResponse, tags=["models"])
def model_metadata(request: Request) -> ModelMetadataResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return ModelMetadataResponse(
        model_name=service.model_name,
        model_version=service.model_version,
        supported_classes=service.label_classes,
    )

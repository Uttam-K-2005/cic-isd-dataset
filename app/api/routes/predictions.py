"""Single and bounded batch predictions."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRecord,
    PredictionResponse,
)

router = APIRouter()


def _service(request: Request):
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return service


def _validation_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(record: PredictionRecord, request: Request) -> PredictionResponse:
    service = _service(request)
    request_id = uuid4()
    try:
        result = service.predict(record.root, request_id)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(body: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    service = _service(request)
    limit = request.app.state.settings.max_batch_size
    if len(body.records) > limit:
        raise HTTPException(
            status_code=422,
            detail=f"Batch has {len(body.records)} records; maximum is {limit}",
        )
    request_id = uuid4()
    try:
        results = service.predict_many([record.root for record in body.records], request_id)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    timestamp = results[0]["predicted_at"]
    return BatchPredictionResponse(
        request_id=request_id,
        predictions=[PredictionResponse(**result) for result in results],
        model_name=service.model_name,
        model_version=service.model_version,
        predicted_at=timestamp,
    )

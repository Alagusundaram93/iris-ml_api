from contextlib import asynccontextmanager
import json
import time
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.config import settings
from app.logging_config import setup_logger
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router

logger = setup_logger()
class PredictionShapeError(Exception):
    pass

@asynccontextmanager
async def lifecycle(app: FastAPI):
    try:
        app.state.trained_model = joblib.load(settings.MODEL_PATH)
        with open(settings.METADATA_PATH,"r",encoding="utf-8") as file:
            app.state.model_metadata = json.load(file)
        logger.info(
            "ML model and metadata loaded successfully"
        )

    except Exception as exc:
        logger.error(f"ML model loading failed error={exc}")
        raise
    yield

app = FastAPI(
    title=settings.API_TITLE,
    description="A simple API to predict and health check.",
    version=settings.API_VERSION,
    lifespan=lifecycle
)

app.include_router(v1_router)
app.include_router(v2_router)

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next
):
    from uuid import uuid4
    request_id = str(uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"request_started"
    )
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start_time
        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration={duration:.4f}s"
        )

@app.exception_handler(PredictionShapeError)
async def prediction_shape_exception_handler(
    request: Request,
    exc: PredictionShapeError
):
    request_id = request.state.request_id
    logger.error(
        f"request_id={request_id} "
        f"prediction_shape_error"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction input shape is invalid"
        }
    )
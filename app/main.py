from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logger


trained_model = None
logger = setup_logger()

class PredictionShapeError(Exception):
    pass


@asynccontextmanager
async def lifecycle(app: FastAPI):
    global trained_model

    try:
        trained_model = joblib.load("ml/saved_model/model.joblib")

        logger.info("ML model loaded successfully")

    except Exception as exc:
        logger.error(f"ML model loading failed error={exc}")
        raise

    yield


app = FastAPI(
    title="MY_IRIS_PREDICTOR",
    description="A simple API to predict and health check.",
    version="1.0.0",
    lifespan=lifecycle
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": trained_model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    request: Request
):
    request_id = request.state.request_id

    input_list = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]]

    try:
        prediction = trained_model.predict(input_list)[0]

        if hasattr(trained_model, "predict_proba"):
            probabilities = trained_model.predict_proba(input_list)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.0

        flower_names = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }

        try:
            prediction = flower_names[int(prediction)]

        except (KeyError, ValueError, TypeError):
            raise PredictionShapeError()

    except PredictionShapeError:
        raise

    except Exception as exc:
        logger.error(
            f"request_id={request_id} "
            f"prediction_failed "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    logger.info(
        f"request_id={request_id} "
        f"prediction={prediction} "
        f"confidence={confidence:.4f} "
        f"prediction_success"
    )

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "request_id": request_id
    }
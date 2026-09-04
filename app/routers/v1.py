import time
from fastapi import APIRouter, HTTPException, Request
from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput,
)
from app.logging_config import setup_logger

router = APIRouter(prefix="/api/v1",tags=["version 1"])
logger = setup_logger()

@router.get("/health")
def health(request: Request):
    trained_model = getattr(
        request.app.state,
        "trained_model",
        None
    )
    return {
        "status": "ok",
        "model_loaded": trained_model is not None
    }

@router.post("/predict",response_model=PredictionOutput)
def predict(input_data: PredictionInput,request: Request):
    request_id = request.state.request_id
    trained_model = request.app.state.trained_model
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
            raise HTTPException(
                status_code=500,
                detail="Prediction input shape is invalid"
            )
    except HTTPException:
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

@router.post("/predict-batch",response_model=PredictionBatchOutput)
def predict_batch(
    batch_data: PredictionBatchInput,
    request: Request
):
    request_id = request.state.request_id
    trained_model = request.app.state.trained_model
    batch_size = len(batch_data.inputs)
    if batch_size > settings.MAX_BATCH_SIZE:
        logger.warning(
            f"request_id={request_id} "
            f"batch_size={batch_size} "
            f"max_batch_size={settings.MAX_BATCH_SIZE} "
            f"batch_size_limit_exceeded"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch size cannot exceed "
                f"{settings.MAX_BATCH_SIZE}"
            )
        )
    start_time = time.perf_counter()
    logger.info(
        f"request_id={request_id} "
        f"batch_size={batch_size} "
        f"batch_prediction_started"
    )
    try:
        input_matrix = [
            [
                item.sepal_length,
                item.sepal_width,
                item.petal_length,
                item.petal_width
            ]
            for item in batch_data.inputs
        ]
        predictions = trained_model.predict(
            input_matrix
        )
        if hasattr(trained_model, "predict_proba"):
            probabilities = trained_model.predict_proba(
                input_matrix
            )
        else:
            probabilities = None

        flower_names = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }
        results = []
        for index, prediction in enumerate(predictions):
            try:
                flower_name = flower_names[
                    int(prediction)
                ]
            except (KeyError, ValueError, TypeError):
                raise HTTPException(
                    status_code=500,
                    detail="Prediction input shape is invalid"
                )
            if probabilities is not None:
                confidence = float(
                    max(probabilities[index])
                )
            else:
                confidence = 0.0
            results.append(
                PredictionOutput(
                    prediction=flower_name,
                    confidence=confidence,
                    request_id=request_id
                )
            )
        duration = time.perf_counter() - start_time
        logger.info(
            f"request_id={request_id} "
            f"batch_size={batch_size} "
            f"duration={duration:.4f}s "
            f"batch_prediction_success"
        )
        return {
            "predictions": results
        }
    except HTTPException:
        raise
    except Exception as exc:
        duration = time.perf_counter() - start_time
        logger.error(
            f"request_id={request_id} "
            f"batch_size={batch_size} "
            f"duration={duration:.4f}s "
            f"batch_prediction_failed "
            f"error={exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )

@router.get("/model-info")
def model_info(request: Request):
    request_id = request.state.request_id
    try:
        metadata = request.app.state.model_metadata
        logger.info(
            f"request_id={request_id} "
            f"model_info_success"
        )
        return metadata
    except Exception as exc:
        logger.error(
            f"request_id={request_id} "
            f"model_info_failed "
            f"error={exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Model information unavailable"
        )
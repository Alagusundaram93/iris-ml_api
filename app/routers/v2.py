import time
from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import (
    PredictionInput,
    PredictionV2Output,
)
from app.logging_config import setup_logger

router=APIRouter(prefix="/api/v2",tags=["version 2"])
logger = setup_logger()

@router.post(
    "/predict",
    response_model=PredictionV2Output
)
def predict(
    input_data: PredictionInput,
    request: Request
):
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
        if not hasattr(trained_model, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="Model does not support probability prediction"
            )
        probabilities = trained_model.predict_proba(input_list)[0]
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

        probability_map = {
            "setosa": float(probabilities[0]),
            "versicolor": float(probabilities[1]),
            "virginica": float(probabilities[2])
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"request_id={request_id} "
            f"v2_prediction_failed "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="V2 prediction failed"
        )

    logger.info(
        f"request_id={request_id} "
        f"prediction={prediction} "
        f"v2_prediction_success"
    )

    return {
        "prediction": str(prediction),
        "probabilities": probability_map,
        "request_id": request_id
    }

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_predict_model_failure(caplog):

    fake_model = Mock()

    fake_model.predict.side_effect = Exception("Model crashed")

    with caplog.at_level("ERROR", logger="iris_ml_api"):

        with patch(
            "app.main.joblib.load",
            return_value=fake_model
        ):

            with TestClient(app) as test_client:

                response = test_client.post(
                    "/predict",
                    json={
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    }
                )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"

    # Task 9: verify error logging
    assert "prediction_failed" in caplog.text
    assert "request_id=" in caplog.text


def test_predict_logs_success(caplog):

    with caplog.at_level("INFO", logger="iris_ml_api"):

        with TestClient(app) as test_client:

            response = test_client.post(
                "/predict",
                json={
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            )

    assert response.status_code == 200

    # Task 9: verify successful prediction logging
    assert "prediction_success" in caplog.text
    assert "request_id=" in caplog.text
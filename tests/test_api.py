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
                    "/api/v1/predict",
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
                "/api/v1/predict",
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


def test_predict_batch():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predict-batch",
            json={
                "inputs": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    },
                    {
                        "sepal_length": 6.2,
                        "sepal_width": 2.9,
                        "petal_length": 4.3,
                        "petal_width": 1.3
                    }
                ]
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "request_id" in prediction


def test_predict_batch_logs_success(caplog):
    with caplog.at_level("INFO", logger="iris_ml_api"):
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/predict-batch",
                json={
                    "inputs": [
                        {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2
                        },
                        {
                            "sepal_length": 6.2,
                            "sepal_width": 2.9,
                            "petal_length": 4.3,
                            "petal_width": 1.3
                        }
                    ]
                }
            )

    assert response.status_code == 200

    # Task 11: verify batch logging
    assert "batch_prediction_started" in caplog.text
    assert "batch_prediction_success" in caplog.text
    assert "batch_size=2" in caplog.text
    assert "duration=" in caplog.text
    assert "request_id=" in caplog.text


def test_predict_batch_accepts_100_inputs():
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    batch = [input_data for _ in range(100)]

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predict-batch",
            json={
                "inputs": batch
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 100


def test_predict_batch_rejects_more_than_100_inputs():
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    batch = [input_data for _ in range(101)]

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predict-batch",
            json={
                "inputs": batch
            }
        )

    assert response.status_code == 422


def test_predict_batch_rejects_empty_list():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predict-batch",
            json={
                "inputs": []
            }
        )

    assert response.status_code == 422


def test_model_info():
    with TestClient(app) as test_client:
        response = test_client.get(
            "/api/v1/model-info"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "RandomForestClassifier"
    assert data["version"] == "1.0"

    assert "training_date" in data

    assert data["features"] == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    assert data["accuracy"] == 0.9


def test_model_info_logs_success(caplog):
    with caplog.at_level("INFO", logger="iris_ml_api"):
        with TestClient(app) as test_client:
            response = test_client.get(
                "/api/v1/model-info"
            )

    assert response.status_code == 200
    assert "model_info_success" in caplog.text
    assert "request_id=" in caplog.text
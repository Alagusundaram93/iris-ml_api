def test_predict_batch_valid(client):
    response = client.post(
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


def test_predict_batch_rejects_oversized_batch(client):
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    batch = [input_data for _ in range(101)]

    response = client.post(
        "/api/v1/predict-batch",
        json={
            "inputs": batch
        }
    )

    assert response.status_code == 422


def test_predict_batch_rejects_empty_list(client):
    response = client.post(
        "/api/v1/predict-batch",
        json={
            "inputs": []
        }
    )

    assert response.status_code == 422
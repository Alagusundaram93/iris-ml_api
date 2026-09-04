def test_predict_valid_input(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
    assert "confidence" in data
    assert "request_id" in data
def test_predict_missing_field(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 422
def test_predict_invalid_value(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 422
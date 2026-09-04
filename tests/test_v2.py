def test_v2_predict(client):
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post(
        "/api/v2/predict",
        json=input_data
    )
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probabilities" in data
    assert "request_id" in data
    assert "confidence" not in data
    assert data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
    probabilities = data["probabilities"]
    assert "setosa" in probabilities
    assert "versicolor" in probabilities
    assert "virginica" in probabilities
    assert all(
        0 <= probability <= 1
        for probability in probabilities.values()
    )

def test_v1_and_v2_have_different_response_shapes(client):
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    v1_response = client.post(
        "/api/v1/predict",
        json=input_data
    )
    v2_response = client.post(
        "/api/v2/predict",
        json=input_data
    )
    assert v1_response.status_code == 200
    assert v2_response.status_code == 200
    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # V1 old response shape
    assert "prediction" in v1_data
    assert "confidence" in v1_data
    assert "request_id" in v1_data
    assert "probabilities" not in v1_data

    # V2 new response shape
    assert "prediction" in v2_data
    assert "probabilities" in v2_data
    assert "request_id" in v2_data
    assert "confidence" not in v2_data

    # Both predictions should be valid
    assert v1_data["prediction"] in[
        "setosa",
        "versicolor",
        "virginica"
    ]
    assert v2_data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
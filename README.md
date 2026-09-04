# Iris ML API

A Machine Learning REST API built using **Python, Scikit-learn, FastAPI, Pydantic, and Pytest**.

This project demonstrates the complete workflow of training a Machine Learning model and serving predictions through a validated and tested REST API.

---

## Overview

The API predicts the species of an Iris flower using four measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Classes

* Setosa
* Versicolor
* Virginica

The project uses the **Iris dataset** and a **RandomForestClassifier**.

---

## Tech Stack

| Technology     | Purpose                      |
| -------------- | ---------------------------- |
| Python         | Application & ML development |
| Scikit-learn   | Model training               |
| FastAPI        | REST API                     |
| Pydantic       | Data validation              |
| Joblib         | Model persistence            |
| Pytest         | API testing                  |
| Python Logging | Application logging          |
| Git & GitHub   | Version control              |

---

## Architecture

```text
Client
   ↓
FastAPI
   ↓
Pydantic Validation
   ↓
Loaded ML Model
   ↓
Prediction + Confidence
   ↓
JSON Response
```

The trained model is loaded once during application startup.

---

## Project Structure

```text
iris-ml_api/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_config.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── routers/
│       ├── v1.py
│       └── v2.py
│
├── ml/
│   ├── train.py
│   ├── predict_saved_model.py
│   └── saved_model/
│       ├── model.joblib
│       └── metadata.json
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_batch.py
│   ├── test_model_info.py
│   └── test_v2.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Development Tasks

### Task 1 — Problem & Architecture

Defined the Iris classification problem, API requirements, inputs, outputs, and overall ML API architecture.

### Task 2 — Environment & Project Structure

Created a Python virtual environment, configured dependencies, and organized the project into application, ML, and testing modules.

### Task 3 — Model Training & Persistence

Trained a `RandomForestClassifier` using the Iris dataset and saved the trained model using Joblib.

```text
Iris Dataset
     ↓
Train/Test Split
     ↓
Random Forest
     ↓
Evaluation
     ↓
model.joblib
```

### Task 4 — Initial FastAPI Application

Created the initial FastAPI application with basic endpoints and interactive Swagger documentation.

### Task 5 — Model Loading

Integrated the saved Machine Learning model into FastAPI and configured it to load during application startup.

### Task 6 — Input Validation

Implemented Pydantic validation for the four Iris measurements.

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Task 7 — Core Prediction API

Implemented the complete `/predict` workflow with:

* Model prediction
* Confidence score
* Request ID
* `/health` endpoint

### Task 8 — Response Models & Error Handling

Added Pydantic response models, HTTP status handling, exception handling, and automated API tests using Pytest.

### Task 9 — Logging & Observability

Added structured application logging to monitor startup, prediction requests, request IDs, prediction results, and errors.

---

### Task 10 — API Versioning

Introduced API versioning using FastAPI `APIRouter` to create a scalable and maintainable API structure.

The existing prediction and supporting endpoints were moved under the `/api/v1` prefix without changing their core behavior.

Implemented version 1 endpoints:

* `/api/v1/health`
* `/api/v1/predict`
* `/api/v1/predict-batch`
* `/api/v1/model-info`

The versioned router is registered with the main FastAPI application using `include_router()`.

This provides a foundation for introducing future API versions without breaking existing clients.

---

### Task 11 — Batch Prediction & Model Information

Extended the API to support multiple Iris predictions in a single request.

Implemented:

* `/api/v1/predict-batch`
* `/api/v1/model-info`
* Configurable maximum batch size
* Batch prediction using a single model inference call
* Confidence calculation for each prediction
* Batch processing duration logging
* Model metadata retrieval

Example batch flow:

```text
Multiple Iris Inputs
        ↓
Pydantic Validation
        ↓
Batch Size Validation
        ↓
Input Matrix
        ↓
ML Model
        ↓
Multiple Predictions
        ↓
Confidence Scores
        ↓
JSON Response
```

Batch prediction improves efficiency by sending multiple inputs to the model in a single inference operation.

The `/api/v1/model-info` endpoint provides metadata associated with the trained model.

---

### Task 12 — Configuration Management

Introduced centralized configuration management using **Pydantic Settings** and environment variables.

Application configuration includes:

* `MODEL_PATH`
* `METADATA_PATH`
* `LOG_LEVEL`
* `MAX_BATCH_SIZE`
* `API_TITLE`
* `API_VERSION`

Configuration values are loaded from the `.env` file instead of being hard-coded throughout the application.

This makes the application easier to configure for different environments such as development, testing, and production.

Example:

```text
.env
   ↓
Pydantic Settings
   ↓
Settings Object
   ↓
FastAPI Application
   ↓
Routers / Services
```

The `.env` file is excluded from Git tracking using `.gitignore`.

---

### Task 13 — Automated API Testing

Implemented a structured automated testing layer using **Pytest** and FastAPI's `TestClient`.

The test suite verifies:

* Health check behavior
* Model loading status
* Prediction responses
* Input validation
* Batch prediction
* Model information
* Error handling
* Response structures
* Logging-related behavior
* Model failure scenarios

A reusable Pytest fixture was created for the FastAPI `TestClient`.

Example testing flow:

```text
Pytest
   ↓
TestClient
   ↓
FastAPI Endpoint
   ↓
Application Logic
   ↓
Response
   ↓
Assertions
   ↓
PASS / FAIL
```

Automated tests provide regression protection and help ensure that existing API behavior continues to work when new features are introduced.

The complete test suite was verified successfully with:

```bash
pytest -v
```

---

### Task 14 — API V2 & Breaking Response Change

Introduced a new **API Version 2** with a different response structure while keeping the existing V1 API unchanged.

The V1 prediction response returns:

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "request_id": "unique-request-id"
}
```

The V2 prediction response returns the complete probability distribution:

```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "request_id": "unique-request-id"
}
```

Implemented:

* `/api/v2/predict`
* `PredictionV2Output` response schema
* Full class probability response
* V2-specific error handling
* Automated V2 API tests
* V1 vs V2 response-shape verification

### V1 vs V2

| Feature            | V1                | V2                |
| ------------------ | ----------------- | ----------------- |
| Prediction         | ✅                 | ✅                 |
| Confidence         | ✅                 | ❌                 |
| Full Probabilities | ❌                 | ✅                 |
| Request ID         | ✅                 | ✅                 |
| API Path           | `/api/v1/predict` | `/api/v2/predict` |

The V1 API was intentionally preserved to maintain backward compatibility for existing clients.

---

# API Endpoints

### `GET /`

Returns API status.

```json
{
  "message": "ML API is alive"
}
```

### `GET /api/v1/health`

Checks API and model status.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /api/v1/predict`

Returns the predicted Iris species and confidence score.

Example response:

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "request_id": "unique-request-id"
}
```

### `POST /api/v1/predict-batch`

Accepts multiple Iris inputs and returns predictions for all inputs.

Example structure:

```json
{
  "predictions": [
    {
      "prediction": "setosa",
      "confidence": 1.0,
      "request_id": "unique-request-id"
    }
  ]
}
```

### `GET /api/v1/model-info`

Returns metadata related to the trained Machine Learning model.

### `POST /api/v2/predict`

Returns the predicted Iris species along with the probability of each class.

Example response:

```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "request_id": "unique-request-id"
}
```

---

# Installation

```bash
git clone https://github.com/Alagusundaram93/iris-ml_api.git
cd iris-ml_api

python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file in the project root:

```text
MODEL_PATH=ml/saved_model/model.joblib
METADATA_PATH=ml/saved_model/metadata.json
LOG_LEVEL=INFO
MAX_BATCH_SIZE=100
API_TITLE=MY_IRIS_PREDICTOR
API_VERSION=1.0.0
```

---

# Run the Application

Train the model:

```bash
python ml/train.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

The versioned endpoints can be tested through Swagger:

```text
/api/v1/health
/api/v1/predict
/api/v1/predict-batch
/api/v1/model-info
/api/v2/predict
```

---

# Run Tests

```bash
pytest -v
```

The test suite covers:

* Health checks
* Model loading
* Predictions
* Input validation
* Batch prediction
* Model information
* Error handling
* V2 response validation
* V1 and V2 response compatibility

### Final Test Result

```text
19 passed
```

---

# Key Learning Outcomes

* Machine Learning model training
* Model persistence with Joblib
* FastAPI REST API development
* Pydantic validation
* API error handling
* Automated testing with Pytest
* Structured logging
* Request tracing
* API versioning
* FastAPI APIRouter
* Batch prediction
* Model metadata management
* Configuration management with Pydantic Settings
* Environment variables
* API response models
* Backward compatibility
* Breaking API changes
* V1 and V2 API design
* Git & GitHub workflow

---

# Future Improvements

* Docker deployment
* CI/CD with GitHub Actions
* API authentication
* Advanced model versioning
* Cloud deployment
* Production monitoring
* API rate limiting
* Database integration
* Advanced observability

---

## Author

**Alagu Sundaram M**

### GitHub Repository

https://github.com/Alagusundaram93/iris-ml_api

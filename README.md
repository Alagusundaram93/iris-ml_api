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
│   └── models/
│       └── schemas.py
│
├── ml/
│   ├── train.py
│   ├── predict_saved_model.py
│   └── saved_model/
│       └── model.joblib
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
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

# API Endpoints

### `GET /`

Returns API status.

```json
{
  "message": "ML API is alive"
}
```

### `GET /health`

Checks API and model status.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Returns the predicted Iris species.

Example response:

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
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

---

# Run Tests

```bash
pytest -v
```

The test suite covers health checks, predictions, model loading, response validation, and model failure handling.

---

# Development Status

| Task                              | Status      |
| --------------------------------- | ----------- |
| Task 1 — Architecture             | ✅ Completed |
| Task 2 — Environment              | ✅ Completed |
| Task 3 — Model Training           | ✅ Completed |
| Task 4 — FastAPI Setup            | ✅ Completed |
| Task 5 — Model Loading            | ✅ Completed |
| Task 6 — Pydantic Validation      | ✅ Completed |
| Task 7 — Prediction API           | ✅ Completed |
| Task 8 — Error Handling & Testing | ✅ Completed |
| Task 9 — Logging & Observability  | ✅ Completed |

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
* Git & GitHub workflow

---

# Future Improvements

* Docker deployment
* CI/CD with GitHub Actions
* API authentication
* Model versioning
* Cloud deployment
* Production monitoring

---

## Author

**Alagu Sundaram M**

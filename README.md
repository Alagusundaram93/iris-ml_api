# Iris ML API

A small learning project that demonstrates how a Machine Learning model can eventually be exposed through a FastAPI service.

This repository covers **Tasks 1–4** of the learning plan:

- Task 1 — Understand the project and plan the architecture
- Task 2 — Set up the development environment and project structure
- Task 3 — Train and save the first ML model
- Task 4 — Build the first bare-bones FastAPI application

> **Important:** Task 4 intentionally keeps `/predict` hardcoded. The saved ML model is connected to the API in Task 5, so this repository should not replace the hardcoded response yet.

## 1. Problem Statement

The project classifies an Iris flower into one of three species based on four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

Possible classes:

- Setosa
- Versicolor
- Virginica

The Iris dataset is the built-in dataset provided by scikit-learn. It contains 150 samples and four input features.

## 2. API Contract

The planned API exposes a `POST /predict` endpoint that will eventually accept four numerical measurements:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The completed API will validate the input, send the measurements to the saved ML model, and return the predicted species as JSON.

Example planned response:

```json
{
  "prediction": "setosa"
}
```

## 3. Architecture Flow

```text
Client
  |
  v
POST /predict
  |
  v
Input validation
  |
  v
Saved ML model
  |
  v
Prediction
  |
  v
JSON response
```

For Task 4, only the FastAPI server layer is implemented; the model and validation layers are intentionally deferred.

## 4. Project Structure

```text
iris-ml-api/
├── app/
│   ├── main.py
│   ├── models/
│   │   └── .gitkeep
│   └── routers/
│       └── .gitkeep
├── ml/
│   ├── train.py
│   ├── predict_saved_model.py
│   └── saved_model/
│       └── model.joblib
├── tests/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Task 3 — Train and Save the Model

`ml/train.py`:

1. Loads the Iris dataset.
2. Splits the data into 80% training and 20% test data.
3. Uses `stratify=y` and `random_state=42` for a repeatable split.
4. Trains a `RandomForestClassifier`.
5. Prints test accuracy.
6. Saves the trained model as `ml/saved_model/model.joblib` using `joblib`.

Run:

```bash
python ml/train.py
```

Expected output is similar to:

```text
Test accuracy: 90% or higher
Saved model to: .../ml/saved_model/model.joblib
```

The exact accuracy can vary if the training implementation is changed, but the current fixed implementation is repeatable.

## 6. Prove Model Reloading Works

The separate script loads the saved `.joblib` file without retraining and performs a prediction.

Run:

```bash
python ml/predict_saved_model.py
```

## 7. Task 4 — Bare-Bones FastAPI

Start the server:

```bash
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

The Task 4 endpoints are:

### GET `/`

Returns:

```json
{
  "message": "ML API is alive"
}
```

### POST `/predict`

Returns the temporary Task 4 response:

```json
{
  "prediction": "hardcoded_result"
}
```

This hardcoded response is intentional because Task 4 is only proving that the FastAPI server and routes work. Task 5 will replace it with the saved model.

## 8. Setup

Create and activate a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 9. Task 1–4 Completion Checklist

- [x] Iris dataset and classification problem selected
- [x] API contract documented
- [x] Request → validation → model → response architecture documented
- [x] GitHub-ready project structure created
- [x] `.gitignore` excludes virtual environments and Python cache files
- [x] `requirements.txt` included
- [x] ML training script included
- [x] Train/test split implemented
- [x] Model metric printed
- [x] Model saved with `joblib`
- [x] Separate saved-model reload script included
- [x] FastAPI application created
- [x] `GET /` implemented
- [x] `POST /predict` implemented as the Task 4 hardcoded endpoint
- [x] FastAPI automatic `/docs` available

## Next Task

Task 5 should load `ml/saved_model/model.joblib` inside FastAPI, add Pydantic request validation, and replace the hardcoded `/predict` response with a real Iris prediction.

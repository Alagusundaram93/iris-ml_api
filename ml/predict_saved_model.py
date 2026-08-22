from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "saved_model" / "model.joblib"


def predict_saved_model():
    """Load the persisted model and make a prediction without retraining."""
    model = joblib.load(MODEL_PATH)

    sample = [[5.0, 4.0, 6.2, 1.8]]
    prediction = int(model.predict(sample)[0])

    names = ["setosa", "versicolor", "virginica"]
    print("Prediction:", names[prediction])


if __name__ == "__main__":
    predict_saved_model()

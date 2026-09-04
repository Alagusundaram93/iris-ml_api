from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = (Path(__file__).resolve().parent/ "saved_model"/ "model.joblib")
METADATA_PATH = (Path(__file__).resolve().parent/ "saved_model"/ "metadata.json")
MODEL_VERSION = "1.0"

def train():
    """Train a Random Forest classifier on the Iris dataset and save it."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        predictions,
    )
    print(f"Test accuracy: {accuracy:.2%}")
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Save the trained ML model
    joblib.dump(model,MODEL_PATH,)

    # Save model metadata
    metadata = {
        "model_type": type(model).__name__,
        "version": MODEL_VERSION,
        "training_date": datetime.now(
            timezone.utc
        ).isoformat(),
        "features": iris.feature_names,
        "accuracy": accuracy,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
if __name__ == "__main__":
    train()
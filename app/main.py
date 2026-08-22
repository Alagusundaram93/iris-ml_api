from fastapi import FastAPI

app = FastAPI(
    title="Iris ML API",
    description="Bare-bones FastAPI application for Task 4.",
    version="0.1.0",
)

@app.get("/")
def root():
    """Health-style endpoint used to verify that the API is running."""
    return {"message": "ML API is alive"}

@app.post("/predict")
def predict():
    """Temporary Task 4 endpoint; the real model is added in Task 5."""
    return {"prediction": "hardcoded_result"}

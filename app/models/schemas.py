from pydantic import BaseModel, Field
class PredictionInput(BaseModel):
    sepal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="List must contain at least 1 PredictionInput item"
    )
    sepal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="List must contain at least 1 PredictionInput item"
    )
    petal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="List must contain at least 1 PredictionInput item"
    )
    petal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="List must contain at least 1 PredictionInput item"
    )
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    request_id: str

class PredictionBatchInput(BaseModel):
    inputs: list[PredictionInput] = Field(
        ...,
        min_length=1,
        description="List must contain at least 1 PredictionInput item"
    )

class PredictionBatchOutput(BaseModel):
    predictions: list[PredictionOutput] = Field(
        ...,
        min_length=1,
        description="List must contain at least 1 PredictionOutput item"
    )

class PredictionV2Output(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    request_id: str
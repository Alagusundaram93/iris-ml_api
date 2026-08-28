from pydantic import BaseModel, Field
class PredictionInput(BaseModel):
    sepal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal length must be greater than 0 and at most 10"
    )
    sepal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal width must be greater than 0 and at most 10"
    )
    petal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal length must be greater than 0 and at most 10"
    )
    petal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal width must be greater than 0 and at most 10"
    )
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    request_id: str
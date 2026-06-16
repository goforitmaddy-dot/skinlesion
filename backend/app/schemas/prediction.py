from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
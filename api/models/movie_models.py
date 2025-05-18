from pydantic import BaseModel
from typing import List, Dict

class MovieInput(BaseModel):
    title: str
    synopsis: str

class BatchMovieInput(BaseModel):
    movies: List[MovieInput]

class PredictionResponse(BaseModel):
    title: str
    predicted_genre: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] 
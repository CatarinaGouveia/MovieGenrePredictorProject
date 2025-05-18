from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import List
from api.models.movie_models import (
    PredictionResponse,
)
from api.service.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Synopsis Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Predict genre for a single movie",
            "/predict_batch": "Predict genres for multiple movies",
            "/models": "List available models"
        }
    }

@router.get("/models")
async def get_models():
    """Get list of available models"""
    return {
        "available_models": prediction_service.get_available_models()
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    title: str = Query(..., description="Movie title"),
    synopsis: str = Query(..., description="Movie synopsis"),
    model_choice: str = Query(default="LightGBM", description="Model to use for prediction")
):
    """
    Predict genre for a single movie using query parameters
    """
    try:
        result = prediction_service.predict_single(
            title=title,
            synopsis=synopsis,
            model_choice=model_choice
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(
    file: UploadFile = File(...),
    model_choice: str = Form(default="LightGBM", description="Model to use for prediction")
):
    """
    Predict genres for multiple movies from a file
    """
    try:
        content = await file.read()
        text = content.decode()
        
        # Parse the tab-separated file
        results = []
        for line in text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                title, synopsis = parts[:2]
                result = prediction_service.predict_single(
                    title=title,
                    synopsis=synopsis,
                    model_choice=model_choice
                )
                results.append(result)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
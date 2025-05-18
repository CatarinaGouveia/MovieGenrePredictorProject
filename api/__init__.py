from fastapi import FastAPI

app = FastAPI(
    title="Movie Synopsis Predictor API",
    description="API for predicting movie genres based on title and synopsis",
    version="1.0.0"
)

from api.controllers import movie_controller

# Include routers
app.include_router(movie_controller.router) 
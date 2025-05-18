from fastapi import FastAPI
from api.controllers.movie_controller import router

# Initialize FastAPI app
app = FastAPI(
    title="Movie Synopsis Predictor API",
    description="API for predicting movie genres based on title and synopsis",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
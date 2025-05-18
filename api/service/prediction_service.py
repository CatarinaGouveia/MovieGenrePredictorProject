import pickle
import nltk
from nltk.corpus import stopwords
import re
from typing import List, Dict, Tuple
import logging
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {str(e)}")

class PredictionService:
    def __init__(self):
        self.models = self._load_models()
        self.vectorizer = self._load_vectorizer()

    def _load_models(self):
        models = {}
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "LightGBM": "lightgbm.pkl"
        }
        
        for model_name, filename in model_files.items():
            with open(f"models/{filename}", 'rb') as f:
                models[model_name] = pickle.load(f)
        return models
    
    def _load_vectorizer(self):
        with open("models/vectorizer.pkl", 'rb') as f:
            return pickle.load(f)

    def clean_text(self, text):
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords (keep important ones)
        stop_words = set(stopwords.words('english'))
        important_words = {'not', 'no', 'never', 'against', 'between', 'through', 'during', 'before', 'after'}
        stop_words = stop_words - important_words
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        
        return text

    def predict_single(self, title: str, synopsis: str, model_choice: str):
        # Prepare input
        combined_text = f"{title} {synopsis}"
        combined_text = self.clean_text(combined_text)
        X = pd.DataFrame(
            self.vectorizer.transform([combined_text]).toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        
        # Get model
        model = self.models[model_choice]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Format response
        return {
            "title": title,
            "predicted_genre": prediction,
            "confidence": float(max(probabilities) * 100),
            "probabilities": {
                genre: float(prob * 100)
                for genre, prob in zip(model.classes_, probabilities)
            }
        }

    def predict_batch(self, movies: List[Tuple[str, str]], model_name: str = "logistic_regression") -> List[Tuple[str, Dict[str, float]]]:
        """Make predictions for multiple movies"""
        if not movies:
            return []
        
        # Validate the model choice
        if model_name not in self.models:
            available_models = list(self.models.keys())
            if available_models:
                model_name = available_models[0]
                logger.warning(f"Requested model not found. Using {model_name} instead.")
            else:
                raise ValueError("No models available for prediction.")
        
        results = []
        for title, synopsis in movies:
            prediction = self.predict_single(title, synopsis, model_name)
            results.append((prediction["predicted_genre"], prediction["probabilities"]))
        
        return results

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys()) 
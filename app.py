import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import plotly.express as px
import os
import logging
from typing import Dict, Any
from nltk.stem import WordNetLemmatizer


# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set page config
st.set_page_config(page_title="Movie Synopsis Classifier", layout="wide")

# Title with film icon
st.markdown("<h1 style='text-align: center;'>🎬 Movie Synopsis Predictor</h1>", unsafe_allow_html=True)

# Add explanation with adjusted padding
st.markdown("""
    <div>
    Welcome to the Movie Synopsis Predictor! This tool uses AI to predict movie genres 
    based on their titles and synopses. You can either enter a single movie's details 
    or upload a file with multiple entries. Our machine learning models will analyze 
    the text and predict the most likely genre categories.
    </div>
    """, unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
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

@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load all models and vectorizer"""
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "LightGBM": "lightgbm.pkl"
    }
    
    for model_name, filename in model_files.items():
        with open(f"models/{filename}", 'rb') as f:
            models[model_name] = pickle.load(f)
            
    with open("models/vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
        
    return {"models": models, "vectorizer": vectorizer}

def predict_genre(title: str, synopsis: str, model_choice: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction using selected model"""
    # Prepare input
    combined_text = f"{title} {synopsis}"
    combined_text = clean_text(combined_text)
    
    # Transform text - use array instead of DataFrame
    X = models["vectorizer"].transform([combined_text])  # Keep as sparse matrix
    
    # Get model and predict
    model = models["models"][model_choice]
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

def main():
    st.title("Movie Genre Predictor")
    
    # Load models
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        list(models["models"].keys())
    )
    
    # Input method selection
    input_method = st.radio(
        "Choose input method",
        ["Single Prediction", "Batch Processing"]
    )
    
    if input_method == "Single Prediction":
        st.subheader("Enter Movie Details")
        
        title = st.text_input("Movie Title")
        synopsis = st.text_area("Movie Synopsis")
        
        if st.button("Predict Genre"):
            if title or synopsis:
                if not title or not synopsis:
                    st.warning("Please enter both title and synopsis for a better prediction")
                try:
                    result = predict_genre(title, synopsis, model_choice, models)
                    
                    # Display results
                    st.success(f"Predicted Genre: {result['predicted_genre']}")
                    st.info(f"Confidence: {result['confidence']:.1f}%")
                    
                    # Display probabilities
                    st.subheader("Probability Distribution")
                    probs_df = pd.DataFrame(
                        list(result["probabilities"].items()),
                        columns=["Genre", "Probability"]
                    )
                    st.bar_chart(probs_df.set_index("Genre"))
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    else:  # Batch Processing
        st.subheader("Upload File")
        st.info("File should be tab-separated with columns: Title, Synopsis")
        
        uploaded_file = st.file_uploader("Choose a file", type="txt")
        
        if uploaded_file:
            try:
                # Read and process file
                content = uploaded_file.read().decode()
                results = []
                
                for line in content.strip().split('\n'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        title, synopsis = parts[:2]
                        result = predict_genre(title, synopsis, model_choice, models)
                        results.append(result)
                
                # Display results
                if results:
                    st.success(f"Processed {len(results)} movies")
                    
                    # Create DataFrame
                    results_df = pd.DataFrame([
                        {
                            "Title": r["title"],
                            "Predicted Genre": r["predicted_genre"],
                            "Confidence": f"{r['confidence']:.1f}%"
                        }
                        for r in results
                    ])
                    
                    st.dataframe(results_df)
                else:
                    st.warning("No valid entries found in file")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 
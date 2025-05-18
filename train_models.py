import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import pickle
import os
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import numpy as np
from sklearn.base import clone

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def clean_text(text):
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


def prepare_data(data_file):
    # Prepare data
    print("Loading and preparing data...")
    data = pd.read_csv(data_file)
    data = data.drop_duplicates()
    data['Combined'] = data['Title'] + ' ' + data['Synopsis']
    data['cleaned_text'] = data['Combined'].apply(clean_text)
    return data

def split_data(data):
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data, 
        test_size=0.3,  # 30% for final testing
        stratify=data['Tag'],
        random_state=42
    )

    return train_val_data, test_data

def vectorize_data(train_val_data, test_data, output_dir):

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    # Fit vectorizer only on training data
    X_train_val = vectorizer.fit_transform(train_val_data['cleaned_text'])
    X_test = vectorizer.transform(test_data['cleaned_text'])
    y_train_val = train_val_data['Tag'].values
    y_test = test_data['Tag'].values

    # Save vectorizer
    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Vectorizer saved successfully")

    return X_train_val, X_test, y_train_val, y_test, vectorizer



def save_evaluation_results(evaluation_results, output_path):
    """Save evaluation results to CSV, appending if file exists"""
    results_df = pd.DataFrame(evaluation_results)
    
    if os.path.exists(output_path):
        existing_results = pd.read_csv(output_path)
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    
    results_df.to_csv(output_path, index=False)
    return results_df


def get_model_params(model):
    """Extract relevant parameters from model"""
    return {
        key: value for key, value in model.get_params().items()
        if not key.startswith('_')  # Skip private attributes
    }


def print_evaluation_summary(results_df):
    """Print evaluation results with both CV and test metrics"""
    print("\nEvaluation Results Summary:")
    print("=" * 80)
    
    # Get latest results for each model
    latest_results = results_df.sort_values('timestamp').groupby('model').last()
    
    for model_name, row in latest_results.iterrows():
        print(f"\nModel: {model_name}")
        print(f"Timestamp: {row['timestamp']}")
        print("\nCross-validation metrics:")
        print(f"CV Accuracy: {row['cv_accuracy_mean']:.3f}")
        print(f"CV ROC AUC: {row['cv_roc_auc_mean']:.3f}")
        print(f"CV Avg Precision: {row['cv_avg_precision_mean']:.3f}")
        print("\nTest set metrics:")
        print(f"Test Accuracy: {row['test_accuracy']:.3f}")
        print(f"Test ROC AUC: {row['test_roc_auc']:.3f}")
        print(f"Test Avg Precision: {row['test_avg_precision']:.3f}")
        print("-" * 40)
    
    # Show best model for each metric on test set
    print("\nBest Models (Test Set Performance):")
    print(f"Best Test Accuracy: {latest_results['test_accuracy'].idxmax()} ({latest_results['test_accuracy'].max():.3f})")
    print(f"Best Test ROC AUC: {latest_results['test_roc_auc'].idxmax()} ({latest_results['test_roc_auc'].max():.3f})")
    print(f"Best Test Avg Precision: {latest_results['test_avg_precision'].idxmax()} ({latest_results['test_avg_precision'].max():.3f})")

def train_and_evaluate(data_file, evaluation_results_file, output_dir='models'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = prepare_data(data_file)
    train_val_data, test_data = split_data(data)
    X_train_val, X_test, y_train_val, y_test, vectorizer = vectorize_data(train_val_data, test_data, output_dir)

    # Initialize models with explicit parameters
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr',
            C=1.0,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            learning_rate=0.1,
        )
    }

    evaluation_results = []
    
    for name, model in models.items():
        print(f"\nEvaluating and training {name}...")
        
        # Cross-validation on train_val data
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_scores = []
        best_model = None
        best_score = -float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            # Oversample training data
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
            
            # Train and evaluate
            fold_model = clone(model)
            fold_model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on validation set
            val_pred = fold_model.predict(X_val)
            val_proba = fold_model.predict_proba(X_val)
            
            # Calculate validation metrics
            val_score = roc_auc_score(y_val, val_proba, multi_class='ovr')
            
            if val_score > best_score:
                best_score = val_score
                best_model = fold_model
            
            fold_scores.append({
                'fold': fold,
                'accuracy': accuracy_score(y_val, val_pred),
                'roc_auc': val_score,
                'avg_precision': average_precision_score(y_val, val_proba, average='macro')
            })

        # Refit best model on the entire training set
        print(f"Refitting {name} on the full training set...")
        best_model.fit(X_train_val, y_train_val)
        
        # Evaluate best model on test set
        test_pred = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test)
        
        test_scores = {
            'accuracy': accuracy_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_proba, multi_class='ovr'),
            'avg_precision': average_precision_score(y_test, test_proba, average='macro')
        }
        
        # Save results
        evaluation_results.append({
            'model': name,
            'cv_accuracy_mean': np.mean([s['accuracy'] for s in fold_scores]),
            'cv_roc_auc_mean': np.mean([s['roc_auc'] for s in fold_scores]),
            'cv_avg_precision_mean': np.mean([s['avg_precision'] for s in fold_scores]),
            'test_accuracy': test_scores['accuracy'],
            'test_roc_auc': test_scores['roc_auc'],
            'test_avg_precision': test_scores['avg_precision'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': str(get_model_params(best_model)),
            'vectorizer_params': str(vectorizer.get_params())
        })
        
        # Save the best model
        with open(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.pkl"), 'wb') as f:
            pickle.dump(best_model, f)
            print(f"{name} model saved successfully")

    # Save and print results
    results_df = save_evaluation_results(evaluation_results, evaluation_results_file)
    print_evaluation_summary(results_df)
    
    return results_df

if __name__ == "__main__":
    
    results_df = train_and_evaluate(
        data_file='task/task.csv',
        evaluation_results_file='evaluation_results.csv'
    )

    print("All models and results have been saved successfully.")

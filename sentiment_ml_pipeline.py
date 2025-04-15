# sentiment_ml_pipeline.py

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib

nltk.download('punkt')

# Step 1: Load and preprocess dataset
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'].str.strip() != '']

    # Encode target variable (sentiment)
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['rating_sentiment'])

    return df, label_encoder

# Step 2: Vectorization and Train-Test Split
def vectorize_and_split(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment_encoded'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Step 3: Train and Evaluate Model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Weighted F1 Score:", f1)
    return model, f1

# Step 4: Save model and vectorizer
def save_model_and_vectorizer(model, vectorizer):
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully.")

# Main function to execute pipeline
def train_model():
    filepath = 'kochi_edutech_reviews_new.csv'
    df, label_encoder = load_and_prepare_data(filepath)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = vectorize_and_split(df)
    model, f1 = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
    save_model_and_vectorizer(model, vectorizer)
    return model, f1

if __name__ == '__main__':
    train_model()

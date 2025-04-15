# EduTech Sentiment Analysis

## Overview

This project performs **Sentiment Analysis** on reviews from various EduTech companies in Kochi. It uses **Natural Language Processing (NLP)** techniques to analyze user feedback and categorize sentiment into **Positive**, **Negative**, or **Neutral**. The model utilizes **TextBlob** for sentiment extraction, **TF-IDF** for text vectorization, and a **Logistic Regression** classifier to predict sentiment.

## Objectives

- Analyze sentiment trends in reviews from EduTech companies in Kochi.
- Provide insights on user opinions about various companies' services.
- Deploy a machine learning model with continuous integration and continuous deployment (CI/CD) to ensure a robust performance.

## Features

- **Sentiment Analysis**: Categorizes reviews as Positive, Negative, or Neutral using TextBlob and a Logistic Regression model.
- **EDA Visualizations**: Provides visual insights using Matplotlib and Seaborn, including sentiment distributions, word clouds, and sentiment trends over time.
- **Time Series Analysis**: Identifies trends in sentiment over time and provides insights into sentiment shifts for strategic decision-making.
- **CI/CD Pipeline**: The project integrates with **GitHub Actions** for automatic retraining and deployment, ensuring that the model is always up-to-date and scalable.

## Technologies Used

- **Python Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - textblob
  - nltk
  - seaborn
  - matplotlib
  - wordcloud
  - joblib
- **Machine Learning**: Logistic Regression for sentiment classification.
- **CI/CD Tools**: GitHub Actions for automating training, testing, and deployment.

## Getting Started

### Prerequisites

1. **Python 3.8+** installed on your machine.
2. Install required dependencies via:

    ```bash
    pip install -r requirements.txt
    ```

### Data Requirements

Make sure you have the dataset in CSV format (e.g., `kochi_edutech_reviews_new.csv`). The dataset should contain columns like:
- **Review**: The textual review left by users.
- **Company**: The EduTech company name.
- **Rating**: The rating given by users (optional).
- **Date**: The date when the review was posted.

### Running the Project

1. Clone the repository:

    ```bash
    git clone https://github.com/sahlapn/EduTech-Sentiment-Analysis.git
    cd EduTech-Sentiment-Analysis
    ```

2. Run the sentiment analysis script:

    ```bash
    python sentiment_ml_pipeline.py
    ```

3. The model will be trained and saved as `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.

4. For CI/CD, push to the `main` branch to automatically retrain the model and deploy via GitHub Actions.

## Workflow & CI/CD

This project includes an automated CI/CD pipeline that:
1. Installs dependencies.
2. Runs training and testing scripts.
3. Uploads the trained model and vectorizer as GitHub artifacts.
4. Deploys the model to the cloud (customizable section for deployment).

## Visualizations

Several types of visualizations are included to provide insights:
- **Sentiment Distribution**: Shows the overall distribution of sentiments (Positive, Negative, Neutral).
- **Company-wise Sentiment**: Compares sentiments across different EduTech companies.
- **Word Clouds**: Displays the most frequent words in Positive, Negative, and Neutral reviews.
- **Polarity vs Subjectivity**: A scatter plot showing the relationship between polarity and subjectivity in the reviews.
- **Sentiment Trend Over Time**: A time series plot displaying sentiment trends for strategic insights.

## Contribution

Feel free to contribute to the project by:
- Reporting bugs
- Suggesting improvements
- Submitting pull requests
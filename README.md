# Sentiment Analysis & Emotion Detection Web App

An NLP-powered full-stack web application that detects human emotions from input text. The application uses a trained Machine Learning model to classify text into six distinct emotional categories in real-time.

## Features

- **Text Preprocessing**: Automated text cleaning (lowercasing, punctuation removal, digit extraction, and ASCII filtering) to sanitize raw text.
- **TF-IDF Vectorization**: Text representation using TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams (1 to 3) and sublinear term frequency scaling.
- **Machine Learning Classifier**: Built on a tuned Logistic Regression model achieving high accuracy.
- **Interactive UI**: A modern, responsive dashboard interface built with Streamlit featuring dynamic color cards corresponding to predicted emotions.

## Tech Stack

- **Frontend / App UI**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Manipulation**: Pandas, NumPy
- **Text Representation**: TF-IDF Vectorizer
- **Serialization**: Pickle

---

## Model Details & Performance

The model is trained on a labeled dataset (`train.txt`) containing 16,000 sentences classified into six emotions:
- 😡 **Anger** (Label `0`)
- 😨 **Fear** (Label `1`)
- 😊 **Joy** (Label `2`)
- ❤️ **Love** (Label `3`)
- 😢 **Sadness** (Label `4`)
- 😲 **Surprise** (Label `5`)

### Performance Metrics (Test Set Evaluated)
- **Accuracy**: **87.5%**
- **Weighted F1-Score**: **0.88**

#### Detailed Classification Report:
```text
              precision    recall  f1-score   support

    Anger       0.88      0.84      0.86       432
    Fear        0.86      0.83      0.85       387
    Joy         0.88      0.91      0.89      1072
    Love        0.75      0.80      0.77       261
    Sadness     0.93      0.90      0.91       933
    Surprise    0.78      0.78      0.78       115

    accuracy                           0.88      3200
```

---

## Project Structure

```text
NLP-Project/
├── venv/                  # Local Python Virtual Environment
├── app.py                 # Streamlit Web Application
├── models.ipynb           # Model Training Jupyter Notebook
├── requirements.txt       # App Python Dependencies
├── emotion_model.pkl      # Serialized Logistic Regression model
├── tfidf_vectorizer.pkl   # Serialized TF-IDF vectorizer
├── train.txt              # Dataset file
└── README.md              # Project Documentation
```

---

## Local Setup

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Installation
Clone the repository (or navigate to the directory) and activate the virtual environment:

```powershell
# On Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
Start the Streamlit application:
```bash
streamlit run app.py
```
Open the Local URL in your browser:
```text
http://localhost:8501
```

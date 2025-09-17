# 💬 Twitter Sentiment Analyzer

A **Sentiment Analysis Web App** built with **Python, NLTK, Scikit-Learn, and Streamlit**.  
It analyzes tweets (or any short text) and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral**.

---

## 🚀 Features
- **Real-time Prediction**: Enter text and instantly get sentiment.
- **Custom Preprocessing**:
  - Lower-casing
  - URL removal
  - Non-alphabetic character removal
  - Tokenization & Lemmatization (NLTK)
- **TF-IDF Vectorization** with bigrams for richer context.
- **Logistic Regression Model** trained directly on your dataset.
- **Streamlit UI** for an interactive, clean web interface.

---

## 📂 Project Structure
├─ app.py # Main Streamlit application
├─ twitter_training.csv # Twitter Sentiment dataset 
├─ README.md # This file
└─ requirements.txt # Python dependencies

## Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

## Install Dependencies

## Launch the App
streamlit run app.py

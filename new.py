import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# --- 1. Load the saved model/vectorizer OR retrain quickly  ---
# If you already trained and saved model+vectorizer with pickle, load them here.
# For demo we quickly train inside the app (only for small datasets).
@st.cache_resource   # caches once for faster reloads
def load_model():
    # Load your dataset
    df = pd.read_csv("twitter_training.csv", header=None)
    df.columns = ["id","topic","sentiment","text"]
    df = df[["sentiment","text"]].dropna()

    # Preprocess
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    lemmatizer = WordNetLemmatizer()
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s!?]", "", text)
        tokens = [lemmatizer.lemmatize(t) for t in text.split()]
        return " ".join(tokens)

    df["clean_text"] = df["text"].apply(clean_text)

    # Vectorizer + model
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    model = LogisticRegression(max_iter=500, C=5, class_weight="balanced")
    model.fit(X, y)
    return model, vectorizer, clean_text

model, vectorizer, clean_text = load_model()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬")
st.title("💬 Twitter Sentiment Analyzer")
st.write("Enter a tweet below and get the predicted sentiment in real time.")

user_input = st.text_area("✍️ Type or paste a tweet:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        clean = clean_text(user_input)
        features = vectorizer.transform([clean])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text.")

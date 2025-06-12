import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🕵️ Fake News Detector")
st.write("Enter a news article or headline below to check if it's **Fake** or **Real**.")

user_input = st.text_area("📰 Paste headline or article text here:", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.success("✅ This news appears to be **REAL**.")
        else:
            st.error("❌ This news appears to be **FAKE**.")

import streamlit as st
import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("📰 AI Fake News Detector")
st.write("Paste a news article below and find out if it's Real or Fake.")

user_input = st.text_area("Enter news article text:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some text first.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        probability = model.predict_proba(input_tfidf)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ Prediction: REAL News")
        else:
            st.error(f"❌ Prediction: FAKE News")

        st.write(f"Confidence: {confidence:.2f}%")
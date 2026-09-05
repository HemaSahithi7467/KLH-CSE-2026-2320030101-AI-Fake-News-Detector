import joblib

# Load the saved model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Paste any news article text here to test
sample_text = """
Scientists have discovered a new medicine that cures all diseases overnight,
according to shocking reports that doctors don't want you to know about.
"""

# Convert the text using the SAME vectorizer (don't retrain it)
sample_tfidf = vectorizer.transform([sample_text])

# Predict
prediction = model.predict(sample_tfidf)[0]

if prediction == 1:
    print("Prediction: REAL news")
else:
    print("Prediction: FAKE news")
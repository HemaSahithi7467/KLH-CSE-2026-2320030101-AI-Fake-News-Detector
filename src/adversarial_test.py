import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict(text):
    tfidf = vectorizer.transform([text])
    pred = model.predict(tfidf)[0]
    conf = max(model.predict_proba(tfidf)[0]) * 100
    label = "REAL" if pred == 1 else "FAKE"
    return label, conf

original = "Scientists SHOCKING discovery cures ALL disease overnight, doctors HATE this simple trick."

adversarial = "Researchers surprising finding treats every illness quickly, physicians dislike this basic method."

label1, conf1 = predict(original)
label2, conf2 = predict(adversarial)

print(f"Original text -> {label1} ({conf1:.2f}% confidence)")
print(f"Adversarial (reworded) text -> {label2} ({conf2:.2f}% confidence)")
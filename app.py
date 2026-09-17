import numpy as np
import cv2
import streamlit as st
import joblib
import easyocr
from PIL import Image


model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
reader = easyocr.Reader(['en'])


def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Increase contrast using thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


st.title("📰 AI Fake News Detector")
st.write("Paste a news article below and find out if it's Real or Fake.")

user_input = st.text_area("Enter news article text:", height=200)
uploaded_image = st.file_uploader("Or upload an image of a news article:", type=["png", "jpg", "jpeg"])

if st.button("Analyze"):
    final_text = user_input

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_container_width=True)
        
        with st.spinner("Extracting text from image..."):
            processed_image = preprocess_image(image)
            ocr_result = reader.readtext(processed_image, detail=0)
            final_text = " ".join(ocr_result)
        
        st.write("**Extracted text:**")
        st.write(final_text)

    if final_text.strip() == "":
        st.warning("Please paste some text or upload an image first.")
    else:
        input_tfidf = vectorizer.transform([final_text])
        prediction = model.predict(input_tfidf)[0]

        probability = model.predict_proba(input_tfidf)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ Prediction: REAL News")
        else:
            st.error(f"❌ Prediction: FAKE News")

        st.write(f"Confidence: {confidence:.2f}%")
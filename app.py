import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Melanoma Detector", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    # Load once & cache
    model = tf.keras.models.load_model("melanoma_cnn.h5")
    return model

model = load_model()
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

st.title("🩺 Melanoma Detection (Demo)")
st.write("Upload a mole/skin lesion image and the model will predict if it's benign or malignant.")

# Threshold control (helps tune recall vs precision)
th = st.slider("Decision threshold (malignant if probability > threshold)", 0.05, 0.95, 0.50, 0.01)

uploaded = st.file_uploader("Choose a skin image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Ensure 3 channels and resize to the size you trained on (150x150)
    img = Image.open(uploaded).convert("RGB").resize((150, 150))
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 150, 150, 3)

    # Predict
    prob = float(model.predict(arr, verbose=0)[0][0])  # probability of class "1" (malignant)
    pred_class = 1 if prob > th else 0

    st.subheader(f"Prediction: **{CLASS_NAMES[pred_class]}**")
    st.write(f"Model malignant probability: **{prob:.2f}**")
    st.caption("Tip: move the threshold slider to trade precision vs recall.")

st.markdown("---")
st.caption("⚠️ Educational demo only. Not a medical device. Not a substitute for professional diagnosis.")

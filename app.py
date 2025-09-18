import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Melanoma Detector",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------
# Load Model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("melanoma_cnn.h5")

model = load_model()
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

# ---------------------------
# App Header
# ---------------------------
st.title("🩺 Melanoma Detection App")
st.markdown(
    """
    This tool uses a **Convolutional Neural Network (CNN)** trained on mole/skin lesion images  
    to predict whether a mole is **benign** or **malignant**.

    ⚠️ **Disclaimer**: This tool is for **educational purposes only** and is **not a medical diagnostic device**.  
    Always consult a qualified healthcare professional for medical concerns.
    """
)

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a skin image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert("RGB").resize((150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    malignant_prob = float(model.predict(arr, verbose=0)[0][0])
    benign_prob = 1 - malignant_prob
    pred_class = 1 if malignant_prob > 0.5 else 0

    # Convert to percentages
    malignant_pct = malignant_prob * 100
    benign_pct = benign_prob * 100

    # Results
    st.subheader(f"🔍 Prediction: **{CLASS_NAMES[pred_class]}**")

    # Display probabilities side by side
    col1, col2 = st.columns(2)
    col1.metric(label="Benign Probability", value=f"{benign_pct:.1f}%")
    col2.metric(label="Malignant Probability", value=f"{malignant_pct:.1f}%")

    # Highlight result
    if pred_class == 1:
        st.warning("⚠️ Model flagged this image as **Malignant**. Please seek medical advice.")
    else:
        st.success("Model flagged this image as **Benign**.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Made with using Streamlit and TensorFlow")

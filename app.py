import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        /* Gradient background */
        .stApp {
            background: linear-gradient(135deg, #667eea, #764ba2);
            background-attachment: fixed;
            color: white;
        }

        /* Make titles pop */
        h1, h2, h3 {
            color: #ffffff;
            text-align: center;
        }

        /* File uploader and other widgets */
        .stUploadButton, .stRadio, .stCameraInput {
            background-color: #ffffff20 !important;
            border-radius: 10px;
            padding: 10px;
        }

        .stButton>button {
            background-color: #ffffff20;
            border: none;
            padding: 0.5em 1em;
            color: white;
            border-radius: 10px;
        }

        /* Output styling */
        .stAlert-success {
            background-color: #ffffff30;
            border-radius: 10px;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

import os
import gdown
import tensorflow as tf
import streamlit as st

# --- Download model from Google Drive if not already present ---
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "18fmCk5nkklwApa7ShzVInCiwjhnGHeW4"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# --- Class Names ---
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

# --- Preprocess Image ---
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- UI ---
st.title("🧠 Smart Garbage Classifier")
st.subheader("♻️ Classify Waste Items Using AI")
st.write("Upload an image or use your webcam to identify the type of garbage.")

input_choice = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if input_choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        st.success(f"🧠 Prediction: **{predicted_class}** ({confidence:.2%} confidence)")

elif input_choice == "Use Webcam":
    camera_image = st.camera_input("Capture a photo")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        st.success(f"🧠 Prediction: **{predicted_class}** ({confidence:.2%} confidence)")

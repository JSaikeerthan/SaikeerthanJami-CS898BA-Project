import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.evaluation.gradcam import generate_gradcam

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
)

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def format_class_name(name: str) -> str:
    """
    Convert dataset class names into a readable format.
    """

    name = name.replace("___", " - ")
    name = name.replace("_", " ")

    return name


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🌿 Plant Disease Detection System")

st.markdown(
    """
Detect plant diseases from leaf images using **EfficientNetB0 Transfer Learning**
with **Grad-CAM Explainable AI**.
"""
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("Project Information")

    st.write("**Model:** EfficientNetB0")

    st.write("**Dataset:** PlantVillage")

    st.write("**Validation Accuracy:** 97.88%")

    st.write("**Framework:** TensorFlow / Keras")

    st.divider()

    st.write("**Developer:**")
    st.write("Saikeerthan Jami")

    st.write("CS898BA")
    st.write("Image Analysis and Computer Vision")

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = Path("outputs/models/efficientnet_b0.keras")
CLASS_NAMES_PATH = Path("outputs/metadata/class_names.json")


# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Load Class Names
# --------------------------------------------------
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()

st.success("✅ Model loaded successfully.")

# --------------------------------------------------
# Upload Image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # --------------------------------------------------
    # Uploaded Image
    # --------------------------------------------------
    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True,
        )

    # --------------------------------------------------
    # Preprocess Image
    # --------------------------------------------------
    resized = image.resize((224, 224))

    img_array = np.array(
        resized,
        dtype=np.float32,
    )

    img_array = np.expand_dims(
        img_array,
        axis=0,
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    prediction = model.predict(
        img_array,
        verbose=0,
    )

    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = class_names[predicted_index]

    confidence = probabilities[predicted_index] * 100

    # --------------------------------------------------
    # Prediction Results
    # --------------------------------------------------
    with col2:

        st.subheader("Prediction")

        st.success(
            f"**Disease:** {format_class_name(predicted_class)}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%",
        )

        st.divider()

        st.subheader("Top 3 Predictions")

        top3 = np.argsort(probabilities)[::-1][:3]

        for idx in top3:

            st.write(
                f"**{format_class_name(class_names[idx])}**"
            )

            st.progress(float(probabilities[idx]))

            st.write(
                f"{probabilities[idx] * 100:.2f}%"
            )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------
    st.divider()

    st.subheader("Grad-CAM Visualization")

    heatmap, overlay = generate_gradcam(
        model,
        image,
    )

    col3, col4 = st.columns(2)

    with col3:

        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True,
        )

    with col4:

        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True,
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Plant Disease Detection using EfficientNetB0 Transfer Learning with Grad-CAM | "
)
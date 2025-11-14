import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

# -------------------------------
# Load Models Once
# -------------------------------
@st.cache_resource
def load_models():
    brain_model_path = "models/brain_tumor_model.keras"
    chest_model_path = "models/chest_xray_model.keras"

    if os.path.exists(brain_model_path):
        brain_model = load_model(brain_model_path)
    else:
        from models.build_models import build_brain_tumor_model
        brain_model = build_brain_tumor_model()

    if os.path.exists(chest_model_path):
        chest_model = load_model(chest_model_path)
    else:
        from models.build_models import build_chest_xray_model
        chest_model = build_chest_xray_model()

    return brain_model, chest_model

brain_model, chest_model = load_models()

# -------------------------------
# Image Type Detection
# -------------------------------
def detect_image_type(image: Image.Image) -> str:
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    avg_intensity = np.mean(img_array)
    aspect_ratio = image.width / image.height

    if avg_intensity < 100 and 0.8 < aspect_ratio < 1.2:
        return "Brain MRI"
    else:
        return "Chest X-ray"

# -------------------------------
# Predict Disease
# -------------------------------
def predict_disease(image: Image.Image, diagnosis_type: str) -> str:
    img = image.resize((224,224)).convert("RGB")  # match VGG16 input
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if diagnosis_type in ["Brain Tumor Detection", "Brain MRI"]:
        pred = brain_model.predict(img_array)
        return "Tumor Detected 🧠" if pred[0][0] > 0.5 else "No Tumor 👍"

    elif diagnosis_type in ["Chest X-Ray Analysis", "Chest X-ray"]:
        pred = chest_model.predict(img_array)
        return "Pneumonia Detected 😷" if pred[0][0] > 0.5 else "Normal Lungs ✅"

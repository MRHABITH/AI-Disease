import streamlit as st
from PIL import Image
from model_run.checking import (
    detect_image_type,
    predict_disease
)

# ================================================
# 🎨 STREAMLIT UI
# ================================================
st.set_page_config(page_title="AI Disease Diagnosis", page_icon="🧠", layout="centered")

st.title("🩺 AI-Based Disease Diagnosis System")
st.write("An integrated CNN-powered system to detect **Brain Tumor** and **Pneumonia** from medical images.")

# ---------------- MODE SELECTION ----------------
st.divider()
mode = st.radio("Select Mode", ["Manual Selection", "Auto Detect"])

if mode == "Manual Selection":
    diagnosis_type = st.radio("Choose Diagnosis Type", ["Brain Tumor Detection", "Chest X-Ray Analysis"])
else:
    diagnosis_type = None  # determined later

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload an MRI or X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Auto detect type
    if mode == "Auto Detect":
        diagnosis_type = detect_image_type(image)
        st.info(f"🧭 Detected Image Type: **{diagnosis_type}**")

    # Perform prediction
    if diagnosis_type:
        st.write(f"Running {diagnosis_type} model...")
        result = predict_disease(image, diagnosis_type)
        st.success(result)

st.markdown("---")
st.caption("© 2025 Neural Gen-AI Networks | AI-Powered Medical Diagnosis System")

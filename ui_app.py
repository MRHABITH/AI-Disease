import streamlit as st
from PIL import Image
from model_run.checking import predict_disease, detect_image_type

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI-Based Disease Diagnosis",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI-Based Disease Diagnosis System")
st.write("Upload a Brain MRI or Chest X-ray image and the system will automatically detect the type and predict the disease.")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (Brain MRI / Chest X-ray)", 
    type=["png","jpg","jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Auto Image Type Detection
    # -------------------------------
    with st.spinner("Detecting image type..."):
        img_type = detect_image_type(image)
    st.info(f"Detected Image Type: **{img_type}**")

    # -------------------------------
    # Auto Disease Prediction
    # -------------------------------
    with st.spinner("Analyzing image and predicting disease..."):
        result = predict_disease(image, img_type)
    st.success(f"Prediction Result: **{result}**")

    # -------------------------------
    # Deployment Notes
    # -------------------------------
    st.caption("© 2025 M Sabari | AI-Powered Medical Diagnosis System")

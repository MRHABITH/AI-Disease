import streamlit as st
from PIL import Image
from model_run.checking import predict_disease

st.set_page_config(
    page_title="AI-Based Disease Diagnosis",
    page_icon="🧠",
    layout="centered",
)

st.title("🩺 AI-Based Disease Diagnosis System")
st.write("First select the image type, then upload the image for disease prediction.")

# -------------------------------
# Step 1: Select image type
# -------------------------------
img_type = st.selectbox(
    "Select Image Type:",
    ["Select", "MRI", "X-ray"]
)

if img_type != "Select":
    st.info(f"Selected Image Type: **{img_type}**")

    # -------------------------------
    # Step 2: Upload Image
    # -------------------------------
    uploaded_file = st.file_uploader(
        f"Upload a {img_type} image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {img_type} Image", use_container_width=True)

        # -------------------------------
        # Step 3: Predict Disease
        # -------------------------------
        if st.button("Predict Disease"):
            with st.spinner("Analyzing image..."):
                result = predict_disease(image, img_type)

            st.success(f"Prediction Result: **{result}**")

# Footer
st.caption("© 2025 M Sabari | AI-Powered Medical Diagnosis System")

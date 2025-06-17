import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="YOLOv8 Segmentation", layout="wide")
st.title("🧠 YOLOv8 Segmentation App")

# Select model
model_type = st.selectbox("Choose model", ["yolov8n-seg.pt", "yolov8s-seg.pt"])

# Upload file
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Preserve file extension
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    try:
        # Load YOLOv8 model
        model = YOLO(model_type)
        st.info(f"🔍 Running inference using {model_type}...")

        # Run prediction
        results = model(temp_path)

        for result in results:
            # Convert OpenCV BGR image to RGB for Streamlit
            im_array = result.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(im_rgb), caption="Prediction", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Inference failed: {e}")

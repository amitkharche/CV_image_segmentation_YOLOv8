import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLOv8 Segmentation", layout="wide")
st.title("🧠 YOLOv8 Segmentation App")

model_type = st.selectbox("Choose model", ["yolov8n-seg.pt", "yolov8s-seg.pt"])
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    model = YOLO(model_type)
    st.info(f"Running inference using {model_type}...")

    results = model(temp_path)
    for result in results:
        im_array = result.plot()
        st.image(im_array, caption="Prediction", use_column_width=True)

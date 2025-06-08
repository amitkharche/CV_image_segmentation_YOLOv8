# 🔸 Project 10: Image Segmentation using YOLOv8

## 📌 Business Problem
In domains like autonomous driving, healthcare, retail, and surveillance, it's crucial to detect not just objects, but also precisely segment them at a pixel level. This project solves that problem using YOLOv8's segmentation capabilities, enabling real-time segmentation on both **images** and **videos**.

## 🎯 Project Scope
- Train and infer using YOLOv8 segmentation (`yolov8n-seg`, `yolov8s-seg`)
- Run segmentation on uploaded images or videos via a Streamlit app
- Evaluate segmentation using metrics: **IoU, Dice, Precision, Recall, F1**
- Visualize predictions directly in the browser
- Includes config files, Dockerfile, and GitHub Actions for CI

---

## 📁 Folder Structure

```
image-segmentation-yolo/
├── notebooks/                 # Optional training/inference notebooks
├── src/
│   ├── train.py               # YOLOv8 segmentation training script
│   ├── inference.py          # Run model inference
│   └── metrics.py            # Evaluate predicted masks
├── streamlit_app/
│   └── app.py                # Upload + visualize predictions
├── demo/                     # GIFs and output samples
├── config.yaml               # Model config (confidence, model type)
├── requirements.txt          # All Python dependencies
├── Dockerfile                # For container deployment
├── .github/workflows/ci.yml # GitHub CI
├── .gitignore
└── README.md
```

---

## 🚀 How to Use

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
```bash
python src/train.py
```

3. **Run Inference**
```bash
python src/inference.py
```

4. **Launch Streamlit App**
```bash
streamlit run streamlit_app/app.py
```

5. **View Results**  
Upload image/video → get segmented output + performance metrics.

---

## 📊 Evaluation Metrics
- **IoU (Intersection over Union)**
- **Dice Score**
- **Precision**
- **Recall**
- **F1 Score**

---

## 🖼️ Sample Streamlit UI

![demo](demo/yolo_segmentation_demo.gif)

---

## 📌 Use Cases
- Autonomous Vehicles (lane/person segmentation)
- Medical Imaging (organ/tumor segmentation)
- Retail Shelf Monitoring
- Smart Surveillance

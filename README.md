
---

# Image Segmentation using YOLOv8

## Business Problem

In domains like autonomous driving, healthcare, retail, and surveillance, it's crucial to detect not just objects, but to precisely segment them at the pixel level. This project addresses that need using YOLOv8's segmentation capabilities, enabling real-time segmentation on both images and videos.

---

## Project Scope

- Train and infer using YOLOv8 segmentation (`yolov8n-seg`, `yolov8s-seg`)
- Upload and segment images/videos via a Streamlit app
- Visualize predicted masks directly in the browser
- Evaluate segmentation manually using provided utility functions
- Includes dataset preparation, config files, and GitHub Actions for CI

---

## Folder Structure

```

CV\_IMAGE\_SEGMENTATION\_YOLOv8/
├── dataset/                    # YOLOv8 segmentation-style dataset
│   ├── images/
│   │   └── train/
│   └── labels/
│       └── train/
├── runs/segment/              # YOLOv8 training results
│   ├── train/
│   └── train2/
│       └── weights/best.pt
├── src/
│   ├── data.yaml              # Dataset config file for YOLOv8
│   ├── train.py               # Train YOLOv8 segmentation
│   ├── inference.py           # Run inference on a sample image
│   └── metrics.py             # Evaluation utilities (manual)
├── streamlit\_app/
│   └── app.py                 # Streamlit-based UI for predictions
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Optional containerization setup
├── config.yaml                # App/model config if needed
├── .github/workflows/ci.yml   # GitHub CI pipeline
├── .gitignore
└── README.md

````

---

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

---

### 2. Train the Model

Make sure your dataset is structured like this:

* `dataset/images/train/*.jpg`
* `dataset/labels/train/*.txt` (YOLOv8 polygon format)

Ensure your `src/data.yaml` looks like:

```yaml
path: dataset
train: images/train
val: images/train

nc: 1
names: ['apple']
```

Then start training:

```bash
python src/train.py
```

Output will be saved at:

```
runs/segment/train2/weights/best.pt
```

---

### 3. Run Inference

Update this line in `src/inference.py` to point to your trained model and test image:

```python
run_inference("runs/segment/train2/weights/best.pt", "sample.jpg")
```

Then run:

```bash
python src/inference.py
```

Segmented output will be saved in:

```
runs/segment/output/
```

---

### 4. Launch the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Upload a `.jpg`, `.png`, or `.mp4` file and view segmentation results live in your browser.

---

## Evaluation Metrics

The file `src/metrics.py` provides utility functions to compute the following segmentation metrics:

* IoU (Intersection over Union)
* Dice Score
* Precision
* Recall
* F1 Score

> These metrics are **not automatically used** in training or inference. You must manually provide:
>
> * The predicted mask (from YOLOv8)
> * The ground truth mask (from your dataset)

Example:

```python
from metrics import evaluate_masks

metrics = evaluate_masks(pred_mask, true_mask)
print(metrics)
```

---

## Use Cases

* Autonomous Vehicles (lane/person segmentation)
* Medical Imaging (organ/tumor segmentation)
* Retail Shelf Monitoring
* Smart Surveillance

---

## Contact

* [LinkedIn](https://www.linkedin.com/in/amit-kharche)
* [Medium](https://medium.com/@amitkharche14)
* [GitHub](https://github.com/amitkharche)

---

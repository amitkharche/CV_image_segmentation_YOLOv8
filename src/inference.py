from ultralytics import YOLO
import cv2
import os

def run_inference(model_path, input_path, save_dir="runs/segment/output"):
    model = YOLO(model_path)
    results = model(input_path, save=True, save_txt=True, project=save_dir)
    return results

if __name__ == "__main__":
    run_inference("yolov8n-seg.pt", "sample.jpg")

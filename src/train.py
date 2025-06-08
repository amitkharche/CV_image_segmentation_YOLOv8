from ultralytics import YOLO

def train_yolov8_seg(data_yaml='data.yaml', model='yolov8n-seg.pt', epochs=25, imgsz=640):
    model = YOLO(model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

if __name__ == '__main__':
    train_yolov8_seg()

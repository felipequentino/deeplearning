from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Inference
model.predict('bus.jpg', save=True, imgsz=640, conf=0.5)
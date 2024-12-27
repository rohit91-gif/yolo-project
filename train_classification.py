# Load YOLOv5n-cls, train it on mnist160 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('weights/yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='/Users/C:\Users\rajput/yolov5-silva/datasets/chair', epochs=100)  # train the model
model('inference/images/chair.jpeg')  # predict on an image
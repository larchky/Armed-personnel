from ultralytics import YOLO

import zipfile
import os

#path to yaml file
yaml_path = "armed--2.yaml"
#path to model
model_path = "yolo11s-pose.pt"

model = YOLO(model_path)
model.train(data=yaml_path,  epochs=100, imgsz=704, project = 'gun_detection',
            name = 'experiment')

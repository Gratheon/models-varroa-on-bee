import os
from ultralytics import YOLO
import yaml

data_yaml_path = "varroa8k.v1-testing.yolov11/data.yaml"

with open(data_yaml_path, 'r') as file:
    data = yaml.safe_load(file)

base_dir = os.path.abspath("varroa8k.v1-testing.yolov11")
data['train'] = os.path.join(base_dir, "train/images")
data['val'] = os.path.join(base_dir, "valid/images")
if 'test' in data:
    data['test'] = os.path.join(base_dir, "test/images")

with open(data_yaml_path, 'w') as file:
    yaml.dump(data, file)

model = YOLO('yolo11n.pt')

# Reduced memory usage: smaller batch, workers=0, imgsz=416
results = model.train(
    data=data_yaml_path, 
    epochs=10, 
    imgsz=416, 
    batch=8, 
    workers=0,
    project='varroa_detection', 
    name='varroa_model', 
    device='mps' # You can change this to 'cpu' if it still crashes
)
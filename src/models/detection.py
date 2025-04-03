from ultralytics import YOLO
from pathlib import Path

yaml_path = Path('config/detection.yaml')
output_dir = Path('models/pollution_detection')

model = YOLO('yolov8s.pt') 

results = model.train(
    data=yaml_path,  
    epochs=40, 
    imgsz=640, 
    device='cpu',                       # use device=0 for GPU
    batch=16, 
    workers=8,                         # Data loading threads
    patience=5, 
    lr0=0.0005,
    lrf=0.005,
    optimizer='AdamW', 
    cos_lr=True,                       # Cosine learning rate schedule
    dropout=0.2, 
    weight_decay=0.001,               # Prevent overfitting
    augment=True, 
    hsv_h=0.015,                       # Color augmentation
    hsv_s=0.8,                         # saturation for weak class 
    hsv_v=0.5,                         # Value augmentation
    mosaic=1.0,                        #  mosaic for context
    mixup=0.4,                         # Mixup for generalization
    close_mosaic=10,                   # Stop mosaic late to refine
    cls=5.0,                           # classification weight 
    box=10.0,                          # High box weight for recall
    dfl=3,                           # Distribution focal loss for precise boxes
    pretrained=True, 
    save=True, 
    save_period=5, 
    project=str(output_dir), 
    name='yolov8s_underwater_garbage' 
)
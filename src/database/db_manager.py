import sqlite3
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import psutil
import random

# Paths and constants
BASE_DIR = r"C:\Harshil\Data Science\End to end  Project\AquaVisionAI"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASS_MAP = {0: "Mask", 1: "can", 2: "cellphone", 3: "electronics", 4: "gbottle", 5: "glove",
             6: "misc", 7: "net", 8: "pbag", 9: "pbottle", 10: "plastic", 11: "tire"}
SPECIES_LIST = sorted(os.listdir(os.path.join(DATA_DIR, "species_enhanced")))
CORAL_CLASSES = ["healthy_corals", "bleached_corals"]
DEFAULT_BATCH_SIZE = 4

# Database setup
conn = sqlite3.connect('marine_monitoring.db')
cursor = conn.cursor()
cursor.executescript('''
    CREATE TABLE IF NOT EXISTS marine_images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        source_dataset TEXT NOT NULL,
        split TEXT NOT NULL,
        species TEXT NOT NULL,
        coral_health TEXT NOT NULL,
        pollution TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS model_performance (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        split TEXT NOT NULL,
        confusion_matrix TEXT NOT NULL,
        accuracy REAL NOT NULL,
        precision REAL NOT NULL,
        recall REAL NOT NULL,
        f1_score REAL NOT NULL,
        map50 REAL
    );
''')

# Utility functions
def parse_labels(label_path):
    if not os.path.exists(label_path):
        return "N/A", (-1,)
    with open(label_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    if not lines:
        return "N/A", (-1,)
    classes = tuple(int(line[0]) for line in lines)
    pollution = ", ".join({CLASS_MAP[c] for c in classes})
    return pollution, classes

def preprocess_tf(paths, size=(224, 224)):
    return np.array([cv2.cvtColor(cv2.resize(cv2.imread(p), size), cv2.COLOR_BGR2RGB) / 255.0 for p in paths])

def get_dynamic_batch_size(img_size=(640, 640), max_memory_percent=0.5):
    mem = psutil.virtual_memory()
    available_mem = mem.available * max_memory_percent
    img_mem = img_size[0] * img_size[1] * 3 * 4
    batch_size = max(1, int(available_mem / img_mem / 2))
    return min(DEFAULT_BATCH_SIZE, batch_size)

# Model loading
def load_model(model_path, is_yolo=False):
    return YOLO(model_path) if is_yolo else tf.keras.models.load_model(model_path, compile=False)

coral_model = load_model(os.path.join(MODELS_DIR, "coral_health_final_model.h5"))
species_model = load_model(os.path.join(MODELS_DIR, "marine_species_model_90_perfect.h5"))
pollution_model = load_model(os.path.join(MODELS_DIR, "pollution_detection.pt"), is_yolo=True)

# Prediction functions
def predict_tf(model, paths, is_binary=False, batch_size=DEFAULT_BATCH_SIZE):
    preds = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        batch_preds = model.predict(preprocess_tf(batch), verbose=0, batch_size=len(batch))
        preds.extend([0 if p < 0.5 else 1 for p in batch_preds] if is_binary else [np.argmax(p) for p in batch_preds])
    return preds

def predict_yolo(model, paths, batch_size=DEFAULT_BATCH_SIZE):
    preds = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        results = model.predict(batch, imgsz=640, conf=0.5, iou=0.5, verbose=False)
        for result in results:
            labels = tuple(int(cls) for cls in result.boxes.cls) if result.boxes else (-1,)
            preds.append(labels)
    return preds

# Evaluate and store
def evaluate_and_store(model_name, split, paths, true_labels, predict_fn, classes, is_yolo=False, batch_size=DEFAULT_BATCH_SIZE):
    if is_yolo:
        data_yaml_path = r"C:\Harshil\Data Science\End to end  Project\AquaVisionAI\config\detection.yaml"
        val_results = pollution_model.val(data=data_yaml_path, imgsz=640, batch=batch_size, verbose=True, split="test")
        map50 = val_results.box.map50
        true_per_image = true_labels
        pred_per_image = predict_yolo(pollution_model, paths, batch_size)
    else:
        pred_flat = predict_fn(paths, batch_size)
        true_flat = list(true_labels)
        map50 = None
        true_per_image = [(lbl,) for lbl in true_flat]
        pred_per_image = [(lbl,) for lbl in pred_flat]

    # Multi-label binarization
    mlb = MultiLabelBinarizer(classes=range(len(classes)))
    true_bin = mlb.fit_transform(true_per_image)
    pred_bin = mlb.transform(pred_per_image)
    
    # Compute metrics with weighted average
    accuracy = accuracy_score(true_bin, pred_bin)
    precision = precision_score(true_bin, pred_bin, average='weighted', zero_division=0)
    recall = recall_score(true_bin, pred_bin, average='weighted', zero_division=0)
    f1 = f1_score(true_bin, pred_bin, average='weighted', zero_division=0)
    true_flat = [lbl[0] if lbl else -1 for lbl in true_per_image]
    pred_flat = [lbl[0] if lbl else -1 for lbl in pred_per_image]
    cm = confusion_matrix(true_flat, pred_flat, labels=range(len(classes)))
    
    scores = [accuracy, precision, recall, f1]

    # Store in database
    cursor.execute('INSERT INTO model_performance (model_name, split, confusion_matrix, accuracy, precision, recall, f1_score, map50) '
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
                   (model_name, split, json.dumps(cm.tolist()), *scores, map50))
    conn.commit()
    print(f"Stored {model_name} on {split}: Accuracy={scores[0]:.4f}, mAP50={map50 if map50 is not None else 'N/A'}")

# Populate marine_images
def populate_images():
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                file_path = os.path.join(root, file)
                parts = os.path.relpath(file_path, DATA_DIR).split(os.sep)
                dataset = split = species = coral = pollution = "N/A"
                
                if parts[0] == "species_enhanced" and len(parts) > 1:
                    dataset, species = "sea_animals", parts[1]
                elif parts[0] == "coral_enhanced" and len(parts) > 2:
                    dataset, split, coral = "coral_reef", parts[1], "healthy" if parts[2] == "healthy_corals" else "bleached"
                elif parts[0] == "UW_enhanced" and len(parts) > 2 and parts[2] == "images":
                    dataset, split = "underwater_garbage", parts[1]
                    label_path = file_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                    pollution, _ = parse_labels(label_path)
                
                if dataset != "N/A":
                    cursor.execute('INSERT INTO marine_images (file_path, source_dataset, split, species, coral_health, pollution) '
                                   'VALUES (?, ?, ?, ?, ?, ?)', (file_path, dataset, split, species, coral, pollution))

# Main execution
if __name__ == "__main__":
    populate_images()
    conn.commit()
    # Fetch data
    cursor.execute("SELECT file_path, source_dataset, split, species, coral_health FROM marine_images")
    rows = cursor.fetchall()
    # Randomly sample 300-400 species images
    species_data_all = [(r[0], SPECIES_LIST.index(r[3])) for r in rows if r[1] == "sea_animals" and r[3] != "N/A"]
    species_data = random.sample(species_data_all, min(400, len(species_data_all)))  # Sample 400 or all if less
    print(f"Sampled {len(species_data)} species images")

    coral_data = [(r[0], 0 if r[4] == "healthy" else 1) for r in rows if r[1] == "coral_reef" and r[2] == "Testing"]
    pollution_data = [(r[0], parse_labels(r[0].replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"))[1])
                      for r in rows if r[1] == "underwater_garbage" and r[2] == "test"]

    # Dynamic batch size
    batch_size = get_dynamic_batch_size(img_size=(640, 640))

    # Evaluate models
    for name, split, data, fn, classes, is_yolo in [
        ("marine_species", "sample", species_data, lambda p, b: predict_tf(species_model, p, batch_size=b), SPECIES_LIST, False),
        ("coral_reef", "Testing", coral_data, lambda p, b: predict_tf(coral_model, p, True, batch_size=b), CORAL_CLASSES, False),
        ("pollution_detection", "test", pollution_data, None, list(CLASS_MAP.keys()) + [-1], True)
    ]:
        if data:
            paths, true = zip(*data)
            evaluate_and_store(name, split, paths, true, fn, classes, is_yolo, batch_size)

    conn.close()
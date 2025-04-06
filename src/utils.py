import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

def load_image(image_path):
    """Load an image with error handling."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image, output_path):
    """Save an image with error handling."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def preprocess_tf(image, size=(224, 224)):
    """Preprocess image for TensorFlow models."""
    img = cv2.resize(image, size)
    return img.astype(np.float32) / 255.0

def load_model(model_path, is_yolo=False):
    """Load a model with error handling."""
    try:
        return YOLO(model_path) if is_yolo else tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise ValueError(f"Failed to load model {model_path}: {str(e)}")

def parse_labels(label_path):
    """Parse YOLO label file and return a tuple of class IDs."""
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return (-1,)
    with open(label_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    return tuple(int(line[0]) for line in lines) if lines else (-1,)

def predict_tf(model, paths, is_binary=False, batch_size=8):
    """Predict using a TensorFlow model."""
    preds = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        batch_images = [preprocess_tf(load_image(p)) for p in batch]
        batch_preds = model.predict(np.array(batch_images), verbose=0)
        if is_binary:
            preds.extend([(1 if p[0] >= 0.5 else 0,) for p in batch_preds])
        else:
            preds.extend([(np.argmax(p),) for p in batch_preds])
    return preds

def predict_yolo(model, paths, batch_size=8):
    """Predict using a YOLO model."""
    preds = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        results = model.predict(batch, imgsz=640, conf=0.5, iou=0.5, verbose=False)
        for result in results:
            labels = tuple(int(cls) for cls in result.boxes.cls) if result.boxes else (-1,)
            preds.append(labels)
    return preds

def compute_metrics(true_labels, pred_labels, classes):
    """Compute evaluation metrics for multi-class or multi-label data."""
    true_labels = [lbl if isinstance(lbl, tuple) else (lbl,) for lbl in true_labels]
    pred_labels = [lbl if isinstance(lbl, tuple) else (lbl,) for lbl in pred_labels]

    if len(classes) == 2 and all(c in ["healthy_corals", "bleached_corals"] for c in classes):
        true_flat = [lbl[0] for lbl in true_labels]
        pred_flat = [lbl[0] for lbl in pred_labels]
        cm = confusion_matrix(true_flat, pred_flat, labels=[0, 1]).tolist()
        return {
            "accuracy": accuracy_score(true_flat, pred_flat),
            "precision": precision_score(true_flat, pred_flat, average='binary', zero_division=0),
            "recall": recall_score(true_flat, pred_flat, average='binary', zero_division=0),
            "f1": f1_score(true_flat, pred_flat, average='binary', zero_division=0),
            "confusion_matrix": cm
        }

    mlb = MultiLabelBinarizer(classes=range(len(classes)))
    true_bin = mlb.fit_transform(true_labels)
    pred_bin = mlb.transform(pred_labels)
    true_flat = [lbl[0] if lbl else -1 for lbl in true_labels]
    pred_flat = [lbl[0] if lbl else -1 for lbl in pred_labels]
    cm = confusion_matrix(true_flat, pred_flat, labels=range(len(classes))).tolist()
    return {
        "accuracy": accuracy_score(true_bin, pred_bin),
        "precision": precision_score(true_bin, pred_bin, average='weighted', zero_division=0),
        "recall": recall_score(true_bin, pred_bin, average='weighted', zero_division=0),
        "f1": f1_score(true_bin, pred_bin, average='weighted', zero_division=0),
        "confusion_matrix": cm
    }
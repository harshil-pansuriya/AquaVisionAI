import sqlite3
import os
import random
import json
from config.paths import DB_PATH, PROCESSED_DIR, SPECIES_MODEL_PATH, CORAL_MODEL_PATH, POLLUTION_MODEL_PATH
from src.utils import parse_labels, predict_tf, predict_yolo, compute_metrics, load_model

CLASS_MAP = {0: "Mask", 1: "can", 2: "cellphone", 3: "electronics", 4: "gbottle", 5: "glove",
             6: "misc", 7: "net", 8: "pbag", 9: "pbottle", 10: "plastic", 11: "tire"}
SPECIES_LIST = sorted(os.listdir(PROCESSED_DIR["sea_animals"]))
CORAL_CLASSES = ["healthy_corals", "bleached_corals"]
BATCH_SIZE = 8

# Database setup
conn = sqlite3.connect(DB_PATH)
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

def populate_images():
    for dataset, base_dir in PROCESSED_DIR.items():
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    file_path = os.path.join(root, file)
                    parts = os.path.relpath(file_path, base_dir).split(os.sep)
                    dataset_name = split = species = coral = pollution = "N/A"
                    
                    if dataset == "sea_animals" and len(parts) > 0:
                        dataset_name, species = "sea_animals", parts[0]
                    elif dataset == "coral_reef" and len(parts) > 1:
                        dataset_name, split, coral = "coral_reef", parts[0], "healthy" if parts[1] == "healthy_corals" else "bleached"
                    elif dataset == "underwater_garbage" and len(parts) > 1 and parts[1] == "images":
                        dataset_name, split = "underwater_garbage", parts[0]
                        label_path = file_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                        pollution_classes = parse_labels(label_path)
                        pollution = ", ".join([CLASS_MAP.get(c, "Unknown") for c in pollution_classes if c != -1]) or "N/A"
                    
                    if dataset_name != "N/A":
                        cursor.execute('INSERT INTO marine_images (file_path, source_dataset, split, species, coral_health, pollution) '
                                       'VALUES (?, ?, ?, ?, ?, ?)', (file_path, dataset_name, split, species, coral, pollution))

def evaluate_and_store(model_name, split, paths, true_labels, predict_fn, classes, is_yolo=False, model=None):
    print(f"Evaluating {model_name} on {split} split with {len(paths)} samples...")
    if is_yolo:
        val_results = model.val(data="config/detection.yaml", imgsz=640, batch=BATCH_SIZE, verbose=False, split="test")
        map50 = val_results.box.map50
        pred_labels = predict_yolo(model, paths, BATCH_SIZE)
    else:
        pred_labels = predict_fn(paths, BATCH_SIZE)
        map50 = None

    true_labels = [(lbl,) if not isinstance(lbl, tuple) else lbl for lbl in true_labels]
    metrics = compute_metrics(true_labels, pred_labels, classes)
    cursor.execute('INSERT INTO model_performance (model_name, split, confusion_matrix, accuracy, precision, recall, f1_score, map50) '
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                   (model_name, split, json.dumps(metrics["confusion_matrix"]), metrics["accuracy"], metrics["precision"],
                    metrics["recall"], metrics["f1"], map50))
    conn.commit()
    print(f"Performance saved for {model_name}: Accuracy={metrics['accuracy']:.4f}, mAP50={map50 if map50 is not None else 'N/A'}")

if __name__ == "__main__":
    print("Populating database with image metadata...")
    populate_images()
    conn.commit()
    print("Database population completed.")

    all_species_images = []
    for species in SPECIES_LIST:
        species_dir = os.path.join(PROCESSED_DIR["sea_animals"], species)
        images = [os.path.join(species_dir, f) for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.png'))]
        all_species_images.extend([(img, SPECIES_LIST.index(species)) for img in images])
    species_data = random.sample(all_species_images, min(400, len(all_species_images)))
    print(f"Sampled {len(species_data)} species images loaded.")

    cursor.execute("SELECT file_path, coral_health FROM marine_images WHERE source_dataset = 'coral_reef' AND split = 'Testing'")
    coral_rows = cursor.fetchall()
    coral_data = [(row[0], 0 if row[1] == "healthy" else 1) for row in coral_rows]
    print(f"Loaded {len(coral_data)} coral test images.")

    cursor.execute("SELECT file_path FROM marine_images WHERE source_dataset = 'underwater_garbage' AND split = 'test'")
    pollution_rows = cursor.fetchall()
    pollution_data = [(row[0], parse_labels(row[0].replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")))
                      for row in pollution_rows]
    print(f"Loaded {len(pollution_data)} pollution test images.")

    print("Loading models...")
    species_model = load_model(SPECIES_MODEL_PATH)
    print("Species model loaded.")
    coral_model = load_model(CORAL_MODEL_PATH)
    print("Coral model loaded.")
    pollution_model = load_model(POLLUTION_MODEL_PATH, is_yolo=True)
    print("Pollution model loaded.")

    for name, split, data, fn, classes, is_yolo, model in [
        ("marine_species", "random_400", species_data, lambda p, b: predict_tf(species_model, p, batch_size=b), SPECIES_LIST, False, species_model),
        ("coral_reef", "test", coral_data, lambda p, b: predict_tf(coral_model, p, True, batch_size=b), CORAL_CLASSES, False, coral_model),
        ("pollution_detection", "test", pollution_data, None, list(CLASS_MAP.keys()) + [-1], True, pollution_model)
    ]:
        if data:
            paths, true = zip(*data)
            evaluate_and_store(name, split, paths, true, fn, classes, is_yolo, model)

    conn.close()
    print("Database connection closed.")
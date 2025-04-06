# src/preprocessing/image_enhancement.py
import os
import cv2
import numpy as np
import shutil
import random
from config.paths import RAW_DIR, PROCESSED_DIR
from src.utils import load_image, save_image

DATASETS = {
    "underwater_garbage": {
        "input_dir": RAW_DIR["underwater_garbage"],
        "output_dir": PROCESSED_DIR["underwater_garbage"],
        "splits": ["train", "valid", "test"],
        "has_labels": True,
        "weak_classes": {6, 12, 13}  # metal, rod, sunglasses
    },
    "coral_reef": {
        "input_dir": RAW_DIR["coral_reef"],
        "output_dir": PROCESSED_DIR["coral_reef"],
        "splits": ["Testing", "Training", "Validation"],
        "subdirs": ["bleached_corals", "healthy_corals"],
        "has_labels": False
    },
    "sea_animals": {
        "input_dir": RAW_DIR["sea_animals"],
        "output_dir": PROCESSED_DIR["sea_animals"],
        "splits": None,
        "has_labels": False,
        "turtle_target": 500
    }
}

def enhance_image(image, dataset_type):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5 if dataset_type == "underwater_garbage" else 3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    if dataset_type == "underwater_garbage":
        img_denoised = cv2.bilateralFilter(img_enhanced, d=5, sigmaColor=25, sigmaSpace=25)
        gaussian = cv2.GaussianBlur(img_denoised, (5, 5), 1.0)
        img_final = cv2.addWeighted(img_denoised, 1.3, gaussian, -0.3, 0)
    else:
        img_final = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    if dataset_type == "coral_reef":
        for i in range(3):
            channel = img_final[:, :, i]
            min_val, max_val = np.percentile(channel, (5, 95))
            img_final[:, :, i] = np.clip((channel - min_val) * 255 / (max_val - min_val), 0, 255).astype(np.uint8)
    elif dataset_type == "sea_animals":
        avg_b, avg_g, avg_r = [np.mean(img_final[:, :, i]) for i in range(3)]
        avg_gray = (avg_b + avg_g + avg_r) / 3
        for i, avg in enumerate([avg_b, avg_g, avg_r]):
            img_final[:, :, i] = np.clip(img_final[:, :, i] * (avg_gray / avg if avg != 0 else 1), 0, 255)

    return img_final

def process_underwater_garbage(config):
    for split in config["splits"]:
        src_img_dir = os.path.join(config["input_dir"], split, "images")
        dst_img_dir = os.path.join(config["output_dir"], split, "images")
        src_label_dir = os.path.join(config["input_dir"], split, "labels")
        dst_label_dir = os.path.join(config["output_dir"], split, "labels")

        os.makedirs(dst_img_dir, exist_ok=True)
        for img_file in os.listdir(src_img_dir):
            if img_file.lower().endswith(('.jpg', '.png')):
                img = load_image(os.path.join(src_img_dir, img_file))
                enhanced = enhance_image(img, "underwater_garbage")
                save_image(enhanced, os.path.join(dst_img_dir, img_file))

        shutil.rmtree(dst_label_dir, ignore_errors=True)
        shutil.copytree(src_label_dir, dst_label_dir)

    # Remove weak classes and clean mismatches (unchanged logic)
    id_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 14: 11}
    for split in config["splits"]:
        img_dir = os.path.join(config["output_dir"], split, "images")
        label_dir = os.path.join(config["output_dir"], split, "labels")
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.png')):
                label_path = os.path.join(label_dir, f"{os.path.splitext(img_file)[0]}.txt")
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                if any(int(line.split()[0]) in config["weak_classes"] for line in lines if line.strip()):
                    os.remove(os.path.join(img_dir, img_file))
                    os.remove(label_path)

        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        img_names = set(os.path.splitext(f)[0] for f in images)
        lbl_names = set(os.path.splitext(f)[0] for f in labels)

        for img in img_names - lbl_names:
            os.remove(os.path.join(img_dir, next(i for i in images if os.path.splitext(i)[0] == img)))
        for lbl in lbl_names - img_names:
            os.remove(os.path.join(label_dir, f"{lbl}.txt"))
        for lbl in lbl_names & img_names:
            lbl_path = os.path.join(label_dir, f"{lbl}.txt")
            img_path = os.path.join(img_dir, next(i for i in images if os.path.splitext(i)[0] == lbl))
            if os.path.getsize(lbl_path) == 0:
                os.remove(lbl_path)
                os.remove(img_path)

        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(label_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    old_id = int(parts[0])
                    if old_id in id_mapping:
                        new_id = id_mapping[old_id]
                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)

def process_coral_reef(config):
    for split in config["splits"]:
        for category in config["subdirs"]:
            src_dir = os.path.join(config["input_dir"], split, category)
            dst_dir = os.path.join(config["output_dir"], split, category)
            os.makedirs(dst_dir, exist_ok=True)
            for img_file in os.listdir(src_dir):
                if img_file.lower().endswith(('.jpg', '.png')):
                    img = load_image(os.path.join(src_dir, img_file))
                    enhanced = enhance_image(img, "coral_reef")
                    save_image(enhanced, os.path.join(dst_dir, img_file))

def process_sea_animals(config):
    for species in os.listdir(config["input_dir"]):
        src_dir = os.path.join(config["input_dir"], species)
        dst_dir = os.path.join(config["output_dir"], species)
        os.makedirs(dst_dir, exist_ok=True)
        for img_file in os.listdir(src_dir):
            if img_file.lower().endswith(('.jpg', '.png')):
                img = load_image(os.path.join(src_dir, img_file))
                enhanced = enhance_image(img, "sea_animals")
                save_image(enhanced, os.path.join(dst_dir, img_file))

        if species == "Turtle_Tortoise":
            turtle_files = [f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.png'))]
            to_remove = max(0, len(turtle_files) - config["turtle_target"])
            if to_remove > 0:
                for file in random.sample(turtle_files, to_remove):
                    os.remove(os.path.join(dst_dir, file))

def main():
    for dataset, config in DATASETS.items():
        if dataset == "underwater_garbage":
            process_underwater_garbage(config)
        elif dataset == "coral_reef":
            process_coral_reef(config)
        elif dataset == "sea_animals":
            process_sea_animals(config)

if __name__ == "__main__":
    main()
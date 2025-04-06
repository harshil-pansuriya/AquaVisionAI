# src/main.py
import cv2
import numpy as np
from config.paths import SPECIES_MODEL_PATH, CORAL_MODEL_PATH, POLLUTION_MODEL_PATH
from src.utils import load_model, preprocess_tf

class MarineMonitoringSystem:
    def __init__(self):
        self.species_model = load_model(SPECIES_MODEL_PATH)
        self.coral_model = load_model(CORAL_MODEL_PATH)
        self.pollution_model = load_model(POLLUTION_MODEL_PATH, is_yolo=True)
        
        self.species_classes = ["Clams", "Corals", "Crabs", "Dolphin", "Eel", "Fish", "Jelly Fish", 
                               "Lobster", "Nudibranchs", "Octopus", "Otter", "Penguin", "Puffers", 
                               "Sea Rays", "Sea Urchins", "Seahorse", "Seal", "Sharks", "Shrimp", 
                               "Squid", "Starfish", "Turtle_Tortoise", "Whale"]
        self.pollution_classes = ["Mask", "can", "cellphone", "electronics", "gbottle", "glove", 
                                 "misc", "net", "pbag", "pbottle", "plastic", "tire"]

    def enhance_image(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image[:, :, ::-1].max() > image.max():
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_float = image.astype(float) / 255.0
        avg_r, avg_g, avg_b = np.mean(img_float, axis=(0, 1))
        gray_world = np.mean([avg_r, avg_g, avg_b])
        scale_r = min(gray_world / (avg_r + 1e-6), 1.5)
        scale_g = min(gray_world / (avg_g + 1e-6), 1.2)
        scale_b = min(gray_world / (avg_b + 1e-6), 1.1)
        img_float[:, :, 0] *= scale_r
        img_float[:, :, 1] *= scale_g
        img_float[:, :, 2] *= scale_b
        image = np.clip(img_float * 255, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        image = cv2.bilateralFilter(image, d=9, sigmaColor=30, sigmaSpace=30)
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        image = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)

        return image

    def predict_species(self, image):
        img = preprocess_tf(image)
        pred = self.species_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        max_conf = np.max(pred)
        if max_conf > 0.6:
            return self.species_classes[np.argmax(pred)], max_conf
        return "Unknown", max_conf

    def predict_coral_health(self, image):
        img = preprocess_tf(image)
        species_pred = self.species_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        coral_idx = self.species_classes.index("Corals")
        coral_conf = species_pred[coral_idx]
        
        if coral_conf > 0.5:
            coral_pred = self.coral_model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            return "bleached" if coral_pred >= 0.6 else "healthy", coral_pred
        return "No coral detected", 0.0

    def predict_pollution(self, image):
        results = self.pollution_model.predict(image, conf=0.65, iou=0.4, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.numpy()
            confs = r.boxes.conf.numpy()
            labels = r.boxes.cls.numpy().astype(int)
            for box, conf, lbl in zip(boxes, confs, labels):
                detections.append((self.pollution_classes[lbl], box, conf))
        return detections

    def process_image(self, image):
        enhanced_img = self.enhance_image(image)
        species, species_conf = self.predict_species(enhanced_img)
        coral_status, coral_conf = self.predict_coral_health(enhanced_img)
        pollution = self.predict_pollution(enhanced_img)
        return species, species_conf, coral_status, coral_conf, pollution

if __name__ == "__main__":
    system = MarineMonitoringSystem()
    img = cv2.imread("sample_image.jpg")
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        species, s_conf, coral, c_conf, pollution = system.process_image(img_rgb)
        print(f"Species: {species} ({s_conf:.2f})")
        print(f"Coral: {coral} ({c_conf:.2f})")
        print(f"Pollution: {pollution}")
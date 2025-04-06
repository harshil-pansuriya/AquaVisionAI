# config/paths.py
import os
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Base directory
BASE_DIR = config["base_dir"]

# Data paths
RAW_DIR = {k: os.path.join(BASE_DIR, v) for k, v in config["data"]["raw"].items()}
PROCESSED_DIR = {k: os.path.join(BASE_DIR, v) for k, v in config["data"]["processed"].items()}

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, config["models"]["dir"])
SPECIES_MODEL_PATH = os.path.join(MODELS_DIR, config["models"]["species"])
CORAL_MODEL_PATH = os.path.join(MODELS_DIR, config["models"]["coral"])
POLLUTION_MODEL_PATH = os.path.join(MODELS_DIR, config["models"]["pollution"])

# Database path
DB_PATH = os.path.join(BASE_DIR, config["database"]["path"])

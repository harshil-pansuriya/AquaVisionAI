import os

# List of folders to create (excluding data/)
FOLDERS = [
    "src/preprocessing",
    "src/models",
    "src/analysis",
    "src/database",
    "src/utils",
    "tests",
    "notebooks",
    "config",
]

# List of empty files to create
FILES = [
    "src/preprocessing/image_enhancement.py",
    "src/models/segmentation.py",
    "src/models/detection.py",
    "src/analysis/species_id.py",
    "src/analysis/coral_health.py",
    "src/analysis/pollution_monitor.py",
    "src/database/db_manager.py",
    "src/utils/data_loader.py",
    "src/utils/visualization.py",
    "src/utils/metrics.py",
    "src/main.py",
    "tests/test_preprocessing.py",
    "tests/test_models.py",
    "tests/test_database.py",
    "notebooks/exploratory_analysis.ipynb",
    "config/config.yaml",
    "config/paths.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Function to create folders
def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Folders created successfully!")

# Function to create empty files
def create_files(files):
    for file_path in files:
        with open(file_path, "a"):
            pass
    print("Empty files created successfully!")

# Main setup function
def setup_project():
    create_folders(FOLDERS)  # Create folders 
    create_files(FILES)   # Create empty files
    
    print("Project structure setup complete! Add your datasets to 'data/' manually.")

if __name__ == "__main__":
    setup_project()
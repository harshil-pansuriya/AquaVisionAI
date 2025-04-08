import os

FOLDERS = [
    "src/preprocessing",
    "src/training",
    "src/database",
    "tests",
    "notebooks",
    "config",
]

FILES = [
    "src/preprocessing/image_enhancement.py",
    "src/training/segmentation.py",
    "src/training/detection.py",
    "src/training/coral_reef.py",
    "src/database/db_manager.py",
    "src/main.py",
    "src/utils.py",
    "config/config.yaml",
    "config/detection.yaml",
    "config/paths.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Folders created successfully!")

def create_files(files):
    for file_path in files:
        with open(file_path, "a"):
            pass
    print("Empty files created successfully!")

def setup_project():
    create_folders(FOLDERS)  
    create_files(FILES) 
    
    print("Project structure setup complete! Add your datasets to 'data/' manually.")

if __name__ == "__main__":
    setup_project()
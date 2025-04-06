import os

# List of folders to create (excluding data/)
FOLDERS = [
    "src/preprocessing",
    "src/models",
    "src/database",
    "tests",
    "notebooks",
    "config",
]

# List of empty files to create
FILES = [
    "src/preprocessing/image_enhancement.py",
    "src/models/segmentation.py",
    "src/models/detection.py",
    "src/models/coral_reef.py",
    "src/database/db_manager.py",
    "src/main.py",
    "src/utils.py",
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
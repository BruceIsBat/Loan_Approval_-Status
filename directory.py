import os

# Define the directory structure
folders = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src"
]

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

# Create empty boilerplate files in src/
src_files = ["preprocessing.py", "train.py"]
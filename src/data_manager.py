import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path("data")
IMG_SIZE = (224, 224)

def save_image(image_file, class_name):
    """Saves an uploaded file to the data directory."""
    class_dir = DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(image_file)
        img.verify() # Verify it's an image
        image_file.seek(0) # Reset file pointer
        img = Image.open(image_file) # Re-open for saving
        
        # Generate a unique filename
        filename = f"{len(os.listdir(class_dir))}_{image_file.name}"
        file_path = class_dir / filename
        
        img.save(file_path)
        return str(file_path)
    except Exception as e:
        return None

def save_captured_image(image_file, class_name):
    """Saves a captured camera image to the data directory."""
    class_dir = DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # image_file from st.camera_input is a file-like object
        img = Image.open(image_file)
        
        filename = f"cam_{len(os.listdir(class_dir))}.jpg"
        file_path = class_dir / filename
        
        img.save(file_path)
        return str(file_path)
    except Exception as e:
        return None

def load_dataset():
    """Loads all images from the data directory."""
    images = []
    labels = []
    
    if not DATA_DIR.exists():
        return np.array([]), np.array([]), []

    class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        for img_path in class_dir.glob("*"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_arr = np.array(img) / 255.0
                images.append(img_arr)
                labels.append(class_map[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return np.array(images), np.array(labels), class_names

def get_class_counts():
    """Returns a dictionary of class counts."""
    counts = {}
    if not DATA_DIR.exists():
        return counts
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            counts[d.name] = len(list(d.glob("*")))
    return counts

def clear_data():
    """Clears all data."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()

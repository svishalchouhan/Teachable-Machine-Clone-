import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODELS_DIR = "models"
IMG_SIZE = (224, 224)

def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        return None
        
    if model_name.endswith(".joblib"):
        return joblib.load(path)
    elif model_name.endswith(".h5"):
        return tf.keras.models.load_model(path)
    return None

def preprocess_image(image):
    # image is PIL Image
    img = image.resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    return img_arr

def predict(image, models_dict):
    # image is PIL Image
    img_arr = preprocess_image(image)
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_arr, axis=0)
    img_flat = img_batch.reshape(1, -1)
    
    results = {}
    
    for name, model in models_dict.items():
        if model is None:
            results[name] = None
            continue
            
        try:
            if name == "CNN":
                probs = model.predict(img_batch)[0]
            else:
                # Sklearn models expect flattened input
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(img_flat)[0]
                else:
                    # Fallback
                    probs = model.predict(img_flat)[0]
                    
            results[name] = probs
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            results[name] = None
        
    return results

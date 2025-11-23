# Teachable Machine Clone (Python/Streamlit)

A web application that allows users to create image classifiers using their own data, train multiple models (Logistic Regression, Random Forest, CNN), and test them in real-time.

## Features
- **Data Collection**: Upload images or capture from webcam for multiple classes.
- **Model Training**: Train Logistic Regression, Random Forest, and CNN models.
- **Evaluation**: View accuracy and confusion matrices for each model.
- **Inference**: Real-time predictions using uploaded images or webcam feed.
- **Comparison**: Side-by-side probability visualization for all trained models.

## Project Structure
- `app.py`: Main Streamlit application.
- `src/`:
    - `data_manager.py`: Handles image storage and dataset loading.
    - `trainers.py`: Contains training logic for Sklearn and TensorFlow models.
    - `inference.py`: Handles model loading and prediction.
- `data/`: Stores user-uploaded images organized by class.
- `models/`: Stores trained model files (`.joblib`, `.h5`).

## Installation

1. Install dependencies:
   ```bash
   pip install streamlit scikit-learn tensorflow opencv-python-headless pandas matplotlib numpy pillow
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Data Collection**: Go to the "Data Collection" tab. Add classes (e.g., "Cat", "Dog") and upload/capture images for each.
2. **Training**: Go to the "Training" tab. Select models and click "Start Training".
3. **Inference**: Go to the "Inference" tab. Use the webcam or upload an image to see predictions.

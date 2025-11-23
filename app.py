import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from src import data_manager, trainers, inference

st.set_page_config(page_title="Teachable Machine Clone", layout="wide")

st.title("Teachable Machine Clone 🤖")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Data Collection", "Training", "Inference"])

# --- Data Collection ---
if app_mode == "Data Collection":
    st.header("1. Data Collection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add Class")
        new_class = st.text_input("Enter Class Name")
        if st.button("Add Class"):
            if new_class:
                (data_manager.DATA_DIR / new_class).mkdir(parents=True, exist_ok=True)
                st.success(f"Class '{new_class}' added!")
                st.rerun()

        st.divider()
        
        class_counts = data_manager.get_class_counts()
        if class_counts:
            selected_class = st.selectbox("Select Class", list(class_counts.keys()))
            
            # File Uploader
            uploaded_files = st.file_uploader(f"Upload images for {selected_class}", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
            if uploaded_files:
                count = 0
                for uploaded_file in uploaded_files:
                    if data_manager.save_image(uploaded_file, selected_class):
                        count += 1
                if count > 0:
                    st.success(f"Saved {count} images to {selected_class}")
                else:
                    st.error("Failed to save images. Invalid format?")
            
            # Webcam
            st.write("Or capture from webcam:")
            cam_img = st.camera_input(f"Capture for {selected_class}", key=f"cam_{selected_class}")
            if cam_img:
                if data_manager.save_captured_image(cam_img, selected_class):
                    st.success(f"Captured image for {selected_class}")
                else:
                    st.error("Failed to save captured image.")

            if st.button(f"Delete Class '{selected_class}'"):
                shutil.rmtree(data_manager.DATA_DIR / selected_class)
                st.warning(f"Deleted class '{selected_class}'")
                st.rerun()
                
    with col2:
        st.subheader("Current Data")
        if class_counts:
            st.bar_chart(class_counts)
            
            # Show sample images
            st.write(f"### Samples for {selected_class}")
            class_dir = data_manager.DATA_DIR / selected_class
            if class_dir.exists():
                images = list(class_dir.glob("*"))
                if images:
                    # Display in a grid
                    cols = st.columns(5)
                    for i, img_path in enumerate(images[:10]):
                        with cols[i % 5]:
                            st.image(str(img_path), use_column_width=True)
        else:
            st.info("No classes added yet. Add a class to get started.")
            
    if st.sidebar.button("Reset All Data", type="primary"):
        data_manager.clear_data()
        st.rerun()

# --- Training ---
elif app_mode == "Training":
    st.header("2. Model Training")
    
    class_counts = data_manager.get_class_counts()
    if len(class_counts) < 2:
        st.warning("You need at least 2 classes to train models.")
    else:
        st.write("Classes found:", list(class_counts.keys()))
        
        st.subheader("Select Models to Train")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
        with col3:
            use_cnn = st.checkbox("CNN (Deep Learning)", value=True)
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Loading and preprocessing data..."):
                X, y, class_names = data_manager.load_dataset()
            
            if len(X) == 0:
                st.error("No data found!")
            else:
                st.write(f"Loaded {len(X)} images. Training started...")
                
                # Train Logistic Regression
                if use_lr:
                    st.subheader("Logistic Regression")
                    with st.spinner("Training Logistic Regression..."):
                        acc, cm = trainers.train_logistic_regression(X, y)
                        st.success(f"Accuracy: {acc:.2f}")
                        
                        fig, ax = plt.subplots()
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.figure.colorbar(im, ax=ax)
                        ax.set(xticks=np.arange(cm.shape[1]),
                               yticks=np.arange(cm.shape[0]),
                               xticklabels=class_names, yticklabels=class_names,
                               title='Confusion Matrix (LR)',
                               ylabel='True label',
                               xlabel='Predicted label')
                        st.pyplot(fig)

                # Train Random Forest
                if use_rf:
                    st.subheader("Random Forest")
                    with st.spinner("Training Random Forest..."):
                        acc, cm = trainers.train_random_forest(X, y)
                        st.success(f"Accuracy: {acc:.2f}")
                        
                        fig, ax = plt.subplots()
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
                        ax.figure.colorbar(im, ax=ax)
                        ax.set(xticks=np.arange(cm.shape[1]),
                               yticks=np.arange(cm.shape[0]),
                               xticklabels=class_names, yticklabels=class_names,
                               title='Confusion Matrix (RF)',
                               ylabel='True label',
                               xlabel='Predicted label')
                        st.pyplot(fig)

                # Train CNN
                if use_cnn:
                    st.subheader("CNN Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    acc, cm, history = trainers.train_cnn(X, y, len(class_names), st_progress_bar=progress_bar, st_status_text=status_text)
                    st.success(f"CNN Trained! Final Accuracy: {acc:.2f}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # Plot history
                        fig, ax = plt.subplots()
                        ax.plot(history.history['accuracy'], label='accuracy')
                        ax.plot(history.history['val_accuracy'], label = 'val_accuracy')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Accuracy')
                        ax.legend(loc='lower right')
                        ax.set_title("Training History")
                        st.pyplot(fig)
                    
                    with col_b:
                        # Confusion Matrix
                        fig, ax = plt.subplots()
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
                        ax.figure.colorbar(im, ax=ax)
                        ax.set(xticks=np.arange(cm.shape[1]),
                               yticks=np.arange(cm.shape[0]),
                               xticklabels=class_names, yticklabels=class_names,
                               title='Confusion Matrix (CNN)',
                               ylabel='True label',
                               xlabel='Predicted label')
                        st.pyplot(fig)

# --- Inference ---
elif app_mode == "Inference":
    st.header("3. Inference")
    
    # Load models
    @st.cache_resource
    def get_models():
        return {
            "Logistic Regression": inference.load_model("logistic_regression.joblib"),
            "Random Forest": inference.load_model("random_forest.joblib"),
            "CNN": inference.load_model("cnn_model.h5")
        }
    
    models_dict = get_models()
    
    # Filter out None models
    models_dict = {k: v for k, v in models_dict.items() if v is not None}
    
    if not models_dict:
        st.warning("No trained models found. Please train models first.")
    else:
        st.write("Active Models:", ", ".join(list(models_dict.keys())))
        
        # Get class names
        class_counts = data_manager.get_class_counts()
        class_names = sorted(list(class_counts.keys()))
        
        col_input, col_pred = st.columns([1, 1])
        
        input_image = None
        with col_input:
            st.subheader("Input")
            input_method = st.radio("Input Method", ["Upload Image", "Live Webcam"])
            
            if input_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
                if uploaded_file is not None:
                    input_image = Image.open(uploaded_file).convert('RGB')
                    st.image(input_image, caption='Uploaded Image', use_column_width=True)
                    
            elif input_method == "Live Webcam":
                cam_img = st.camera_input("Take a picture")
                if cam_img is not None:
                    input_image = Image.open(cam_img).convert('RGB')
            
        with col_pred:
            st.subheader("Predictions")
            if input_image is not None:
                results = inference.predict(input_image, models_dict)
                
                for model_name, probs in results.items():
                    if probs is not None:
                        st.write(f"**{model_name}**")
                        # Create a dataframe for the bar chart
                        df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probs
                        })
                        st.bar_chart(df.set_index('Class'))
            else:
                st.info("Provide an image to see predictions.")

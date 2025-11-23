import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def train_logistic_regression(X, y):
    if len(np.unique(y)) < 2:
        return 0.0, None

    X_flat = flatten_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Specify labels to ensure proper dtype and shape
    labels = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    return acc, cm

def train_random_forest(X, y):
    if len(np.unique(y)) < 2:
        return 0.0, None

    X_flat = flatten_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Specify labels to ensure proper dtype and shape
    labels = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    return acc, cm

def train_cnn(X, y, num_classes, st_progress_bar=None, st_status_text=None):
    if len(np.unique(y)) < 2:
        return 0.0, None, None

    # X is (N, 224, 224, 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train for a few epochs for demo purposes
    callbacks = []
    epochs = 10
    if st_progress_bar and st_status_text:
        callbacks.append(StreamlitCallback(st_progress_bar, st_status_text, epochs))

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0, callbacks=callbacks)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Specify labels to ensure proper dtype and shape
    labels = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    model.save(os.path.join(MODELS_DIR, "cnn_model.h5"))
    return acc, cm, history

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")


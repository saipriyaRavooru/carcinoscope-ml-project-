import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Path to your trained model
model_path = r"C:\Users\MBU\Desktop\ml project\carcinoscope\skin_cancer_cnn_model.h5"

# Load the model
print("🧠 Loading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Labels — these correspond to the 7 HAM10000 classes
class_labels = [
    'akiec',  # Actinic keratoses
    'bcc',    # Basal cell carcinoma
    'bkl',    # Benign keratosis-like lesions
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic nevi
    'vasc'    # Vascular lesions
]

# Path to the test image
img_path = r"C:\Users\MBU\Desktop\ml project\carcinoscope\test2.jpg"  # change this as needed

# Load and preprocess the image
print("🖼️ Loading test image...")
img = image.load_img(img_path, target_size=(128, 128))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # normalize

# Predict
print("🔍 Making prediction...")
pred = model.predict(x)
pred_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred) * 100

# Output
print(f"\n✅ Predicted Class: {class_labels[pred_class]}")
print(f"💡 Confidence: {confidence:.2f}%")

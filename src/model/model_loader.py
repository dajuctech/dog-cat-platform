import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing import image

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_model(model_path='models/saved_model'):
    """
    Load a saved TensorFlow model.
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

def predict_image(model, img_path, target_size=(300, 300)):
    """
    Predict the class of a single image.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print(f"Prediction: {prediction[0][0]:.4f}")
    return prediction[0][0]

if __name__ == "__main__":
    model = load_model('models/saved_model')
    img_path = 'path/to/test/image.jpg'  # Update with actual image path
    predict_image(model, img_path)

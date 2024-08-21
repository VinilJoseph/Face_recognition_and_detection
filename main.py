import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from keras.saving import register_keras_serializable
import json

# Define and register the l2_normalize function
@register_keras_serializable()
def l2_normalize(x, axis=-1, epsilon=1e-10):
    return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)

# Define the scaling function using TensorFlow operations
def scaling(x):
    x = tf.image.resize(x, (160, 160))  # Resize using TensorFlow
    x = tf.cast(x, tf.float32)
    x = (x - 127.5) / 128.0
    return x

# Load the trained FaceNet model with the custom objects
model_loaded = tf.keras.models.load_model('custom_facenet_model.h5',
                          custom_objects={'scaling': scaling, 'l2_normalize': l2_normalize})

# model_loaded.summary()

# Load the class indices
with open('class_indices.json', 'r') as f:
    class_label = json.load(f)

# Invert the dictionary to map indices to class labels
class_label = {v: k for k, v in class_label.items()}

# Preprocessing function (same as used during training)
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))  # Resize to 160x160
    image = image.astype('float32')
    image = (image - 127.5) / 128.0  # Normalize
    return image


test_tensor = tf.random.uniform((1, 224, 224, 3))  # Example tensor
scaled_tensor = scaling(test_tensor)
print(scaled_tensor.shape)


test_image = np.random.random((1, 160, 160, 3)).astype('float32')
predictions = model_loaded.predict(test_image)
print(predictions)


# Streamlit app
st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

    image = preprocess_image(image)  # Resize to (160, 160) and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model_loaded.predict(image)

    # Predict the identity using the loaded model
    predictions = model_loaded.predict(image)
    class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = class_label[class_idx]

    st.write(f"Predicted identity: {predicted_label}")
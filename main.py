import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Load the saved model
model = load_model('custom_facenet_model.h5')

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (160, 160))  # FaceNet expects 160x160 images
    img = img.astype('float32')
    img = (img - 127.5) / 128.0
    return img

# Function to predict the class of the image
def predict_face(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return np.argmax(predictions, axis=1)[0]

# Streamlit app
st.title('Face Detection and Recognition App')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img = np.array(img)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Predict the face
    class_index = predict_face(img)
    st.write(f'Predicted class index: {class_index}')
    
    # Optionally, display the class name if you have a mapping
    class_names = train_generator.class_indices  # Assuming you have the same generator available
    class_names = {v: k for k, v in class_names.items()}  # Reverse the dictionary
    predicted_class = class_names.get(class_index, 'Unknown')
    st.write(f'Predicted class name: {predicted_class}')

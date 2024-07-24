import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# Loading the model
model = load_model("dog_breed_classification_model.h5")

# Name of classes
CLASS_NAMES = ["Scottish Deerhound", "Maltese Dog", "Bernese Mountain Dog"]

# Setting title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an image..", type="png")
submit = st.button("Predict")

# On predict button click

if submit:

    if dog_image is not None:

        # convert the file to an opencv image

        file_byte = np.asanyarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_byte, 1)

        # Display the image
        st.image(opencv_image, channels="BGR")

        # Resize the image
        opencv_image = cv2.resize(opencv_image, (224, 224))

        # convert image to 4 dimension
        opencv_image.shape = (1, 224, 224, 3)

        # Make prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The dog breed is " + CLASS_NAMES[np.argmax(Y_pred)]))

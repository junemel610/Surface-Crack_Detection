import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

@st.cache_resource

def load_model_from_file():
  model=tf.keras.models.load_model('SurfaceCrackDetection2.h5')
  return model
model=load_model_from_file()
classes = {0: 'Visible Crack/s', 1: 'No Visible Crack/s'}
st.write("""
# Surface Crack Detection System"""
)
file=st.file_uploader("Choose a photo from computer",type=["jpg","png"])

def import_and_predict(image_data, model):
    size = (120,120)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            class_names = classes
            string = f"Surface: {class_names[np.argmax(prediction)]}!"
            st.success(string)
        else:
            st.text("The file is invalid. Upload a valid image file.")
    except Exception as e:
        st.text("An error occurred while processing the image.")
        st.text(str(e))
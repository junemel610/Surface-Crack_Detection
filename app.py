import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

@st.cache(allow_output_mutation=True)
def load_model_from_file():
    model = tf.keras.models.load_model('SurfaceCrackDetection2.h5')
    return model

model = load_model_from_file()

st.write("""
# Surface Crack Detection System
""")
file = st.file_uploader("Choose a photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (120, 120)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    if len(img.shape) == 2:  # Convert grayscale images to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_reshape = np.reshape(img, (1, 120, 120, 3))
    prediction = model.predict(img_reshape)
    return prediction[0][0]

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            if prediction >= 0.5:
                result = 'No Visible Crack/s'
            else:
                result = 'Visible Crack/s'
            string = f"Surface: {result}!"
            st.success(string)
        else:
            st.text("The file is invalid. Upload a valid image file.")
    except Exception as e:
        st.text("An error occurred while processing the image.")
        st.text(str(e))

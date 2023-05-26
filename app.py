import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

st.set_page_config(page_title="Surface Crack Detection App")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model_from_file():
    model = tf.keras.models.load_model('SurfaceCrackDetection2.h5')
    return model

model = load_model_from_file()

# Define class labels
classes = {0: 'No Crack', 1: 'Crack'}

# Set app title and layout
st.title("Surface Crack Detection System")
st.markdown("---")

# Create sidebar
st.sidebar.title("Upload Images")
file = st.sidebar.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

# Process the uploaded image
def import_and_predict(image_data, model):
    size = (120, 120)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = np.reshape(img, (1, 120, 120, 3))
    prediction = model.predict(img_reshape)
    return prediction[0][0]

# Main app content
if file is None:
    st.markdown("Please upload an image file.")
else:
    try:
        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)

            # Perform prediction
            prediction = import_and_predict(image, model)
            if prediction >= 0.5:
                result = 'Crack'
            else:
                result = 'No Crack'

            # Display prediction result
            st.markdown("---")
            st.markdown(f"**Surface Condition:** {result}")
            st.success(f"The uploaded surface image is classified as **{result}**.")

        else:
            st.markdown("The file is invalid. Please upload a valid image file.")

    except Exception as e:
        st.markdown("An error occurred while processing the image.")
        st.markdown(f"Error Details: {str(e)}")

st.markdown("---")
st.markdown("Developed by Andrea Faith Alimorong and Meljune Royette G. Go")

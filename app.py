import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2

@st.cache(allow_output_mutation=True)
def load_saved_model():
  model=tf.keras.models.load_model('SurfaceCrackDetection.h5')
  return model
model=load_model()
classes = {0: 'Positive', 1: 'Negative'}
st.write("""
# Surface Crack Detection System"""
)
file=st.file_uploader("Choose a photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(120,120)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names= classes
    #class_names=['Looks like there is a crack on that image you just provided',
    #             'Looks like there is no crack on that image you just provided']
    string="PREDICTION : "+class_names[np.argmax(prediction)]
    st.success(string)
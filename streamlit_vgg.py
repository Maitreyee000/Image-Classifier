from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import re
import base64


import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")



model=load_model("https://drive.google.com/file/d/1XfFJOQsDBsnd0fvEhCz18IxmJdR8UmSL/view?usp=share_link")
#model = create_model()
#from keras.utils.data_utils import get_file
#weights_path = get_file(
#            'model.h5',
#            'https://drive.google.com/file/d/1XfFJOQsDBsnd0fvEhCz18IxmJdR8UmSL/view?usp=share_link')
#model.load_model(weights_path)
#load weights of the trained model.
#input_shape = (224, 224, 3)
#optim_1 = Adam(learning_rate=0.0001)
#n_classes=6
# vgg_model = model(input_shape, n_classes, optim_1, fine_tune=2)
# vgg_model.load_weights('/content/drive/MyDrive/vgg/tune_model19.weights.best.hdf5')

# prediction on model
#vgg_preds = vgg_model.predict(img)
#vgg_pred_classes = np.argmax(vgg_preds, axis=1)

 
st.markdown('<h1 style="color:black;">Vgg 19 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> street,  buildings, forest, sea, mountain, glacier</h3>', unsafe_allow_html=True)
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('/content/background.webp')

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  img= Image.open(upload)
  #img= np.asarray(im)
  #image= cv2.resize(img,(180, 180))
  #img= preprocess_input(image)
  img = img.resize((180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch
  #img= np.expand_dims(img, 0)
  class_names=['authentic', 'fake']
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  pred_proba=100 * np.max(score)
  pred_class=class_names[np.argmax(score)]
  c1.header('Input Image')
  c1.image(img)
  c1.write(img.shape)

c2.header('Output')
c2.subheader('Predicted class :')
c2.write(str(pred_class))

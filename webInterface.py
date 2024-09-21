# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 03:01:03 2024

@author: tanyi
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('CK_fruit_vegetable_model.h5')

# Assuming these are the classes from your training data
class_indices = {0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bell pepper', 4: 'Cabbage',
                 5: 'Capsicum', 6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper',
                 9: 'Corn', 10: 'Cucumber', 11: 'Eggplant', 12: 'Garlic',
                 13: 'Ginger', 14: 'Grapes', 15: 'Jalepeno', 16: 'Kiwifruit',
                 17: 'Lemon', 18: 'Lettuce', 19: 'Mango', 20: 'Onion',
                 21: 'Orange', 22: 'Paprika', 23: 'Pear', 24: 'Peas',
                 25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato', 28: 'Raddish',
                 29: 'Soy beans', 30: 'Spinach', 31: 'Sweetcorn', 32: 'Sweetpotato',
                 33: 'Tomato', 34: 'Turnip', 35: 'Watermelon'}   # Update according to your classes

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model's input shape
    image = np.array(image)  # Convert image to array
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the fruit/vegetable
def predict_fruit_vegetable(image):
    prediction = model.predict(image)  # Get model prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get index of the class with highest probability
    predicted_label = class_indices[predicted_class[0]]  # Get the predicted label
    return predicted_label

# CSS for this web app
app_css = '''
<style>

.stApp {
    background-color: #E5F9E0;  /* Light Green for freshness */
}

.stImage {
    display: flex;
    justify-content: center; 
    align-items: center; 
    margin-bottom: 20px;    
}


/* Set header and title text color to dark green */
h1 {
    color: #2E7D32;  /* Dark Green for headings */
    text-align: center;
    font-size: 4.5em;
    margin-top: -80px;
    margin-right: -30px;
}

h3 {
    color: #205622;
    margin-top: -25px;
    text-align: center;
    margin-right: -33px;
    font-size: 1.3em;
}

    .stButton>button {
        background-color: #EF5350;  /* Bright Red for buttons */
        color: white;
        border-radius: 8px;
}

/* Center text inside the file uploader */
    .stFileUploader label {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        color: #333333;  /* You can change the text color here if needed */
}

/* Modify text and paragraph font color */
.stMarkdown p, .stMarkdown h1 {
    color: #333333;  /* Charcoal Gray for body text */
}



</style>
'''

st.markdown(app_css, unsafe_allow_html=True)

# Streamlit app

st.image("banner.jpg")

st.logo("icon.png")

st.image("logoBody.jpeg", width=350)

st.title('CropSnap')
st.subheader('Powered by MobileNetV2')

st.write("With AI image recognition, identifying fruits and vegetables from images becomes easier, enabling more accurate and efficient food classification, which benefits both consumers and online platforms.")

st.markdown(
    """
    
    <hr>

    """, unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Upload an image of a fruit or vegetable below and see what is it called.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and predict
    processed_image = preprocess_image(image)
    predicted_label = predict_fruit_vegetable(processed_image)

    # Display prediction
    st.write(f'The fruit/vegetable in this image is called {predicted_label}')
    
    cropFact_url = "https://fruitsandveggies.org/fruits-and-veggies/" + predicted_label    

    st.link_button("Want to learn more about it? Click me!", cropFact_url)
    
st.image("footerIMG.jpg")
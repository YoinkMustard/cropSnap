# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 03:59:03 2024

@author: tanyi
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Caching the model loading to improve performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('CK_fruit_vegetable_model.h5')

# Load the model
model = load_model()

# Assuming these are the classes from your training data
class_indices = {
    0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bell pepper', 4: 'Cabbage',
    5: 'Capsicum', 6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper',
    9: 'Corn', 10: 'Cucumber', 11: 'Eggplant', 12: 'Garlic', 13: 'Ginger',
    14: 'Grapes', 15: 'Jalepeno', 16: 'Kiwifruit', 17: 'Lemon', 18: 'Lettuce',
    19: 'Mango', 20: 'Onion', 21: 'Orange', 22: 'Paprika', 23: 'Pear', 24: 'Peas',
    25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato', 28: 'Raddish', 29: 'Soy beans',
    30: 'Spinach', 31: 'Sweetcorn', 32: 'Sweetpotato', 33: 'Tomato', 34: 'Turnip',
    35: 'Watermelon'
}

# Image preprocessing function
IMG_SIZE = 224

def preprocess_image(image):
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        st.warning('Image is not in RGB format, converting to RGB.')
    
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize image to match model's input shape
    image = np.array(image)  # Convert image to array
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the fruit/vegetable
def predict_fruit_vegetable(image):
    try:
        prediction = model.predict(image)  # Get model prediction
        predicted_class = np.argmax(prediction, axis=1)  # Get index of the class with highest probability
        confidence = np.max(prediction) * 100 # Get the confidence level and convert to percentage
        predicted_label = class_indices.get(predicted_class[0], 'Unknown')  # Get the predicted label
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return 'Error'
    return predicted_label, confidence

# CSS for styling the web app
app_css = '''
<style>

.stApp {
    background-color: #E5F9E0;  /* Light Green for freshness */
}

.header-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.logo-container {
    width: 150px;  /* Adjust this value to change logo size */
    margin-bottom: 20px;
}

.title-container {
    text-align: center;
}

h1 {
    color: #2E7D32;  /* Dark Green for headings */
    text-align: center;
    font-size: 3.5em;
    margin-top: -55px;
    margin-right: -30px;
    line-height: 1.2;
}

h3 {
    color: #205622;
    text-align: center;
    font-size: 1.2em;
    margin-top: -22px;
    margin-right: -30px;
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
    color: #333333;
}

/* Center text inside the camera */
.stCameraInput label {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    padding-top: 30px;
    color: #333333;
}

/* Modify text and paragraph font color */
.stMarkdown p, .stMarkdown h1 {
    color: #333333;  /* Charcoal Gray for body text */
}

@media (max-width: 768px) {
    .header-container {
        flex-direction: column;
        align-items: center;
    }

    .logo-container {
       width: 100px;  /* Smaller logo on mobile */
    }

    .title-container {
        text-align: center;
    }

    h1 {
        font-size: 2.5em;
    }

    h3 {
        font-size: 1em;
    }
}

</style>
'''

# Applying the CSS
st.markdown(app_css, unsafe_allow_html=True)

# Streamlit app
st.image("banner.jpg", use_column_width=True)

# Create a container for centering
container = st.container()

# Center the logo and title
with container:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo
        st.image("logoBody.jpeg", use_column_width=True)
        
        # Title
        st.title('CropSnap')
        st.subheader('Powered by MobileNetV2')
  
st.write("""
With AI image recognition, identifying fruits and vegetables from images becomes easier, enabling more accurate 
and efficient food classification, which benefits both consumers and online platforms.
""")

st.markdown("<hr>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload an image of a fruit or vegetable below and see what it is called.", 
                                 type=["jpg", "jpeg", "png"])

# Add camera input
camera_image = st.camera_input("Or take a picture")

if uploaded_file is not None or camera_image is not None:
    try:
        # Determine which image to use
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            caption = 'Uploaded Image'
        else:
            image = Image.open(camera_image)
            caption = 'Captured Image'

        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make prediction with a loading spinner
        with st.spinner('Processing image and making prediction...'):
            processed_image = preprocess_image(image)
            predicted_label, confidence = predict_fruit_vegetable(processed_image)

        # Display prediction
        if predicted_label != 'Error':
            st.success(f'The fruit/vegetable in this image is called: **{predicted_label}**')
            st.info(f'Confidence: {confidence:.2f}%')

            # Create a progress bar for confidence
            st.progress(confidence / 100)
            
            # Generate a link to learn more about the predicted fruit/vegetable
            predicted_label_sanitized = predicted_label.lower().replace(' ', '_')
            cropFact_url = f"https://en.wikipedia.org/wiki/{predicted_label_sanitized}"
            
            if confidence >= 60:
                # Display clickable link
                st.markdown(f"""
                    <div style="display: flex; justify-content: center;">
                        <a href="{cropFact_url}" target="_blank">
                            <button style="
                                background-color: #a4c3d; /* Pastel Blue */
                                color: white;
                                padding: 20px 30px;                            
                                font-size: 16px;
                                border: none;
                                border-radius: 12px;
                                cursor: pointer;
                                text-decoration: none;">
                                Want to learn more about it? Click me!
                            </button>
                        </a>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.warning(f"The model is not very confident about this prediction, with a confidence level of just {confidence:.2f}%. You might want to try uploading a clearer image or a different angle. Don't worry, AI can be a little unsure sometimes!")
                st.warning(f"FYI the model isn't informed by its creator that some others the fruits and vegetables exists, stay tuned for more diverse crop scanning!")

            
            
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown(
    """
    <div style="padding-top: 80px;"></div>
    """, unsafe_allow_html=True)

st.image("footerIMG.jpg")
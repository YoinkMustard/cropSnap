# cropSnap
AI-powered image classification for fruits and vegetables. Using MobileNetV2

# CropSnap - AI-based Fruit and Vegetable Identifier

CropSnap is a Streamlit web application that uses a pre-trained TensorFlow model to identify fruits and vegetables from uploaded images. The app provides accurate and efficient food classification, leveraging AI image recognition powered by MobileNetV2.

## Features
- **Fruit & Vegetable Classification**: Upload an image of a fruit or vegetable, and the app predicts what it is.
- **Learn More**: A link is provided to learn more about the identified fruit or vegetable.
- **Responsive Design**: Custom CSS for a visually appealing and user-friendly interface.

## How It Works
1. Upload an image of a fruit or vegetable.
2. The app preprocesses the image and sends it to the trained model for classification.
3. The predicted name of the fruit or vegetable is displayed.
4. Users can click a link to learn more about the item.

## Model
The app uses a TensorFlow model (`CK_fruit_vegetable_model.h5`) trained to classify various fruits and vegetables based on images. The model is based on MobileNetV2 for efficient and accurate image recognition.

## Technologies Used
- **Streamlit**: For building the web app.
- **TensorFlow**: For deep learning and image classification.
- **Pillow**: For image processing.
- **NumPy**: For numerical computations.
- **Custom CSS**: For enhanced UI styling.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YoinkMustard/CropSnap.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CropSnap
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Requirements
Make sure you have the following installed:
- Python 3.x
- Streamlit
- TensorFlow
- Pillow
- NumPy

These are also listed in the `requirements.txt` file.

## Usage
- Visit the deployed version of the app [here](https://cropsnap.streamlit.app/).
- Upload an image of a fruit or vegetable to get a prediction.
- Explore more about the identified item by clicking the provided link.

## File Structure
```bash
├── app.py                  # Main Streamlit application file
├── CK_fruit_vegetable_model.h5 # Pre-trained TensorFlow model
├── banner.jpg              # Header banner image
├── icon.png                # Logo icon image
├── logoBody.jpeg           # Body logo image
├── footerIMG.jpg           # Footer image
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

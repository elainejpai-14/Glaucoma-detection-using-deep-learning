import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import load_model

# Function to load and preprocess image
def preprocess_image(image):
    processed_image = np.array(image.resize((256, 256)))  # Resize to model input size
    processed_image = processed_image / 255.0  # Normalize pixel values
    return processed_image

# Function to make glaucoma prediction
def predict_glaucoma(image, classifier):
    image = np.expand_dims(image, axis=0)
    prediction = classifier.predict(image)
    if prediction[0][0] > prediction[0][1]:
        return "Glaucoma"
    else:
        return "Normal"

# Define the background image URL
background_image_url = "https://cdn-prod.medicalnewstoday.com/content/images/articles/326/326753/an-ophthalmologist-doing-eye-surgery.jpg"

# Load pretrained model
classifier = load_model("combinee_cnn.h5")

# Set background image using HTML
background_image_style = f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100vh;  /* Adjust the height as needed */
        width: 100vw;   /* Adjust the width as needed */
    }}
    </style>
"""

# Display background image using HTML
st.markdown(background_image_style, unsafe_allow_html=True)

# Set title color using HTML
st.markdown("<h1 style='color: black;'>Glaucoma Detection App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize empty DataFrame for results
all_results = pd.DataFrame(columns=["Image", "Prediction"])

# Sidebar
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="file_uploader", help="Upload an image for glaucoma detection (Max size: 200 MB)")

# Main content
if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)  # Change text color to black
    st.markdown("---")

    # Display uploaded image
    original_image = Image.open(uploaded_file)
    st.image(original_image, use_column_width=True)
    st.markdown("<p style='color: black; text-align: center;'>Uploaded Image</p>", unsafe_allow_html=True)

    # Perform glaucoma detection
    with st.spinner("Detecting glaucoma..."):
        processed_image = preprocess_image(original_image)
        prediction = predict_glaucoma(processed_image, classifier)
    if prediction == "Glaucoma":
        st.markdown("<p style='color: black; background-color: lightgreen; padding: 10px;'>Glaucoma detected!</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: black; background-color: lightgreen; padding: 10px;'>No glaucoma detected.</p>", unsafe_allow_html=True)

    # Add new result to DataFrame
    new_result = pd.DataFrame({"Image": ["Uploaded Image"], "Prediction": [prediction]})
    all_results = pd.concat([new_result, all_results], ignore_index=True)

# Display all results in table
if not all_results.empty:
    st.markdown("---")
    st.markdown("<h3 style='color: black;'>Detection Results</h3>", unsafe_allow_html=True)
    table_style = {
        'selector': 'table',
        'props': [
            ('border-collapse', 'collapse'),
            ('border', '2px solid black')  # Adjust the width and style of the border
        ]
    }
    text_color_style = {
        'selector': 'td, th',
        'props': [('color', 'black')]
    }
    all_results_styled = all_results.style.set_table_styles([table_style, text_color_style])
    st.table(all_results_styled)
else:
    st.markdown("<p style='color: black; background-color: lightcoral; padding: 10px;'>No images uploaded yet.</p>", unsafe_allow_html=True)

import subprocess
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Install dependencies from requirements.txt file
subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

# Define the background image URL
background_image_url = "https://img.freepik.com/free-photo/security-access-technologythe-scanner-decodes-retinal-data_587448-5015.jpg"

# File path to the uploaded model file
model_file_path = "https://github.com/elainejpai-14/major-project/raw/main/combinee_cnn.h5"

# Load pretrained model
classifier = load_model(model_file_path)

# Load the model with error handling
try:
    classifier = load_model(model_file_path)
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    
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
    .red-bg {{
        background-color: red;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: white;   /* Text color */
    }}
    .green-bg {{
        background-color: green;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: white;   /* Text color */
    }}
    .yellow-bg {{
        background-color: yellow;
        padding: 10px;  /* Adjust the padding as needed */
        margin: 10px;   /* Adjust the margin as needed */
        color: black;   /* Text color */
    }}
    </style>
"""

# Display background image using HTML
st.markdown(background_image_style, unsafe_allow_html=True)

# Set title in dark mode
st.markdown("<h1 style='text-align: center; color: #ecf0f1;'>GlaucoGuard: Gaining Clarity in Glaucoma diagnosis through Deep Learning</h1>", unsafe_allow_html=True)
st.markdown("---")

# Paragraph with content about uploading fundus images
st.markdown("""<p style='font-size: 20px; text-align: center; background-color: orange; color: black;'>This is a simple image classification web application to predict glaucoma through fundus images of the eye. <strong><em>Please upload fundus images only.</em></strong></p>""", unsafe_allow_html=True)

st.markdown("---")

# Initialize empty DataFrame for results
all_results = pd.DataFrame(columns=["Image", "Prediction"])

# Sidebar for uploading image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="file_uploader", help="Upload an image for glaucoma detection (Max size: 200 MB)")

# Main content area
if uploaded_file is not None:
    # Display uploaded image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded Image", use_column_width=True)

    # Perform glaucoma detection
    with st.spinner("Detecting glaucoma..."):
        processed_image = preprocess_image(original_image)
        prediction = predict_glaucoma(processed_image, classifier)

    # Customize messages based on prediction
    if prediction == "Glaucoma":
        st.markdown("<p class='red-bg'>Your eye is diagnosed with Glaucoma. Please consult an ophthalmologist.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='green-bg'>Your eyes are healthy.</p>", unsafe_allow_html=True)

    # Add new result to DataFrame
    new_result = pd.DataFrame({"Image": [uploaded_file.name], "Prediction": [prediction]})
    all_results = pd.concat([new_result, all_results], ignore_index=True)

# Display all results in table with black background color
if not all_results.empty:
    st.markdown("---")
    st.subheader("Detection Results")
    st.table(all_results.style.applymap(lambda x: 'color: red' if x == 'Glaucoma' else 'color: green', subset=['Prediction']))

    # Pie chart
    st.markdown("### Pie Chart")
    pie_data = all_results['Prediction'].value_counts()
    fig, ax = plt.subplots()
    colors = ['green' if label == 'Normal' else 'red' for label in pie_data.index]
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Bar chart
    st.markdown("### Bar Chart")
    bar_data = all_results['Prediction'].value_counts()
    fig, ax = plt.subplots()
    colors = ['green' if label == 'Normal' else 'red' for label in bar_data.index]
    ax.bar(bar_data.index, bar_data, color=colors)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Option to download prediction report
    st.markdown("---")
    st.subheader("Download Prediction Report")
    csv = all_results.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )
else:
    st.markdown("<p class='yellow-bg'>No images uploaded yet.</p>", unsafe_allow_html=True)

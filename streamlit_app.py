import subprocess

# Install dependencies from requirements.txt file
subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import gdown

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
background_image_url = "https://cdcssl.ibsrv.net/ibimg/smb/654x436_80/webmgr/07/d/l/shutterstock_475175770.jpg.webp?812655164adcac539a96922aa296d8dd"
# Set background color and text color for dark mode
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2c3e50; /* Dark background color */
        color: #ecf0f1; /* Light text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set title in dark mode
st.markdown("<h1 style='text-align: center; color: #ecf0f1;'>GlaucoGuard: Gaining Clarity in Glaucoma diagnosis through Deep Learning</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize empty DataFrame for results
all_results = pd.DataFrame(columns=["Image", "Prediction"])

# Sidebar for uploading image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="file_uploader", help="Upload an image for glaucoma detection (Max size: 200 MB)")

# Load pretrained model from Google Drive
model_file_id = '1lhBtxhP18L-KA7wDh4N72xTHZMLUZT82'
model_url = f'https://drive.google.com/uc?id={model_file_id}'
local_model_path = 'combinee_cnn.h5'
gdown.download(model_url, local_model_path, quiet=False)
classifier = load_model(local_model_path)

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
        st.error("Your eye is diagnosed with Glaucoma. Please consult an ophthalmologist.")
    else:
        st.success("Your eyes are healthy.")

    # Add new result to DataFrame
    new_result = pd.DataFrame({"Image": [uploaded_file.name], "Prediction": [prediction]})
    all_results = pd.concat([new_result, all_results], ignore_index=True)

# Display all results in table
if not all_results.empty:
    st.markdown("---")
    st.subheader("Detection Results")
    st.table(all_results)

    # Pie chart
    st.markdown("### Pie Chart")
    pie_data = all_results['Prediction'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Bar chart
    st.markdown("### Bar Chart")
    bar_data = all_results['Prediction'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(bar_data.index, bar_data)
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
    st.warning("No images uploaded yet.")

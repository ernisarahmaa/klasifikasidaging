import streamlit as st
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import pickle

# GLCM properties
properties = ['contrast', 'energy', 'homogeneity', 'dissimilarity', 'correlation']

# Function to calculate GLCM properties for specified distances and angles
def calc_glcm_all_agls(img, red, green, blue, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    # Menghitung GLCM
    glcm = graycomatrix(img, distances=dists, angles=agls, levels=lvl, symmetric=sym, normed=norm)

    feature = []
    # Menghitung properti GLCM yang diinginkan untuk setiap sudut dan jarak
    glcm_props = [prop for name in props for prop in graycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    # Menambahkan komponen warna RGB ke fitur
    feature.extend([red, green, blue])

    return feature

# Load the trained model and scaler
with open('model_eliminasi_contrast.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_eliminasi_contrast.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app
st.title('Image Classification using GLCM and KNN')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Convert image to numpy array
    image = np.array(image)

    # Resize image
    resized_image = cv.resize(image, (250, 250))

    # Calculate average RGB values
    red, green, blue = np.mean(resized_image, axis=(0, 1))

    # Convert image to grayscale
    img_grayscale = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

    # Display original and grayscale images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Image.', use_column_width=True)
    with col2:
        st.image(img_grayscale, caption='Grayscale Image.', use_column_width=True)

    # Extract GLCM features
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_features = calc_glcm_all_agls(img_grayscale, red, green, blue, props=properties, dists=distances, agls=angles)

    # Display extracted GLCM features for debugging
    st.write('Extracted GLCM Features:', glcm_features)

    # Normalize features using the same scaler as in training
    scaled_features = scaler.transform([glcm_features])

    # Display features after scaling for debugging
    st.write('Features after Scaling:', scaled_features)

    # Predict the class of the image
    prediction = model.predict(scaled_features)
    st.write(f'Predicted Class: {prediction[0]}')

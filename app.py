
import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

st.title("MNIST Digit Classifier 🎨")
st.write("Upload a handwritten digit image (JPG/PNG). It will be processed and classified using a HOG + ML model.")

# Load trained model and scaler
model = pickle.load(open("mnist_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Improved image preprocessing function
def preprocess_image(uploaded_image):
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours and crop to bounding box
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize to 20x20 and pad to 28x28
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    padded = padded / 255.0  # normalize

    # Extract HOG features
    features = hog(padded, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return scaler.transform([features])  # apply saved scaler

# File uploader
uploaded_file = st.file_uploader("Upload a Digit Image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and decode the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess and predict
    features = preprocess_image(image)
    prediction = model.predict(features)[0]

    # Display results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Predicted Digit:** `{prediction}`")


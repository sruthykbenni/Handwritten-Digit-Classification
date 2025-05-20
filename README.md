# Handwritten Digit Classification using HOG and Logistic Regression

## Overview
This project is a Handwritten Digit Recognition System that uses Histogram of Oriented Gradients (HOG) for feature extraction and Logistic Regression for classification. The trained model is deployed using Streamlit, allowing users to upload digit images and get real-time predictions.

## Features
- **Image Preprocessing:** Resizing, grayscale conversion, and normalization.
- **HOG Feature Extraction:** Captures shape and edge features from images.
- **Logistic Regression Classifier:** Fast and interpretable model for digit prediction.
- **Machine Learning with Scikit-learn:** Includes training, testing, and model evaluation.
- **Interactive Web Application:** Built with Streamlit for a user-friendly interface.

## Technologies Used
- **Python**
- **OpenCV**
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit**
- **Pickle (for model serialization)**

## Installation
To run this project locally, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/sruthykbenni/handwritten-digit-classifier.git
cd handwritten-digit-classifier
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Streamlit App**
```bash
streamlit run app.py
```

## Dataset
The project uses the MNIST-like handwritten digits dataset from sklearn.datasets.load_digits, which contains:
- 1797 grayscale images of size 8×8 pixels.
- 10 classes representing digits from 0 to 9.

## Model Training
The model follows these steps:
1. HOG Feature Extraction from flattened grayscale digit images
2. Standardization using StandardScaler to normalize features.
3. Train-Test Split (80% training, 20% testing).
4. Model Selection: Logistic Regression was chosen over Naive Bayes and Decision Tree due to better accuracy and generalization.
5. Model Evaluation: Using accuracy and confusion matrix.
6. Model Persistence: Trained model and scaler are saved using pickle.

## Deployment
The model is deployed as a web application using Streamlit. The app allows users to:
- Upload a digit image (PNG/JPG).
- View the preprocessed image.
- Get a predicted digit instantly.

You can access the live demo here:  
[🔗 Live App](https://handwritten-digit-classification-ufnv8ksaytwke5a8b6g73o.streamlit.app/)

## Usage
1. Open the web app.
2. Upload a digit image (clear, centered image works best).
3. View the processed image and prediction result.

## Future Improvements
- Improve image preprocessing for noisy or off-centered digits.
- Add support for larger and varied digit datasets like full MNIST (28x28).
- Explore deep learning models like CNNs for higher accuracy.
- Allow batch image uploads and prediction history.

## Contributing
Feel free to contribute! Fork the repository, make your changes, and submit a pull request. Let's make digit recognition more accessible together.

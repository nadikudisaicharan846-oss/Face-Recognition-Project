import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load saved model
with open("saved_model.pkl", "rb") as f:
    model_data = pickle.load(f)

ann = model_data["ann"]
pca = model_data["pca"]
names = model_data["names"]

st.title("AI Face Recognition System")

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    # Resize image
    image = image.resize((100, 100))

    # Show image
    st.image(image, caption="Uploaded Image")

    # Convert image to numpy array
    img_array = np.array(image)

    # Flatten image
    img_flat = img_array.flatten().reshape(1, -1)

    # PCA transform
    img_pca = pca.transform(img_flat)

    # Predict
    prediction = ann.predict(img_pca)[0]

    # Confidence
    confidence = np.max(ann.predict_proba(img_pca)) * 100

    # Show results
    st.success(f"Predicted Person: {names[prediction]}")
    st.info(f"Confidence: {confidence:.2f}%")
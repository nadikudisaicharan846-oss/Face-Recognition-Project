import streamlit as st

st.title("Face Recognition Project")

st.write("My Streamlit app is working!")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file)
    st.success("Image Uploaded Successfully!")
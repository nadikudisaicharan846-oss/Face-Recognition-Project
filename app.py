import streamlit as st
import pickle

with open("saved_model.pkl", "rb") as f:
    model_data = pickle.load(f)

st.write(model_data.keys())
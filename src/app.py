from src.pipeline.utils import load_pkl
from src.pipeline.utils import preprocess
import streamlit as st
import pickle
import nltk
import numpy as np

# File paths
vectorizer_file_path = 'models/vectorizer.pkl'
model_file_path = 'models/model.pkl'

# Load pickle
tfidf = load_pkl(vectorizer_file_path)
model = load_pkl(model_file_path)

# Streamlit
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message...")

if st.button('Predict'):

    #1. Preprocess
    preprocessed_sms = preprocess(input_sms)

    #2. Vectorize
    vector_input = tfidf.transform([preprocessed_sms]).toarray()

    #3. Predict
    result = model.predict(vector_input)[0]

    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
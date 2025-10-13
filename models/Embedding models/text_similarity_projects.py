# Install dependencies if not already installed
# pip install streamlit langchain_google_genai numpy python-dotenv

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

# Load environment variables (like API keys)
load_dotenv()

# Initialize Gemini embeddings model
model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Streamlit frontend
st.title("Text Similarity Calculator using Gemini Embeddings")

# User input
text1 = st.text_area("Enter Text 1:", "")
text2 = st.text_area("Enter Text 2:", "")

# Calculate similarity when button is pressed
if st.button("Calculate Similarity"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts!")
    else:
        # Generate embeddings (wrap texts in a list)
        vector1 = model.embed_documents([text1])[0]
        vector2 = model.embed_documents([text2])[0]

        # Calculate similarity
        similarity_score = cosine_similarity(vector1, vector2)
        similarity_percentage = similarity_score * 100

        # Display results
        st.success(f"Cosine Similarity Score: {similarity_score:.4f}")
        st.info(f"Similarity Percentage: {similarity_percentage:.2f}%")

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
st.title("mychatbot")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
user_input = st.text_input("You: ")
if user_input:
    response = model.invoke(user_input)
    st.write("ishowr gu: ", response.content)
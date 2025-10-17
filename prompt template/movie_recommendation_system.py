import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

st.title("Movie Recommendation System")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt = PromptTemplate(
    template="You are a movie recommendation system. Based on the user's preferences, recommend three movies along with a brief description for each:\n\n{user_preferences}",
    input_variables=["user_preferences"]
)

user_preferences = st.text_area("Enter your movie preferences:", height=150)

if st.button("Recommend Movies"):
    user_input = prompt.format(user_preferences=user_preferences)
    response = model(user_input)  # Use callable, not invoke()
    st.write("Recommended Movies:", response)

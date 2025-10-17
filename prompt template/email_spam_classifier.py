from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    temperature=0.7)
st.title("Email Spam Classifier")
email_content = st.text_area("Enter the email content:")
prompt = PromptTemplate(
    template="You are an email spam classifier. Classify the following email content as 'Spam' or 'Not Spam':\n\n{email_content}",

)
if st.button("Classify"):
    user_input = prompt.format(email_content=email_content)
    response = model.invoke(user_input)
    st.write("this email is : ", response.content)

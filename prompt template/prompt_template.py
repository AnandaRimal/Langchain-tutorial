from langchain import PromptTemplate
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
load_dotenv()
st.title("Research tool")
paper=st.selectbox("Select a model", ["attentuation all you need ", "lstm", "transformer", "bert", "gpt", "gemini"])
language=st.selectbox("Select a language", ["en", "nepali", "hindi", "french", "german"])
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
format=st.selectbox("Select a task", ["summarization ", "only key points", "fact"])

prompt=PromptTemplate(
    template="You are a research assistant. You will be provided with a research paper title. Your task is to generate a {format} of the research paper titled '{paper}' in {language}.",
    input_variables=["paper", "language", "format"]
)
user_input = prompt.format(paper=paper, language=language, format=format)
if st.button("Generate"):
    response = model.invoke(user_input)
    st.write("Research Assistant: ", response.content)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",task_type="RETRIEVAL_DOCUMENT")
vector = model.embed_query("What is the capital of India?")
print(vector)

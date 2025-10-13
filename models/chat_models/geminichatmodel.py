from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    temperature=0.7)
response=model.invoke("what is the name of india")
print(response.content)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)
response = model.invoke("what is the name of india")
print(response.c)
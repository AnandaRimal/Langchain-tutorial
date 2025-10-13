from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
response=model.invoke("what is another name of dolakha")
print(response)

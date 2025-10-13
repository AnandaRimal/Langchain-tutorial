from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-7-sonnet-20250219", model_provider="anthropic")

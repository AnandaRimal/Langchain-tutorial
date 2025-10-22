from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
load_dotenv()
chat_prompt=ChatPromptTemplate([
    ("system","you are a helpful assistant explain about {topic} in simple terms"),
    ( "human","explain in simple term what is {topic}?")
])
final_prompt=chat_prompt.format_messages(topic="gemini 2.5")
print(final_prompt)
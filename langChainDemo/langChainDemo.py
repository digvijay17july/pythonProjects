
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(api_key="****_YOUR_API_KEY_****")

# Generate text
response = llm.invoke("Once upon a time")
print(response)

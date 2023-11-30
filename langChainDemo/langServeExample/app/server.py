from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

model = ChatOpenAI(openai_api_key="_YOUR_OPEN_API_KEY_")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# Edit this to add the chain you want to add
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

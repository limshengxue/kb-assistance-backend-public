from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import Chatbot
from retriever import Retriever
from contextlib import asynccontextmanager

from utils import get_file_name_from_source_nodes


class ChatInput(BaseModel):
    chat: str
    session_id : Optional[str] = "TEST-001"

@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever()
    await retriever.initialize()
    app.state.chatbot = Chatbot(retriever)
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/chat")
def chat(chat_input: ChatInput):
    chatbot = app.state.chatbot
    output = chatbot.chat(chat_input.chat, chat_input.session_id)

    relevant_files = get_file_name_from_source_nodes(output.source_nodes)

    return {"response": output.response, "relevant_files": relevant_files}

@app.get("/get_chat_history")
def get_chat_history(session_id: str = "TEST-001"):
    chatbot = app.state.chatbot
    return chatbot.get_chat_history(session_id)

@app.get("/clear_memory")
def clear_memory(session_id: str = "TEST-001"):
    chatbot = app.state.chatbot
    chatbot.clear_memory(session_id)

    return "Memory cleared"

    
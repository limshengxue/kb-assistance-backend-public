from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import CondensePlusContextChatEngine, ContextChatEngine
from retriever import Retriever
from llama_index.storage.chat_store.postgres import PostgresChatStore
from llama_index.core.memory import ChatMemoryBuffer
import os
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

class Chatbot:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever.get_retriever()
        self.llm = OpenAI(model="gpt-4o")

    def _build_chat_engine(self, session_id: str):
        chat_store = PostgresChatStore.from_uri(
            uri=os.environ.get("DATABASE_CONN"),
        )

        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=session_id,
        ) 

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            context_prompt=(
                "You are a helpful assistant that can answer questions about CelcomDigi services. "
                "You have access to a knowledge base of CelcomDigi documents. "
                "If you don't know the answer, just say that you don't know. Do not hallucinate."
                "Generate a response based on the input from the user."
                "Here are the relevant documents for the context:\n"
                "{context_str}"
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            ),
            verbose=False,
            memory=chat_memory,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
        )
        return chat_engine
         

    def chat(self, user_input: str, session_id : str = "TEST-001"):
        chat_engine = self._build_chat_engine(session_id)
        output = chat_engine.chat(user_input)

        return output
    
    def clear_memory(self, session_id : str = "TEST-001"):
        chat_store = PostgresChatStore.from_uri(
            uri=os.environ.get("DATABASE_CONN"),
        )

        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=session_id,
        ) 
        chat_memory.reset()
    
    def get_chat_history(self, session_id: str = "TEST-001"):
        chat_store = PostgresChatStore.from_uri(
            uri=os.environ.get("DATABASE_CONN"),
        )

        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=session_id,
        ) 
        return chat_memory.get_all()
    def get_chat_engine(self):
        chat_engine = ContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            context_prompt=(
                "You are a helpful assistant that can answer questions about CelcomDigi services. "
                "You have access to a knowledge base of CelcomDigi documents. "
                "If you don't know the answer, just say that you don't know. Do not hallucinate."
                "Generate a response based on the input from the user."
                "Here are the relevant documents for the context:\n"
                "{context_str}"
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            ),
            verbose=False,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
        )
        return chat_engine



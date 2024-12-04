from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
import os

def get_ollama(model: str, base_url: str = "http://127.0.0.1:11434") -> ChatOllama:
    """
    Returns a ChatOllama object with the given model and base_url.
    """
    return ChatOllama(
        model=model,
        temperature=0,
        base_url=base_url,
    )

def get_ollama_embedding(model: str, base_url: str = "http://127.0.0.1:11434") -> OllamaEmbeddings:
    """ 
    Returns an OllamaEmbeddings object with the given model and base_url.
    """
    return OllamaEmbeddings(
        model="all-minilm:l6-v2",
        base_url=os.getenv("BASE_URL"),
    )
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL = os.getenv("LLM_MODEL", "codellama:7b")
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))
    FAISS_INDEX_PATH = "./RAG/data/vector_db_ollama"
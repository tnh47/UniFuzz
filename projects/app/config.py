import os
from pathlib import Path

class Config:
    # Path
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data"))) # Configurable data dir
    RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
    PROCESSED_DIR = DATA_DIR / "processed"
    VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", str(Path(__file__).parent.parent / "vector_db"))) # Configurable vector db dir
    FUZZING_WORKSPACE = Path(os.getenv("FUZZING_WORKSPACE", str(Path(__file__).parent.parent / "fuzz_tests"))) # Configurable fuzz workspace

    # PDF
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) # Configurable chunk size
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200)) # Configurable chunk overlap

    # API - Google AI Studio (Gemini API)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # MUST be set in environment
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001") # Model for embeddings
    GEMINI_EMBEDDING_ENDPOINT = os.getenv("GEMINI_EMBEDDING_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/") # Base endpoint for embedding API
    GEMINI_EMBEDDING_DIMENSION = 768 # Dimension for 'models/embedding-001'

    GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "models/gemini-pro") # Model for text generation (chat)
    GEMINI_GENERATE_CONTENT_ENDPOINT = os.getenv("GEMINI_GENERATE_CONTENT_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/") # Base endpoint for text generation API


    # Fuzzing
    DEFAULT_FUZZING_TOOL = os.getenv("DEFAULT_FUZZING_TOOL", "echidna") # Configurable default fuzz tool
    FUZZING_TRIGGER_KEYWORDS = [ # Configurable fuzz trigger keywords
        "lỗ hổng", "vulnerability", "kiểm thử", "fuzz", "bug", "exploit", "attack", "reentrancy", "overflow", "underflow"
    ]
    LLM_FUZZING_TRIGGER_ENABLED = os.getenv("LLM_FUZZING_TRIGGER_ENABLED", "false").lower() == "true" # Enable/disable LLM fuzz trigger

    @classmethod
    def setup_directories(cls):
        cls.RAW_PDF_DIR.mkdir(exist_ok=True, parents=True)
        cls.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)
        cls.FUZZING_WORKSPACE.mkdir(exist_ok=True, parents=True)

    @classmethod
    def get_gemini_api_key(cls):
        """Get Gemini API key from environment, raise error if not set"""
        key = cls.GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        return key
import os
from dataclasses import dataclass
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    reranker: str = os.getenv("RERANKER", "crossencoder")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    use_multi_query: bool = os.getenv("USE_MULTI_QUERY", "false").lower() == "true"
    memory_turns: int = int(os.getenv("MEMORY_TURNS", "3"))

settings = Settings()

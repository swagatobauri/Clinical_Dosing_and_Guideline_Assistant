import os
from dataclasses import dataclass
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    reranker: str = os.getenv("RERANKER", "crossencoder")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    top_k: int = int(os.getenv("TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

settings = Settings()

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import faiss
from typing import List, Dict, Any, TypedDict
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings

class ScoredChunk(TypedDict):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class DenseRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        # Uses local HuggingFace embeddings (100% Free, runs offline)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.faiss_index = None
        self.chunks = []

    def index(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        texts = [c.get("text", "") for c in chunks]
        
        # Generate embeddings
        embs = self.embeddings.embed_documents(texts)
        embs_array = np.array(embs).astype("float32")
        
        # Dynamically initialize FAISS index based on embedding dimension
        if self.faiss_index is None:
            dim = embs_array.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            
        self.faiss_index.add(embs_array)

    def search(self, query: str, k: int = 10) -> List[ScoredChunk]:
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        query_emb = self.embeddings.embed_query(query)
        query_emb_array = np.array([query_emb]).astype("float32")
        
        distances, indices = self.faiss_index.search(query_emb_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                # FAISS L2 distance: lower is better. We invert it so higher is better for RRF.
                chunk["score"] = 1.0 / (1.0 + float(distances[0][i]))
                results.append(chunk)
                
        return results

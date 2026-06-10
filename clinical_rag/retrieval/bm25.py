from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from retrieval.dense import ScoredChunk

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        tokenized_corpus = [c.get("text", "").lower().split() for c in chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 10) -> List[ScoredChunk]:
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        scored_indices = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:k]
        
        scored_chunks = []
        for idx, score in scored_indices:
            chunk = self.chunks[idx]
            scored_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })
            
        return scored_chunks

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from retrieval.dense import ScoredChunk
from retrieval.config import settings

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[ScoredChunk], top_n: int = 5) -> List[ScoredChunk]:
        pass

class CohereReranker(BaseReranker):
    def __init__(self):
        # pyrefly: ignore [missing-import]
        import cohere
        # Picks up COHERE_API_KEY from environment
        self.client = cohere.Client()
        self.model = "rerank-english-v3.0"

    def rerank(self, query: str, chunks: List[ScoredChunk], top_n: int = 5) -> List[ScoredChunk]:
        if not chunks:
            return []
            
        docs = [c.get("text", "") for c in chunks]
        results = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=top_n
        )
        
        reranked_chunks = []
        for result in results.results:
            idx = result.index
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked_chunks.append(chunk)
            
        return reranked_chunks

class CrossEncoderReranker(BaseReranker):
    def __init__(self):
        # pyrefly: ignore [missing-import]
        from sentence_transformers import CrossEncoder
        # Automatically downloads model weights if not cached
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, chunks: List[ScoredChunk], top_n: int = 5) -> List[ScoredChunk]:
        if not chunks:
            return []
            
        pairs = [[query, c.get("text", "")] for c in chunks]
        scores = self.model.predict(pairs)
        
        # Zip chunks with their new scores
        scored_pairs = list(zip(chunks, scores))
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks = []
        for chunk, score in scored_pairs[:top_n]:
            new_chunk = chunk.copy()
            # Convert float32 to python float for JSON serialization safety
            new_chunk["rerank_score"] = float(score)
            reranked_chunks.append(new_chunk)
            
        return reranked_chunks

def get_reranker() -> BaseReranker:
    """Factory function to get the appropriate reranker based on config settings."""
    reranker_type = settings.reranker.lower()
    
    if reranker_type == "cohere":
        return CohereReranker()
    elif reranker_type == "crossencoder":
        return CrossEncoderReranker()
    else:
        print(f"Warning: Unknown reranker '{reranker_type}', falling back to crossencoder.")
        return CrossEncoderReranker()

if __name__ == "__main__":
    print("Testing get_reranker() factory...")
    
    # Test CrossEncoder (default) since it requires no API key
    os.environ["RERANKER"] = "crossencoder"
    import importlib
    import retrieval.config
    importlib.reload(retrieval.config)
    from retrieval.config import settings
    
    print(f"Current setting: {settings.reranker}")
    reranker = get_reranker()
    print(f"Loaded reranker: {type(reranker).__name__}")
    
    # Test reranking
    sample_chunks = [
        {"chunk_id": "1", "text": "The patient has acute bronchitis. Prescribe antibiotics."},
        {"chunk_id": "2", "text": "Metformin is used as a first-line treatment for type 2 diabetes."},
        {"chunk_id": "3", "text": "Lisinopril is an ACE inhibitor used commonly for hypertension and heart failure."}
    ]
    query = "What is a treatment for high blood pressure?"
    
    print("\nReranking with CrossEncoder (this may take a moment to download the model the first time)...")
    results = reranker.rerank(query, sample_chunks, top_n=2)
    
    print("\nRanked Results:")
    for i, res in enumerate(results):
        print(f"{i+1}. [Score: {res.get('rerank_score', 0):.4f}] {res['text']}")
        
    print("\nSmoke test passed!")

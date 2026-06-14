from typing import List
from retrieval.dense import DenseRetriever, ScoredChunk
from retrieval.bm25 import BM25Retriever
from retrieval.config import settings

class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, reranker=None):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker

    def index(self, chunks: list[dict]):
        self.dense.index(chunks)
        self.bm25.index(chunks)

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        # Step 1: Get top 20 from both
        dense_results = self.dense.search(query, k=20)
        bm25_results = self.bm25.search(query, k=20)
        
        # Step 2: Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, doc in enumerate(dense_results, start=1):
            chunk_id = doc["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": doc, "score": 0.0}
            rrf_scores[chunk_id]["score"] += 1.0 / (settings.rrf_k + rank)
            
        for rank, doc in enumerate(bm25_results, start=1):
            chunk_id = doc["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": doc, "score": 0.0}
            rrf_scores[chunk_id]["score"] += 1.0 / (settings.rrf_k + rank)
            
        # Sort by RRF score
        fused = list(rrf_scores.values())
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        # Extract the chunks and update their scores to the RRF score
        fused_chunks = []
        for item in fused:
            chunk = item["chunk"].copy()
            chunk["score"] = item["score"]
            fused_chunks.append(chunk)
            
        # Top 20 RRF results
        top_candidates = fused_chunks[:20]
        
        # Step 3 & 4: Rerank if available
        if self.reranker:
            top_candidates = self.reranker.rerank(query, top_candidates, top_n=k)
            return top_candidates
            
        return top_candidates[:k]

if __name__ == "__main__":
    # Smoke test
    sample_chunks = [
        {"chunk_id": "c1", "text": "Metformin is a common treatment for type 2 diabetes.", "metadata": {}},
        {"chunk_id": "c2", "text": "For a patient with eGFR 45, the maximum metformin dose is 1000mg/day.", "metadata": {}},
        {"chunk_id": "c3", "text": "High blood pressure can be treated with ACE inhibitors.", "metadata": {}},
        {"chunk_id": "c4", "text": "Lisinopril is an ACE inhibitor used for hypertension.", "metadata": {}},
        {"chunk_id": "c5", "text": "Metformin should be immediately discontinued if eGFR falls below 30.", "metadata": {}},
    ]
    
    print("Initializing Retrievers (Using Free Local Embeddings)...")
    try:
        dense_retriever = DenseRetriever()
        bm25_retriever = BM25Retriever()
        hybrid_retriever = HybridRetriever(dense=dense_retriever, bm25=bm25_retriever)
        
        print("Indexing 5 sample chunks...")
        hybrid_retriever.index(sample_chunks)
        
        query = "metformin dose eGFR 45"
        print(f"\nSearching for: '{query}'")
        
        results = hybrid_retriever.search(query, k=3)
        
        print("\nRanked Results:")
        for i, res in enumerate(results, 1):
            print(f"{i}. [Score: {res['score']:.4f}] Chunk {res['chunk_id']}: {res['text']}")
            
        print("\nSmoke test passed!")
    except Exception as e:
        print(f"Smoke test failed: {e}")

from typing import List, Dict, Any
from retrieval.dense import DenseRetriever, ScoredChunk
from retrieval.bm25 import BM25Retriever

class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, reranker=None):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker

    def index(self, chunks: List[Dict[str, Any]]):
        self.dense.index(chunks)
        self.bm25.index(chunks)

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        # Step 1: dense.search(query, k=20) + bm25.search(query, k=20)
        dense_results = self.dense.search(query, k=20)
        bm25_results = self.bm25.search(query, k=20)
        
        # Step 2: Reciprocal Rank Fusion: score = sum(1 / (60 + rank_i)) for each result
        rrf_scores = {}
        chunk_map = {}
        
        for rank, res in enumerate(dense_results, start=1):
            cid = res["chunk_id"]
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60.0 + rank)
            
        for rank, res in enumerate(bm25_results, start=1):
            cid = res["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60.0 + rank)
            
        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_results = []
        for cid, score in sorted_chunks[:20]:
            chunk = chunk_map[cid].copy()
            chunk["score"] = score
            top_results.append(chunk)
            
        # Step 3: If reranker is set, pass top-20 RRF results through reranker, return top-k
        if self.reranker:
            top_results = self.reranker.rerank(query, top_results, top_n=k)
        else:
            top_results = top_results[:k]
            
        # Step 4: Return top-k by final score
        return top_results

if __name__ == "__main__":
    import os
    
    sample_chunks = [
        {
            "chunk_id": "c1", 
            "text": "Metformin is a first-line treatment for type 2 diabetes.", 
            "metadata": {"source": "guideline"}
        },
        {
            "chunk_id": "c2", 
            "text": "For a patient with eGFR 45, the maximum metformin dose is 1000mg/day.", 
            "metadata": {"source": "guideline"}
        },
        {
            "chunk_id": "c3", 
            "text": "Apixaban dose should be adjusted for patients over 80 years old.", 
            "metadata": {"source": "guideline"}
        },
        {
            "chunk_id": "c4", 
            "text": "Lisinopril is an ACE inhibitor used for hypertension.", 
            "metadata": {"source": "guideline"}
        },
        {
            "chunk_id": "c5", 
            "text": "Metformin should be immediately discontinued if eGFR falls below 30.", 
            "metadata": {"source": "guideline"}
        }
    ]
    
    print("Initializing Retrievers...")
    try:
        # Fallback to FakeEmbeddings if API key is missing to ensure smoke test prints results
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not found. Using FakeEmbeddings for smoke test.")
            # pyrefly: ignore [missing-import]
            from langchain_community.embeddings import FakeEmbeddings
            import retrieval.dense
            retrieval.dense.OpenAIEmbeddings = lambda model=None: FakeEmbeddings(size=1536)

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

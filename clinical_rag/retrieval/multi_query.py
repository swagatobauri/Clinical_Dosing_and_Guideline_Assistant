from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
from retrieval.dense import DenseRetriever, ScoredChunk
from retrieval.bm25 import BM25Retriever
from retrieval.config import settings

class MultiQueryRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, llm, reranker=None):
        self.dense = dense
        self.bm25 = bm25
        self.llm = llm
        self.reranker = reranker
        self.prompt = PromptTemplate.from_template(
            "You are a clinical assistant. Generate 3 different versions of the following clinical query "
            "to retrieve relevant guidelines from a database. Focus on medical synonyms, generic/brand names, "
            "and different ways to phrase the condition or drug.\n"
            "Provide the 3 alternative questions separated by newlines, with no extra text.\n"
            "Original query: {query}\n"
            "Output (3 questions ONLY):"
        )
        
    def index(self, chunks: list[dict]):
        self.dense.index(chunks)
        self.bm25.index(chunks)

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        queries = [query]
        
        # 1. Generate Query Variants
        if self.llm:
            try:
                prompt_val = self.prompt.format(query=query)
                response = self.llm.invoke(prompt_val)
                # Split by newlines, remove numbers/bullets if any
                variants = [q.strip(" -1234567890.") for q in response.content.split('\n') if q.strip()]
                queries.extend(variants[:3])
            except Exception as e:
                print(f"Failed to generate query variants: {e}")
                
        # 2. Retrieve for all queries (both dense and sparse)
        rrf_scores = {}
        
        for q in queries:
            dense_results = self.dense.search(q, k=20)
            bm25_results = self.bm25.search(q, k=20)
            
            # Apply RRF for this specific query's results
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
                
        # 3. Sort by aggregated RRF score (RAG Fusion)
        fused = list(rrf_scores.values())
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        fused_chunks = []
        for item in fused:
            chunk = item["chunk"].copy()
            chunk["score"] = item["score"]
            fused_chunks.append(chunk)
            
        top_candidates = fused_chunks[:20]
        
        # 4. Rerank if available using the original query
        if self.reranker:
            top_candidates = self.reranker.rerank(query, top_candidates, top_n=k)
            return top_candidates
            
        return top_candidates[:k]

if __name__ == "__main__":
    from retrieval.dense import DenseRetriever
    from retrieval.bm25 import BM25Retriever
    # pyrefly: ignore [missing-import]
    from langchain_groq import ChatGroq
    import os
    
    # Smoke test
    sample_chunks = [
        {"chunk_id": "c1", "text": "Metformin is a common treatment for type 2 diabetes.", "metadata": {}},
        {"chunk_id": "c2", "text": "For a patient with eGFR 45, the maximum metformin dose is 1000mg/day.", "metadata": {}},
        {"chunk_id": "c3", "text": "High blood pressure can be treated with ACE inhibitors.", "metadata": {}},
        {"chunk_id": "c4", "text": "Lisinopril is an ACE inhibitor used for hypertension.", "metadata": {}},
        {"chunk_id": "c5", "text": "Glucophage should be immediately discontinued if eGFR falls below 30.", "metadata": {}},
    ]
    
    print("Initializing Retrievers...")
    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()
    llm = ChatGroq(model=settings.llm_model, temperature=0) if os.getenv("GROQ_API_KEY") else None
    
    mq_retriever = MultiQueryRetriever(dense=dense_retriever, bm25=bm25_retriever, llm=llm)
    mq_retriever.index(sample_chunks)
    
    query = "Max dose of Glucophage with low kidney function"
    print(f"\nSearching for: '{query}'")
    
    results = mq_retriever.search(query, k=3)
    
    print("\nRanked Results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. [Score: {res.get('score', 0):.4f}] Chunk {res['chunk_id']}: {res['text']}")
        
    print("\nSmoke test passed!")

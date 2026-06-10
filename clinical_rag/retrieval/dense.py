from typing import TypedDict, List, Dict, Any
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import OpenAIEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

class ScoredChunk(TypedDict):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class DenseRetriever:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.chunk_map = {}

    def index(self, chunks: List[Dict[str, Any]]):
        docs = []
        for c in chunks:
            chunk_id = c.get("chunk_id", "")
            self.chunk_map[chunk_id] = c
            
            metadata = c.get("metadata", {}).copy()
            metadata["chunk_id"] = chunk_id
            
            docs.append(Document(page_content=c.get("text", ""), metadata=metadata))
        
        if docs:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def search(self, query: str, k: int = 10) -> List[ScoredChunk]:
        if not self.vectorstore:
            return []
            
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        scored_chunks = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id", "")
            
            scored_chunks.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })
            
        return scored_chunks

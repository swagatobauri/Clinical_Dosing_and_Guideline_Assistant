import os
from typing import List, Dict, Any

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate

# We need the config and retrievers from our retrieval module
from retrieval.config import settings
from retrieval.dense import DenseRetriever, ScoredChunk
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.multi_query import MultiQueryRetriever
from retrieval.reranker import get_reranker
from app.memory import ConversationMemory
from ingestion.chunker import (
    fixed_size_chunker, 
    recursive_chunker, 
    semantic_chunker, 
    hierarchical_chunker
)

class ClinicalRAGChain:
    def __init__(self, strategy="hierarchical"):
        self.strategy = strategy
        self.memory = ConversationMemory(max_turns=settings.memory_turns)
        
        # Build the retrievers
        # Uses local HuggingFace embeddings by default (100% Free)
        self.dense_retriever = DenseRetriever(embedding_model=settings.embedding_model)
        self.bm25_retriever = BM25Retriever()
        self.reranker = get_reranker()
        
        # Setup the LLM
        try:
            self.llm = ChatGroq(model=settings.llm_model, temperature=0)
        except Exception as e:
            print(f"Warning: ChatGroq initialization failed (missing GROQ_API_KEY?): {e}")
            self.llm = None

        if settings.use_multi_query:
            print("INFO: Using MultiQueryRetriever (RAG Fusion)")
            self.retriever = MultiQueryRetriever(
                dense=self.dense_retriever, 
                bm25=self.bm25_retriever, 
                llm=self.llm,
                reranker=self.reranker
            )
        else:
            self.retriever = HybridRetriever(
                dense=self.dense_retriever, 
                bm25=self.bm25_retriever, 
                reranker=self.reranker
            )
        
        # Define the system prompt with conversation history
        self.prompt = PromptTemplate.from_template(
            "You are a clinical dosing assistant. Use ONLY the provided context and conversation history to answer.\n\n"
            "Conversation History:\n{history}\n\n"
            "Context Documents:\n{context}\n\n"
            "Question: {question}\n\n"
            "If the answer is not in the context, say 'I cannot find the answer in the provided clinical guidelines.' "
            "Do not guess. Answer:\n"
        )

    def index(self, documents_or_chunks: List[Dict[str, Any]]):
        """Indexes the chunks or raw sections using the specified chunking strategy."""
        chunks = documents_or_chunks
        
        # If they don't have chunk_id, they are raw sections
        if chunks and "chunk_id" not in chunks[0]:
            strategies = {
                "fixed": lambda s: fixed_size_chunker(s, chunk_size=512, overlap=50),
                "recursive": recursive_chunker,
                "semantic": semantic_chunker,
                "hierarchical": hierarchical_chunker
            }
            func = strategies.get(self.strategy, hierarchical_chunker)
            chunks = func(chunks)
            
        self.retriever.index(chunks)

    def answer(self, query: str) -> Dict[str, Any]:
        """Runs the full pipeline to answer a query."""
        
        # 1. Retrieve & Rerank
        top_chunks = self.retriever.search(query, k=settings.top_k)
        
        # 2. Format Context and History
        context_blocks = []
        sources = []
        
        top_score = 0.0
        if top_chunks:
            top_score = top_chunks[0].get("rerank_score", 0.0)
            
        for i, chunk in enumerate(top_chunks, start=1):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            doc = meta.get("source", "Unknown Document")
            sec = meta.get("section", "Unknown Section")
            score = chunk.get("rerank_score", 0.0)
            
            # Format for LLM
            block = f"--- Document {i} ---\nSource: {doc}\nSection: {sec}\nText: {text}"
            context_blocks.append(block)
            
            # Track sources for output
            sources.append({
                "text": text,
                "document": doc,
                "section": sec,
                "rerank_score": score
            })
            
        context_str = "\n\n".join(context_blocks)
        history_str = self.memory.get_context_string()
        
        # 3. Generate Answer
        answer_text = ""
        if not self.llm:
            answer_text = "[LLM ERROR: ChatGroq not initialized. Please set GROQ_API_KEY.]"
        else:
            try:
                prompt_val = self.prompt.format(
                    context=context_str,
                    history=history_str,
                    question=query
                )
                response = self.llm.invoke(prompt_val)
                answer_text = response.content
            except Exception as e:
                answer_text = f"Error generating answer: {e}"
            
        # 4. Save to memory
        self.memory.add_turn(query=query, answer=answer_text)
            
        # 4. Compute Confidence
        confidence = "high"
        
        if top_score < 0.6 or "INSUFFICIENT INFORMATION" in answer_text.upper():
            confidence = "low"
        elif top_score < 0.8:
            confidence = "medium"
            
        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence
        }

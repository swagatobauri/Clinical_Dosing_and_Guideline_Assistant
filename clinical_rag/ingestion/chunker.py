import os
import uuid
import re
import nltk
from typing import List, Dict, Any

# Ensure NLTK punkt data is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # Use quiet=True to avoid printing download messages in stdout
    nltk.download('punkt_tab', quiet=True)

# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import OpenAIEmbeddings

def generate_chunk_id() -> str:
    return str(uuid.uuid4())

def fixed_size_chunker(sections: List[Dict[str, Any]], chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Naive fixed-size split ignoring structure.
    This is the BASELINE — it is intentionally bad for clinical use.
    """
    chunks = []
    for section in sections:
        text = section.get("text", "")
        metadata = {
            "source": section.get("source"),
            "section": section.get("section"),
            "page": section.get("page")
        }
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "chunk_id": generate_chunk_id(),
                "text": chunk_text,
                "strategy": "fixed_size",
                "metadata": metadata.copy()
            })
            start += chunk_size - overlap
            if start >= len(text):
                break
    return chunks

def recursive_chunker(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use LangChain RecursiveCharacterTextSplitter.
    Respects paragraphs and sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for section in sections:
        text = section.get("text", "")
        metadata = {
            "source": section.get("source"),
            "section": section.get("section"),
            "page": section.get("page")
        }
        
        split_texts = splitter.split_text(text)
        for chunk_text in split_texts:
            chunks.append({
                "chunk_id": generate_chunk_id(),
                "text": chunk_text,
                "strategy": "recursive",
                "metadata": metadata.copy()
            })
    return chunks

def semantic_chunker(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use langchain_experimental SemanticChunker with OpenAI embeddings.
    Groups sentences by semantic similarity before splitting.
    """
    # Note: Requires OPENAI_API_KEY to be set in the environment
    embeddings = OpenAIEmbeddings()
    splitter = SemanticChunker(embeddings)
    
    chunks = []
    for section in sections:
        text = section.get("text", "")
        metadata = {
            "source": section.get("source"),
            "section": section.get("section"),
            "page": section.get("page")
        }
        
        try:
            split_texts = splitter.split_text(text)
            for chunk_text in split_texts:
                chunks.append({
                    "chunk_id": generate_chunk_id(),
                    "text": chunk_text,
                    "strategy": "semantic",
                    "metadata": metadata.copy()
                })
        except Exception as e:
            raise e
            
    return chunks

def hierarchical_chunker(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parent-document strategy:
    Parent chunks = full section (h2 level)
    Child chunks = individual sentences within that section
    Each child chunk stores parent_id pointing to its parent chunk
    
    THIS IS THE KEY STRATEGY — keeps dosing rules attached to their qualifying conditions.
    """
    chunks = []
    for section in sections:
        text = section.get("text", "")
        metadata = {
            "source": section.get("source"),
            "section": section.get("section"),
            "page": section.get("page")
        }
        
        # Create parent chunk
        parent_id = generate_chunk_id()
        chunks.append({
            "chunk_id": parent_id,
            "text": text,
            "strategy": "hierarchical_parent",
            "metadata": metadata.copy()
        })
        
        # Create child chunks
        # Use NLTK sentence tokenizer for clinical-aware splitting
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            child_metadata = metadata.copy()
            child_metadata["parent_id"] = parent_id
            
            chunks.append({
                "chunk_id": generate_chunk_id(),
                "text": sentence,
                "strategy": "hierarchical_child",
                "metadata": child_metadata
            })
            
    return chunks

if __name__ == "__main__":
    sample_text = (
        "Patient presents with acute myocardial infarction. "
        "Aspirin 324mg should be administered immediately. "
        "Monitor for signs of bleeding. "
        "\n\n"
        "If patient is allergic to aspirin, consider clopidogrel as an alternative. "
        "Dr. Smith recommends adjusting the dose for renal impairment, e.g. for eGFR < 30 mL/min/1.73 m2. "
        "Give 500mg b.i.d. for best results."
    )
    
    sample_sections = [
        {
            "text": sample_text,
            "source": "AHA_Guidelines_2023.pdf",
            "section": "Acute Coronary Syndrome",
            "page": 42
        }
    ]
    
    print("========================================")
    print("Testing Chunking Strategies")
    print("========================================\n")
    
    # Using a smaller chunk size here so our short sample text actually gets split
    fixed_chunks = fixed_size_chunker(sample_sections, chunk_size=100, overlap=20)
    print(f"1. Fixed Size Chunks: {len(fixed_chunks)}")
    
    recursive_chunks = recursive_chunker(sample_sections)
    print(f"2. Recursive Chunks: {len(recursive_chunks)}")
    
    hierarchical_chunks = hierarchical_chunker(sample_sections)
    print(f"3. Hierarchical Chunks: {len(hierarchical_chunks)}")
    
    print("\n   Sample Hierarchical Child Metadata:")
    for chunk in hierarchical_chunks:
        if chunk["strategy"] == "hierarchical_child":
            print(f"   {chunk['metadata']}")
            break
            
    print("\n4. Semantic Chunks:")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            semantic_chunks = semantic_chunker(sample_sections)
            print(f"   Count: {len(semantic_chunks)}")
        except Exception as e:
            print(f"   Failed to run semantic chunker: {e}")
    else:
        print("   Skipped: OPENAI_API_KEY not set in environment.")
        
    print("\nDone.")

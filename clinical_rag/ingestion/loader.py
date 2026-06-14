# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(filepath: str) -> list[dict]:
    """
    Loads a PDF file and returns a list of dictionaries representing sections.
    Each dictionary has 'source', 'section', 'page', and 'text'.
    """
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    
    sections = []
    base_name = os.path.basename(filepath)
    
    for i, doc in enumerate(docs):
        # We simulate sections by treating pages as rough sections
        sections.append({
            "source": base_name,
            "section": f"Page {doc.metadata.get('page', i+1)}",
            "page": doc.metadata.get("page", i+1),
            "text": doc.page_content
        })
    return sections

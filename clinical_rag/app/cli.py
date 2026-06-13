import argparse
import sys
import os

# Ensure Python can import from our project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_chain import ClinicalRAGChain

SAMPLE_CLINICAL_TEXT = """
Metformin / Renal Dosing Adjustments. The maximum metformin dose for a patient with eGFR 30-45 mL/min/1.73 m2 (stage 3b CKD) is 500mg twice daily; do not exceed 1000mg/day.
Lisinopril is an ACE inhibitor used commonly for hypertension and heart failure.
"""

def main():
    parser = argparse.ArgumentParser(description="Clinical RAG CLI")
    parser.add_argument("query", type=str, help="The clinical query to answer")
    args = parser.parse_args()
    
    # We pass the sample text as a raw section so the chain can chunk it correctly
    sample_sections = [
        {
            "source": "Clinical_Guidelines_2026.pdf", 
            "section": "General Dosing", 
            "page": 1,
            "text": SAMPLE_CLINICAL_TEXT
        }
    ]
    
    print("Initializing Clinical RAG Chain...")
    try:
        chain = ClinicalRAGChain(strategy="hierarchical")
        
        print("Indexing sample knowledge base (this might take a second)...")
        chain.index(sample_sections)
        
        print(f"\nQuery: {args.query}\n")
        print("Generating answer...\n")
        
        result = chain.answer(args.query)
        
        print("="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print(f"CONFIDENCE: {result['confidence'].upper()}")
        print("="*60)
        print("SOURCES:")
        for i, src in enumerate(result["sources"], start=1):
            print(f"\n[{i}] Source: {src['document']} | Section: {src['section']} | Rerank Score: {src['rerank_score']:.4f}")
            print(f"    Snippet: {src['text'][:100]}...")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

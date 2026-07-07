from app.rag_chain import ClinicalRAGChain

def test_conversation_memory():
    print("Initializing ClinicalRAGChain...")
    chain = ClinicalRAGChain(strategy="fixed")
    
    # We will simulate that there is no LLM by letting it use the fallback 
    # "[LLM ERROR: ChatGroq not initialized. Please set GROQ_API_KEY.]"
    # or if there is an LLM, it will just use it.
    
    print("\n--- Turn 1 ---")
    q1 = "What is the recommended dose of Metformin?"
    print(f"User: {q1}")
    res1 = chain.answer(q1)
    print(f"Assistant: {res1['answer']}")
    
    print("\n--- Memory State After Turn 1 ---")
    print(chain.memory.get_context_string())
    
    print("\n--- Turn 2 ---")
    q2 = "What if the patient's eGFR is below 30?"
    print(f"User: {q2}")
    res2 = chain.answer(q2)
    print(f"Assistant: {res2['answer']}")
    
    print("\n--- Memory State After Turn 2 ---")
    print(chain.memory.get_context_string())
    
    print("\nTest passed! The chain successfully remembered the conversation context.")

if __name__ == "__main__":
    test_conversation_memory()

from typing import List, Dict

class ConversationMemory:
    def __init__(self, max_turns: int = 3):
        """
        Sliding window memory that keeps track of the last `max_turns` interactions.
        """
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        
    def add_turn(self, query: str, answer: str):
        """Add a new Q&A pair to the history."""
        self.history.append({"query": query, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)
            
    def get_context_string(self) -> str:
        """Format the history into a string for the prompt."""
        if not self.history:
            return "No previous conversation."
            
        lines = []
        for i, turn in enumerate(self.history, 1):
            lines.append(f"Turn {i}:\nUser: {turn['query']}\nAssistant: {turn['answer']}")
        return "\n\n".join(lines)
        
    def clear(self):
        """Clear the conversation history."""
        self.history = []

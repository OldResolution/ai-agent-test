from app.models import SessionState

class ConversationManager:
    def __init__(self, max_history: int = 6):
        self.state = SessionState()
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.state.messages.append({"role": role, "content": content})
        if len(self.state.messages) > self.max_history:
            # Always keep system prompt if it was the first message? 
            # In our setup, we'll pass system prompt separately to Ollama each time.
            self.state.messages = self.state.messages[-self.max_history:]

    def get_history(self) -> list[dict]:
        return self.state.messages

    def set_order_id(self, order_id: str):
        self.state.current_order_id = order_id
        
    def get_order_id(self) -> str | None:
        return self.state.current_order_id

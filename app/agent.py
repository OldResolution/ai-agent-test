import re
from typing import List, Dict
from ollama import Client
from app.config import settings
from app.models import AgentResponse
from app.conversation import ConversationManager
from app.prompts import SYSTEM_PROMPT, format_knowledge_context, format_order_context
from app.tools import lookup_order
from app.retrieval.retriever import HybridRetriever
from app.retrieval.index import KnowledgeIndex

ORDER_ID_PATTERN = r"\b[oO][rR][dD]-\d{4}\b"
ORDER_INTENT_KEYWORDS = {"order", "status", "shipped", "arrive", "where", "tracking"}

class SupportAgent:
    def __init__(self, index: KnowledgeIndex):
        self.retriever = HybridRetriever(index)
        self.conversation = ConversationManager()
        self.client = Client(host=settings.ollama_host)
        
    def _is_order_intent(self, text: str) -> bool:
        if re.search(ORDER_ID_PATTERN, text):
            return True
        if self.conversation.get_order_id():
            text_lower = text.lower()
            return any(kw in text_lower for kw in ORDER_INTENT_KEYWORDS)
        return False
        
    def _extract_order_id(self, text: str) -> str | None:
        match = re.search(ORDER_ID_PATTERN, text)
        if match:
            return match.group(0).upper()
        return self.conversation.get_order_id()
        
    def chat(self, user_input: str) -> AgentResponse:
        self.conversation.add_message("user", user_input)
        
        is_order = self._is_order_intent(user_input)
        handoff = False
        sources = []
        context_str = ""
        
        if is_order:
            order_id = self._extract_order_id(user_input)
            if not order_id:
                answer = "Please provide your order ID, such as ORD-1007."
                self.conversation.add_message("assistant", answer)
                return AgentResponse(answer=answer, handoff=False, sources=[])
                
            self.conversation.set_order_id(order_id)
            order_result = lookup_order(order_id)
            
            if order_result.get("human_assistance_required"):
                handoff = True
                
            context_str = format_order_context(order_result)
            if settings.debug:
                print(f"[DEBUG] Route: Order Lookup")
                print(f"[DEBUG] Current Order ID: {order_id}")
                print(f"[DEBUG] Sanitized Order Result: {order_result}")
        else:
            if settings.debug:
                print(f"[DEBUG] Route: Knowledge Retrieval")
            
            retrieval_result = self.retriever.search(user_input, top_k=3)
            
            if settings.debug:
                print(f"[DEBUG] Retrieval Trace:\n{retrieval_result.debug_trace}")
                
            if retrieval_result.insufficient:
                handoff = True
                context_str = "The supplied company information is insufficient to answer reliably. Do not guess. Recommend human assistance.\n\n"
            elif retrieval_result.conflict:
                handoff = True
                context_str = "Current authoritative sources conflict. Do not silently select one. Recommend human confirmation.\n\n"
                
            for chunk in retrieval_result.chunks:
                context_str += format_knowledge_context(chunk.filename, chunk.heading, chunk.text) + "\n"
                sources.append(chunk.source_reference)
                
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Copy history so we don't modify the stored messages
        history = self.conversation.get_history()
        for msg in history[:-1]:
            messages.append(msg.copy())
            
        last_msg = history[-1].copy()
        if context_str:
            last_msg["content"] += f"\n\nContext:\n{context_str}"
        messages.append(last_msg)
            
        response = self.client.chat(model=settings.llm_model, messages=messages)
        answer = response["message"]["content"]
        
        # Add the pure answer to history (without the context injection)
        self.conversation.add_message("assistant", answer)
        
        return AgentResponse(
            answer=answer,
            sources=list(dict.fromkeys(sources)),
            handoff=handoff
        )

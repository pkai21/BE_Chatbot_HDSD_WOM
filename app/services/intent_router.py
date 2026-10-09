from typing import Optional, Dict, List, Tuple
from app.core.logger import logger
from app.services.intent_service import intent_service

STRATEGY_PROCEDURAL_EXTRACTIVE = "procedural_extractive"
STRATEGY_TARGETED_QA = "targeted_qa"
STRATEGY_OUT_OF_SCOPE = "out_of_scope"


class IntentRouter:
    """
    Zero-Keyword Adapter kết nối với Single-Pass Multi-Aspect Router (Module 4 & 5).
    Đảm bảo tương thích ngược 100% với toàn bộ pipeline RAG hiện tại.
    """

    CONVERSATIONAL_STOPWORDS: List[str] = [
        "cách", "để", "trong", "app", "làm", "sao", "hướng", "dẫn", "cho",
        "tôi", "hỏi", "với", "như", "thế", "nào", "ở", "đâu", "phần", "mềm",
        "trên", "hệ", "thống", "vào", "được", "không", "giúp", "với"
    ]

    def classify_target_module(self, query: str, role: str = "phuong") -> Optional[str]:
        """Classifies user query to an exact business module target via Policy Router."""
        result = intent_service.classify_intent(query, role=role)
        return result.target_module

    def classify_query_strategy(self, query: str, role: str = "phuong") -> Tuple[str, Optional[str]]:
        """
        Determines the optimal execution strategy via Single-Pass Multi-Aspect Router:
        1. STRATEGY_PROCEDURAL_EXTRACTIVE: Full procedural How-To action workflow.
        2. STRATEGY_TARGETED_QA: Wh-questions, specific entity, condition or single detail inquiry.
        """
        result = intent_service.classify_intent(query, role=role)
        logger.info(f"Single-Pass Router: Strategy={result.strategy} | Target={result.target_module} | Query='{query}'")
        return result.strategy, result.target_module

    async def classify_query_strategy_async(self, query: str, role: str = "phuong") -> Tuple[str, Optional[str]]:
        """Async version of classify_query_strategy for FastAPI and SSE streaming."""
        result = await intent_service.classify_intent_async(query, role=role)
        logger.info(f"Single-Pass Router (Async): Strategy={result.strategy} | Target={result.target_module} | Query='{query}'")
        return result.strategy, result.target_module

    def strip_conversational_noise(self, query: str) -> str:
        """Strips conversational filler tokens for BM25 search."""
        words = query.lower().split()
        cleaned_words = [w for w in words if w not in self.CONVERSATIONAL_STOPWORDS]
        cleaned_query = " ".join(cleaned_words)
        return cleaned_query if cleaned_query else query


intent_router = IntentRouter()

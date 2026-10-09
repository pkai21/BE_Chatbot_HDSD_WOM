from typing import Optional, Dict, List
from app.models.chat import QuickActionChip


class IntentResult:
    """
    Data model biểu diễn kết quả phân loại của Intent Router và Semantic Router.
    """
    def __init__(
        self,
        intent: str,
        direct_answer: Optional[str] = None,
        quick_action_chips: Optional[List[QuickActionChip]] = None,
        confidence_score: float = 1.0,
        matched_exemplar: Optional[str] = None,
        all_scores: Optional[Dict[str, float]] = None,
        target_module: Optional[str] = None,
        strategy: str = "targeted_qa"
    ):
        self.intent = intent
        self.direct_answer = direct_answer
        self.quick_action_chips = quick_action_chips
        self.confidence_score = confidence_score
        self.matched_exemplar = matched_exemplar
        self.all_scores = all_scores or {intent: confidence_score}
        self.target_module = target_module
        self.strategy = strategy

    def __iter__(self):
        return iter((self.intent, self.direct_answer))

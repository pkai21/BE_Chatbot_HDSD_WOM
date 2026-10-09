import os
import pytest
from pathlib import Path
from app.services.semantic_router import semantic_router
from app.services.chat_history_service import ChatHistoryService
from app.models.chat import ChatResponse, QuickActionChip


def test_semantic_router_warmup_eagerly_loads_model_in_ram():
    """TDD RED: semantic_router.warmup() MUST eagerly load the SentenceTransformer model into RAM."""
    # Reset model to simulate fresh server startup
    semantic_router._model = None
    semantic_router._is_warmed_up = False

    semantic_router.warmup()

    # The model weights must be pre-loaded in memory so that first user query is instant (< 500ms)
    assert semantic_router._is_warmed_up is True
    assert semantic_router._model is not None, "SentenceTransformer model must be loaded in RAM during warmup!"


def test_chat_history_local_sqlite_fallback_on_unreachable_cloud(tmp_path):
    """TDD RED: ChatHistoryService must gracefully fallback to local SQLite when Supabase is unreachable."""
    service = ChatHistoryService()
    # Point to an unreachable host to simulate local offline environment
    service.db_url = "postgresql://postgres:fake@db.unreachable-domain-xyz-123.supabase.co:5432/postgres"
    service.sqlite_db_path = tmp_path / "test_chat.db"
    service._init_sqlite()

    import uuid
    session_id = str(uuid.uuid4())
    query = "cho tôi xem hướng dẫn"
    response = ChatResponse(
        session_id=session_id,
        answer="Đây là câu trả lời kiểm thử",
        intent="knowledge_query",
        quick_action_chips=[QuickActionChip(id="chip-1", label="Chi tiết", query_text="chi tiết")]
    )

    # 1. Saving should NOT raise OperationalError and should succeed via SQLite
    success = service._save_interaction_sync(
        session_id=session_id,
        user_query=query,
        bot_response=response,
        role="phuong"
    )
    assert success is True, "Saving interaction should succeed via local SQLite fallback"

    # 2. Retrieving sessions should return the saved session
    sessions = service.get_all_sessions(limit=10)
    assert len(sessions) >= 1
    assert any(s["id"] == session_id for s in sessions)

    # 3. Retrieving messages should return user and assistant messages
    messages = service.get_session_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == query
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == response.answer

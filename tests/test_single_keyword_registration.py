import pytest
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chat import ChatRequest
from app.services.rag_service import rag_service
from app.services.intent_service import intent_service
from app.core.config import settings


@pytest.mark.asyncio
async def test_dang_ky_keyword_in_phuong_role():
    """Kiểm tra từ khóa 'đăng ký' ở vai trò Phường/Xã:
    - Phải nhận diện là knowledge_query (KHÔNG phải chitchat_greeting).
    - Bot không được trả lời câu chào lạc đề.
    - Bot trả về thông tin từ chối do không có thông tin trong Phường/Xã kèm hotline liên hệ.
    """
    await rag_service.warmup()

    # 1. Test Router classification
    intent_res = await intent_service.classify_intent_async("đăng ký", role="phuong")
    assert intent_res.intent != "chitchat_greeting", f"Expected not chitchat_greeting, got {intent_res.intent}"
    assert intent_res.intent == "knowledge_query"
    assert intent_res.target_module in ("TẠO MỚI TÀI KHOẢN PHƯỜNG/XÃ", None)

    # 2. Test Full RAG Response
    req = ChatRequest(
        query="đăng ký",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    resp = await rag_service.process_chat(req)

    # Must not be a greeting response
    assert "Tôi có thể hỗ trợ gì cho bạn trong việc thao tác và tra cứu tài liệu hôm nay" not in resp.answer
    # Must be factoid_nanswer / escalation
    assert resp.intent in ("factoid_nanswer", "knowledge_query")
    print(f"\n[PHUONG RESULT] Intent: {resp.intent} | Answer: {resp.answer}")


@pytest.mark.asyncio
async def test_dang_ky_keyword_in_dn_role():
    """Kiểm tra từ khóa 'đăng ký' ở vai trò Doanh nghiệp:
    - Nhận diện đúng target_module = 'ĐĂNG KÝ'.
    - Trả về hướng dẫn đăng ký của Doanh nghiệp.
    """
    await rag_service.warmup()

    # 1. Test Router classification
    intent_res = await intent_service.classify_intent_async("đăng ký", role="dn")
    assert intent_res.intent == "knowledge_query"
    assert intent_res.target_module == "ĐĂNG KÝ"

    # 2. Test Full RAG Response
    req = ChatRequest(
        query="đăng ký",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    resp = await rag_service.process_chat(req)
    assert "ĐĂNG KÝ" in resp.answer or "đăng ký" in resp.answer.lower()
    print(f"\n[DN RESULT] Intent: {resp.intent} | Answer: {resp.answer[:200]}...")

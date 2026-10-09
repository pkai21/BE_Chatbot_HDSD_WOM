import pytest
import asyncio
from app.services.intent_service import intent_service


@pytest.mark.asyncio
async def test_ambiguity_positive_conversational_variations():
    """
    Kiểm tra các câu hỏi mơ hồ có từ đệm giao tiếp tự nhiên phải được nhận diện
    và kích hoạt làm rõ (Make-Clear) kèm Quick Action Chips.
    Đặc biệt bao gồm các câu gây lỗi ở Turn 2 & Turn 4: 'cho tôi báo cáo tai nạn lao động'.
    """
    queries = [
        "cho tôi báo cáo tai nạn lao động",
        "cho em xin báo cáo tai nạn",
        "tôi muốn xem báo cáo tai nạn lao động",
        "làm ơn hướng dẫn về tai nạn lao động",
        "thao tác tài khoản",
        "cho mình xem thông tin tài khoản"
    ]
    for q in queries:
        result = await intent_service.classify_intent_async(q, role="phuong")
        assert result.intent == "clarification_needed", (
            f"Query '{q}' failed to trigger clarification, got intent='{result.intent}', target_module='{result.target_module}'"
        )
        assert result.quick_action_chips is not None and len(result.quick_action_chips) >= 2, (
            f"Query '{q}' missing quick action chips"
        )
        assert result.target_module is None, (
            f"Query '{q}' should have target_module=None, got '{result.target_module}'"
        )


@pytest.mark.asyncio
async def test_ambiguity_negative_specific_queries():
    """
    Kiểm tra các câu hỏi đã có thực thể/chi tiết cụ thể TUYỆT ĐỐI KHÔNG được
    nhận diện nhầm thành mơ hồ (Ngăn ngừa hiện tượng Semantic False Positive).
    """
    specific_queries = [
        "kích thước tối đa của file đính kèm trong báo cáo TNLĐ đột xuất là bao nhiêu ?",
        "hướng dẫn báo cáo tai nạn lao động định kỳ",
        "hướng dẫn quy trình báo cáo tai nạn lao động đột xuất",
        "hướng dẫn tạo mới tài khoản phường xã",
        "thay đổi thông tin cá nhân cán bộ"
    ]
    for q in specific_queries:
        result = await intent_service.classify_intent_async(q, role="phuong")
        assert result.intent != "clarification_needed", (
            f"Query '{q}' was wrongly flagged as ambiguous clarification_needed!"
        )


@pytest.mark.asyncio
async def test_ambiguity_dn_role():
    """
    Kiểm tra nhận diện mơ hồ đối với vai trò Doanh nghiệp ('dn').
    """
    result = await intent_service.classify_intent_async("cho tôi báo cáo tai nạn lao động", role="dn")
    assert result.intent == "clarification_needed", (
        f"Query for role 'dn' failed to trigger clarification, got intent='{result.intent}'"
    )
    assert result.quick_action_chips is not None and len(result.quick_action_chips) >= 2
    assert any("doanh nghiệp" in c.query_text.lower() for c in result.quick_action_chips)


@pytest.mark.asyncio
async def test_end_to_end_clarification_with_conversational_prefix():
    """
    Kiểm thử tích hợp E2E đầy đủ:
    Lượt 1: User hỏi 'cho tôi báo cáo tai nạn lao động' -> Nhận intent clarification_needed kèm Chips.
    Lượt 2: User trả lời cụt 'đột xuất' kèm history từ lượt 1 -> Nhận đúng quy trình Báo cáo TNLĐ đột xuất.
    """
    from app.models.chat import ChatRequest, ChatMessage
    from app.services.rag_service import rag_service

    await rag_service.warmup()

    # Lượt 1
    req1 = ChatRequest(
        query="cho tôi báo cáo tai nạn lao động",
        history=[],
        role="phuong"
    )
    res1 = await rag_service.process_chat_message(req1)
    assert res1.intent == "clarification_needed"
    assert res1.quick_action_chips is not None
    assert len(res1.quick_action_chips) == 2

    # Lượt 2
    req2 = ChatRequest(
        query="đột xuất",
        history=[
            ChatMessage(role="user", content="cho tôi báo cáo tai nạn lao động"),
            ChatMessage(role="assistant", content=res1.answer)
        ],
        role="phuong"
    )
    res2 = await rag_service.process_chat_message(req2)
    assert res2.intent == "knowledge_query"
    assert len(res2.source_chunks) > 0
    assert "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỘT XUẤT" in res2.answer or "tai nạn" in res2.answer.lower()


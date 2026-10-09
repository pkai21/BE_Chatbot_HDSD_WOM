import pytest
import asyncio
from app.models.chat import ChatRequest, ChatMessage
from app.services.intent_service import intent_service
from app.services.cqr_service import cqr_service
from app.services.rag_service import rag_service


@pytest.mark.asyncio
async def test_ambiguity_detection_phuong():
    """Kiểm tra Ambiguity Detector phát hiện câu hỏi thiếu ngữ cảnh."""
    # Case 1: Báo cáo tai nạn
    result = await intent_service.classify_intent_async("báo cáo tai nạn", role="phuong")
    assert result.intent == "clarification_needed"
    assert "định kỳ" in result.direct_answer and "đột xuất" in result.direct_answer
    assert result.quick_action_chips is not None
    assert len(result.quick_action_chips) == 2
    assert any("đột xuất" in c.label.lower() for c in result.quick_action_chips)
    assert any("định kỳ" in c.label.lower() for c in result.quick_action_chips)

    # Case 2: Tài khoản
    result_acc = await intent_service.classify_intent_async("tài khoản", role="phuong")
    assert result_acc.intent == "clarification_needed"
    assert result_acc.quick_action_chips is not None
    assert len(result_acc.quick_action_chips) == 4


@pytest.mark.asyncio
async def test_cqr_is_dependent_query():
    """Kiểm tra Dependency Gatekeeper phân loại câu độc lập vs câu phụ thuộc."""
    history = [
        ChatMessage(role="user", content="báo cáo tai nạn"),
        ChatMessage(role="assistant", content="Bạn đang muốn tìm hiểu về Báo cáo tai nạn lao động định kỳ hay đột xuất?")
    ]

    # Câu cụt / phụ thuộc -> True
    assert cqr_service.is_dependent_query("đột xuất", history) is True
    assert cqr_service.is_dependent_query("định kỳ", history) is True
    assert cqr_service.is_dependent_query("ở đâu", history) is True

    # Câu độc lập đầy đủ -> False
    assert cqr_service.is_dependent_query("hướng dẫn tạo mới tài khoản phường xã", history) is False
    assert cqr_service.is_dependent_query("cho tôi xin số điện thoại hotline hỗ trợ", history) is False

    # Không có history -> False
    assert cqr_service.is_dependent_query("đột xuất", []) is False


@pytest.mark.asyncio
async def test_cqr_fast_slot_filling():
    """Kiểm tra Fast-Path Slot-Filling tái tạo truy vấn < 1ms."""
    history = [
        ChatMessage(role="user", content="báo cáo tai nạn"),
        ChatMessage(role="assistant", content="Bạn đang muốn tìm hiểu về Báo cáo tai nạn lao động định kỳ hay đột xuất?")
    ]

    resolved_dotxuat, res_type = await cqr_service.resolve_context("đột xuất", history, role="phuong")
    assert res_type == "FAST_SLOT_FILL"
    assert "đột xuất" in resolved_dotxuat.lower()
    assert "báo cáo tai nạn lao động" in resolved_dotxuat.lower()

    resolved_dinhky, res_type = await cqr_service.resolve_context("định kỳ", history, role="phuong")
    assert res_type == "FAST_SLOT_FILL"
    assert "định kỳ" in resolved_dinhky.lower()
    assert "báo cáo tai nạn lao động" in resolved_dinhky.lower()


@pytest.mark.asyncio
async def test_end_to_end_clarification_and_followup():
    """Kiểm thử toàn diện E2E: Lượt 1 hỏi mơ hồ -> Lượt 2 trả lời cụt -> Nhận đúng quy trình."""
    await rag_service.warmup()

    # Lượt 1: User hỏi "báo cáo tai nạn" (chưa có history)
    req1 = ChatRequest(
        query="báo cáo tai nạn",
        history=[],
        role="phuong"
    )
    res1 = await rag_service.process_chat_message(req1)
    assert res1.intent == "clarification_needed"
    assert res1.quick_action_chips is not None
    assert len(res1.quick_action_chips) == 2

    # Lượt 2: User trả lời "đột xuất" kèm history từ lượt 1
    req2 = ChatRequest(
        query="đột xuất",
        history=[
            ChatMessage(role="user", content="báo cáo tai nạn"),
            ChatMessage(role="assistant", content=res1.answer)
        ],
        role="phuong"
    )
    res2 = await rag_service.process_chat_message(req2)
    assert res2.intent == "knowledge_query"
    assert len(res2.source_chunks) > 0
    # Câu trả lời phải chứa hướng dẫn về Báo cáo tai nạn lao động đột xuất
    assert "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỘT XUẤT" in res2.answer or "tai nạn" in res2.answer.lower()

import pytest
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chat import ChatRequest
from app.services.rag_service import rag_service
from app.core.config import settings


@pytest.mark.asyncio
async def test_mode_a_procedural_extractive():
    """Kiểm tra Mode A: Trả về trực tiếp Parent Chunk đầy đủ quy trình."""
    await rag_service.warmup()

    req = ChatRequest(
        query="Hướng dẫn đăng nhập hệ thống tài khoản phường xã",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res = await rag_service.process_chat(req)
    
    assert res.intent == "knowledge_query"
    assert len(res.answer) > 50
    assert "đăng nhập" in res.answer.lower()
    assert len(res.quick_action_chips or []) > 0


@pytest.mark.asyncio
async def test_mode_b_targeted_qa_factoid_dn():
    """Kiểm tra Mode B: Câu hỏi ngách Factoid Doanh nghiệp (Tổng quỹ lương)."""
    await rag_service.warmup()

    req = ChatRequest(
        query="Trường Tổng quỹ lương khi báo cáo TNLĐ nhập đơn vị là Triệu đồng hay gì?",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    res = await rag_service.process_chat(req)

    assert res.intent == "knowledge_query"
    # Should mention ĐỒNG (và không nhập triệu đồng)
    assert "đồng" in res.answer.lower()
    # Scoped Child context check
    assert len(res.source_chunks) <= 4


@pytest.mark.asyncio
async def test_mode_b_targeted_qa_factoid_phuong():
    """Kiểm tra Mode B: Câu hỏi ngách Factoid Phường/Xã (Khung giờ làm việc)."""
    await rag_service.warmup()

    req = ChatRequest(
        query="Buổi sáng tổng đài làm việc từ mấy giờ?",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res = await rag_service.process_chat(req)

    assert res.intent == "knowledge_query"
    assert "07h30" in res.answer or "7h30" in res.answer or "11h30" in res.answer or "sáng" in res.answer.lower()


@pytest.mark.asyncio
async def test_mode_b_targeted_qa_specific_fields_phuong():
    """Kiểm tra Mode B: Câu hỏi chi tiết về các trường thông tin cần điền khi tạo mới tài khoản."""
    await rag_service.warmup()

    req = ChatRequest(
        query="tôi cần điền các thông tin gì khi tạo mới tài khoản ?",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res = await rag_service.process_chat(req)

    assert res.intent == "knowledge_query"
    assert "tên đăng nhập" in res.answer.lower()
    assert "mật khẩu" in res.answer.lower()
    # Phải trả lời đúng trọng tâm Mode B, không hiển thị full quy trình Mode A
    assert not res.answer.startswith("Dưới đây là hướng dẫn chi tiết quy trình")


@pytest.mark.asyncio
async def test_mode_b_targeted_qa_dot_xuat_fields_phuong():
    """Kiểm tra Mode B: Câu hỏi các trường thông tin phần Thông tin Phường/Xã trong Báo cáo TNLĐ đột xuất."""
    await rag_service.warmup()

    req = ChatRequest(
        query="cần nhập thông tin gì ở phần Thông tin Phường/Xã khi khai báo cáo tai nạn lao động đột xuất",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res = await rag_service.process_chat(req)

    assert res.intent == "knowledge_query"
    assert "địa chỉ" in res.answer.lower()
    assert "điện thoại" in res.answer.lower() or "fax" in res.answer.lower()
    assert not res.answer.startswith("Dưới đây là hướng dẫn chi tiết quy trình")
    assert "không có thông tin" not in res.answer.lower()



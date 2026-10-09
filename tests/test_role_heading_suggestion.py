import pytest
from app.services.suggestion_service import suggestion_service


def test_dn_role_receives_only_dn_heading1_chips():
    """Test that role 'dn' only receives suggestions from DN Heading Level 1s."""
    chips = suggestion_service.get_suggested_chips(
        query="hướng dẫn chung về phần mềm",
        role="dn",
        limit=3
    )
    assert len(chips) == 3
    for chip in chips:
        full_text = (chip.label + " " + chip.query_text).lower()
        assert "phường" not in full_text, f"Leak phuong detected in DN chip: {chip}"
        assert "tuyến dưới" not in full_text, f"Leak phuong detected in DN chip: {chip}"
        assert "không có hđlđ" not in full_text, f"Leak phuong detected in DN chip: {chip}"
        assert "không theo hđlđ" not in full_text, f"Leak phuong detected in DN chip: {chip}"


def test_phuong_role_receives_only_phuong_heading1_chips():
    """Test that role 'phuong' only receives suggestions from Phuong Heading Level 1s."""
    chips = suggestion_service.get_suggested_chips(
        query="hướng dẫn chung về phần mềm",
        role="phuong",
        limit=3
    )
    assert len(chips) == 3
    for chip in chips:
        full_text = (chip.label + " " + chip.query_text).lower()
        assert "doanh nghiệp mới" not in full_text, f"Leak DN detected in Phuong chip: {chip}"
        assert "mã số thuế" not in full_text, f"Leak DN detected in Phuong chip: {chip}"


def test_active_heading_is_excluded():
    """Test that the active heading is excluded from suggested follow-up chips."""
    chips = suggestion_service.get_suggested_chips(
        query="hướng dẫn đăng ký tài khoản doanh nghiệp",
        role="dn",
        target_module="ĐĂNG KÝ",
        limit=3
    )
    assert len(chips) == 3
    for chip in chips:
        # Must not suggest 'Đăng ký' when active module is 'ĐĂNG KÝ'
        assert "đăng ký" not in chip.label.lower(), f"Active heading not excluded: {chip.label}"


def test_fallback_preserves_role_isolation():
    """Test that unknown queries without target module preserve strict role isolation."""
    chips_dn = suggestion_service.get_suggested_chips(
        query="asdfghjkl không khớp gì cả",
        role="dn",
        target_module=None,
        primary_chunk=None,
        limit=3
    )
    assert len(chips_dn) == 3
    for chip in chips_dn:
        full_text = (chip.label + " " + chip.query_text).lower()
        assert "phường" not in full_text
        assert "tuyến dưới" not in full_text
        assert "không có hđlđ" not in full_text
        assert "không theo hđlđ" not in full_text


@pytest.mark.asyncio
async def test_e2e_rag_service_role_dn_chips():
    """Kiểm tra End-to-End RAGService với role 'dn' trả về chips chuẩn Heading 1 của DN."""
    from app.services.rag_service import rag_service
    from app.models.chat import ChatRequest
    from app.core.config import settings

    await rag_service.warmup()
    req = ChatRequest(
        query="Hướng dẫn đăng ký tài khoản doanh nghiệp",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    res = await rag_service.process_chat(req)

    assert res.quick_action_chips is not None
    assert len(res.quick_action_chips) == 3
    for chip in res.quick_action_chips:
        full_text = (chip.label + " " + chip.query_text).lower()
        assert "phường" not in full_text, f"Leak phuong in E2E DN: {chip}"
        assert "tuyến dưới" not in full_text, f"Leak phuong in E2E DN: {chip}"
        assert "không có hđlđ" not in full_text, f"Leak phuong in E2E DN: {chip}"
        assert "không theo hđlđ" not in full_text, f"Leak phuong in E2E DN: {chip}"
        # Heading Đăng ký phải bị loại trừ
        assert "đăng ký" not in chip.label.lower()


@pytest.mark.asyncio
async def test_e2e_rag_service_role_phuong_chips():
    """Kiểm tra End-to-End RAGService với role 'phuong' trả về chips chuẩn Heading 1 của Phường."""
    from app.services.rag_service import rag_service
    from app.models.chat import ChatRequest
    from app.core.config import settings

    await rag_service.warmup()
    req = ChatRequest(
        query="Hướng dẫn tạo mới tài khoản phường xã",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res = await rag_service.process_chat(req)

    assert res.quick_action_chips is not None
    assert len(res.quick_action_chips) == 3
    for chip in res.quick_action_chips:
        full_text = (chip.label + " " + chip.query_text).lower()
        assert "doanh nghiệp mới" not in full_text, f"Leak DN in E2E Phuong: {chip}"
        assert "mã số thuế" not in full_text, f"Leak DN in E2E Phuong: {chip}"
        # Heading Tạo mới tài khoản phải bị loại trừ
        assert "tạo tài khoản" not in chip.label.lower()


@pytest.mark.asyncio
async def test_contact_inquiry_clean_display():
    """Kiểm tra câu hỏi hotline/hỗ trợ chỉ trả về contact_support card, answer rỗng để không trùng lặp."""
    from app.services.rag_service import rag_service
    from app.models.chat import ChatRequest
    from app.core.config import settings

    await rag_service.warmup()

    # Test DN
    req_dn = ChatRequest(
        query="Cho tôi thông tin hotline và zalo hỗ trợ kỹ thuật",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    res_dn = await rag_service.process_chat(req_dn)
    assert res_dn.contact_support is not None, "contact_support phải có để hiển thị Card vàng"
    assert res_dn.answer == "", f"answer phải rỗng khi đã có Card vàng, nhận được: {res_dn.answer}"

    # Test Phuong
    req_p = ChatRequest(
        query="Cho tôi thông tin hotline hỗ trợ kỹ thuật",
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_p = await rag_service.process_chat(req_p)
    assert res_p.contact_support is not None, "contact_support phải có để hiển thị Card vàng"
    assert res_p.answer == "", f"answer phải rỗng khi đã có Card vàng, nhận được: {res_p.answer}"


@pytest.mark.asyncio
async def test_dn_report_chips_comprehensive_content():
    """Kiểm tra khi click gợi ý Báo cáo TNLĐ và ATVSLĐ phải trả về đầy đủ các bước, không bị cộc lốc 94 ký tự."""
    from app.services.rag_service import rag_service
    from app.models.chat import ChatRequest
    from app.core.config import settings

    await rag_service.warmup()

    # 1. Báo cáo định kỳ tai nạn lao động
    req_tnld = ChatRequest(
        query="Hướng dẫn nộp báo cáo định kỳ tai nạn lao động",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    res_tnld = await rag_service.process_chat(req_tnld)
    assert len(res_tnld.answer) > 500, f"Câu trả lời quá ngắn ({len(res_tnld.answer)} chars): {res_tnld.answer}"
    assert "tai nạn lao động" in res_tnld.answer.lower()
    assert "bước" in res_tnld.answer.lower()

    # 2. Báo cáo định kỳ An toàn vệ sinh lao động
    req_atvsld = ChatRequest(
        query="Hướng dẫn nộp báo cáo định kỳ An toàn vệ sinh lao động",
        role="dn",
        collection=settings.CHROMA_COLLECTION_DN
    )
    res_atvsld = await rag_service.process_chat(req_atvsld)
    assert len(res_atvsld.answer) > 500, f"Câu trả lời quá ngắn ({len(res_atvsld.answer)} chars): {res_atvsld.answer}"
    assert "vệ sinh lao động" in res_atvsld.answer.lower()
    assert "bước" in res_atvsld.answer.lower()


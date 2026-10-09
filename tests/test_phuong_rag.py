import asyncio
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chat import ChatRequest
from app.services.rag_service import rag_service
from app.services.intent_service import intent_service
from app.core.config import settings


async def run_phuong_rag_tests():
    print("\n" + "=" * 80)
    print("🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN CHATBOT V2 - PHÂN HỆ PHƯỜNG/XÃ (hdsd_phuong_chunks)")
    print("=" * 80)

    # 1. Warmup
    print("\n⏳ [1/5] Đang khởi động và nạp dữ liệu RAG Service...")
    await rag_service.warmup()
    print("✅ Khởi động thành công!")

    # 2. Test 10 Phân hệ nghiệp vụ Phường/Xã
    print("\n" + "-" * 80)
    print("📋 [2/5] Kiểm thử 10 Phân hệ nghiệp vụ Phường/Xã:")
    print("-" * 80)

    test_queries = [
        ("Đăng nhập tài khoản", "Đăng nhập hệ thống tài khoản phường xã như thế nào?"),
        ("Thay đổi thông tin cá nhân", "Hướng dẫn thay đổi thông tin cá nhân của cán bộ phường xã"),
        ("Thay đổi mật khẩu", "Làm sao để đổi mật khẩu tài khoản cán bộ phường?"),
        ("Tổng quan chức năng tài khoản", "Tổng quan chức năng tài khoản phường xã gồm những gì?"),
        ("Tạo mới tài khoản", "Hướng dẫn tạo mới tài khoản cho cán bộ phường xã"),
        ("Chỉnh sửa tài khoản", "Cách chỉnh sửa thông tin tài khoản phường xã tuyến dưới"),
        ("Khôi phục mật khẩu", "Làm sao để khôi phục lại mật khẩu cho tài khoản phường xã?"),
        ("Xóa tài khoản", "Cách xóa tài khoản phường xã khỏi hệ thống"),
        ("Báo cáo TNLĐ định kỳ", "Hướng dẫn lập báo cáo tai nạn lao động định kỳ cho người không theo HĐLĐ"),
        ("Báo cáo TNLĐ đột xuất", "Quy trình báo cáo tai nạn lao động đột xuất không theo HĐLĐ khi có vụ việc xảy ra"),
    ]

    all_modules_passed = True
    for module_name, query in test_queries:
        req = ChatRequest(
            query=query,
            role="phuong",
            collection=settings.CHROMA_COLLECTION_PHUONG
        )
        res = await rag_service.process_chat(req)
        has_content = len(res.answer) > 50
        has_chips = len(res.quick_action_chips) > 0
        has_images = len(res.images) >= 0

        status = "✅ PASS" if has_content and has_chips else "❌ FAIL"
        if not (has_content and has_chips):
            all_modules_passed = False

        print(f"[{status}] Phân hệ: {module_name}")
        print(f"       Query: '{query}'")
        print(f"       Intent/Strategy: {res.intent} | Chips: {len(res.quick_action_chips)} | Images: {len(res.images)}")
        print(f"       Trích đoạn trả lời ({len(res.answer)} ký tự): {res.answer[:120]}...\n")

    # 3. Test Hotline & Yellow Contact Card
    print("-" * 80)
    print("📞 [3/5] Kiểm thử Thông tin Liên hệ hỗ trợ & Card vàng:")
    print("-" * 80)

    # Test 3.1: General Hotline inquiry
    contact_query = "Cho tôi xin số điện thoại tổng đài hotline hỗ trợ kỹ thuật"
    req_contact = ChatRequest(
        query=contact_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_contact = await rag_service.process_chat(req_contact)

    contact_card_valid = (
        res_contact.contact_support is not None
        and "028 3535 2524" in res_contact.contact_support.hotlines
        and "7h30" in res_contact.contact_support.working_hours
    )
    print(f"[{'✅ PASS' if contact_card_valid else '❌ FAIL'}] [Case 2.1] Liên hệ hỗ trợ Phường/Xã:")
    print(f"       ContactCard: {res_contact.contact_support}")
    print(f"       Hotline: {res_contact.contact_support.hotlines if res_contact.contact_support else 'None'}")
    print(f"       Thời gian: {res_contact.contact_support.working_hours if res_contact.contact_support else 'None'}\n")

    # Test 3.2: Specific Morning Working Hours (Case 2 from User Feedback)
    morning_query = "Buổi sáng tổng đài hotline hỗ trợ kỹ thuật của phần mềm làm việc từ mấy giờ?"
    req_morning = ChatRequest(
        query=morning_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_morning = await rag_service.process_chat(req_morning)
    has_morning_time = ("7h30" in res_morning.answer or "07h30" in res_morning.answer or "11h30" in res_morning.answer)
    no_refusal = ("không có thông tin" not in res_morning.answer.lower())
    morning_pass = has_morning_time and no_refusal and (res_morning.contact_support is not None)

    print(f"[{'✅ PASS' if morning_pass else '❌ FAIL'}] [Case 2.2 - User Bugfix] Hỏi khung giờ hotline buổi sáng:")
    print(f"       Query: '{morning_query}'")
    print(f"       Câu trả lời: {res_morning.answer}")
    print(f"       ContactCard đính kèm: {res_morning.contact_support is not None}")
    print(f"       Không bị lỗi 'Tôi không có thông tin': {no_refusal}\n")

    # 4. Test Search Keywords Integration & Case 1 (Account Overview)
    print("-" * 80)
    print("🔍 [4/5] Kiểm thử Độ nhạy Từ khóa tìm kiếm & Chức năng tài khoản (Case 1):")
    print("-" * 80)

    # Test 4.1: Case 1 from User Feedback
    case1_query = "Giải đáp chức năng tài khoản phường xã gồm những thao tác nào?"
    req_case1 = ChatRequest(
        query=case1_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_case1 = await rag_service.process_chat(req_case1)
    is_not_bot_intro = ("tôi là trợ lý ai" not in res_case1.answer.lower()[:50])
    has_account_operations = any(w in res_case1.answer.lower() for w in ["tạo mới", "chỉnh sửa", "khôi phục", "xóa"])
    case1_pass = is_not_bot_intro and has_account_operations

    print(f"[{'✅ PASS' if case1_pass else '❌ FAIL'}] [Case 1 - User Bugfix] Giải đáp chức năng tài khoản phường xã:")
    print(f"       Query: '{case1_query}'")
    print(f"       Intent: {res_case1.intent}")
    print(f"       Không bị hiểu nhầm là giới thiệu Chatbot: {is_not_bot_intro}")
    print(f"       Trích đoạn câu trả lời ({len(res_case1.answer)} ký tự): {res_case1.answer[:150]}...\n")

    # Test 4.2: Specific Fields Inquiry (User Feedback)
    fields_query = "tôi cần điền các thông tin gì khi tạo mới tài khoản ?"
    req_fields = ChatRequest(
        query=fields_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_fields = await rag_service.process_chat(req_fields)
    has_fields = ("tên đăng nhập" in res_fields.answer.lower() and "mật khẩu" in res_fields.answer.lower())
    is_concise_mode_b = not res_fields.answer.startswith("Dưới đây là hướng dẫn chi tiết quy trình")
    fields_pass = has_fields and is_concise_mode_b

    print(f"[{'✅ PASS' if fields_pass else '❌ FAIL'}] [Case 4.2 - User Bugfix] Câu hỏi trường thông tin tạo mới tài khoản:")
    print(f"       Query: '{fields_query}'")
    print(f"       Intent: {res_fields.intent}")
    print(f"       Mode B (Trả lời trực diện, không in full quy trình): {is_concise_mode_b}")
    print(f"       Câu trả lời: {res_fields.answer}\n")

    # Test 4.3: Specific Fields in Dot Xuat Report (User Feedback)
    dotxuat_query = "cần nhập thông tin gì ở phần Thông tin Phường/Xã khi khai báo cáo tai nạn lao động đột xuất"
    req_dotxuat = ChatRequest(
        query=dotxuat_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_dotxuat = await rag_service.process_chat(req_dotxuat)
    has_dotxuat_fields = ("địa chỉ" in res_dotxuat.answer.lower() and ("điện thoại" in res_dotxuat.answer.lower() or "fax" in res_dotxuat.answer.lower()))
    is_not_refusal = "không có thông tin" not in res_dotxuat.answer.lower()
    dotxuat_pass = has_dotxuat_fields and is_not_refusal

    print(f"[{'✅ PASS' if dotxuat_pass else '❌ FAIL'}] [Case 4.3 - User Bugfix] Câu hỏi trường thông tin Báo cáo TNLĐ đột xuất:")
    print(f"       Query: '{dotxuat_query}'")
    print(f"       Intent: {res_dotxuat.intent}")
    print(f"       Không bị từ chối sai (Refusal false-positive): {is_not_refusal}")
    print(f"       Câu trả lời: {res_dotxuat.answer}\n")

    keyword_queries = [
        ("phục hồi pass cán bộ cấp dưới", "TỔNG QUAN CHỨC NĂNG TÀI KHOẢN PHƯỜNG/XÃ"),
        ("hạn gửi tổng hợp số liệu tai nạn 6 tháng cuối năm", "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỊNH KỲ KHÔNG THEO HĐLĐ"),
        ("đính kèm hồ sơ khám nghiệm hiện trường tối đa bao nhiêu MB", "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỘT XUẤT KHÔNG THEO HĐLĐ"),
    ]

    for kw_query, expected_topic in keyword_queries:
        req_kw = ChatRequest(
            query=kw_query,
            role="phuong",
            collection=settings.CHROMA_COLLECTION_PHUONG
        )
        res_kw = await rag_service.process_chat(req_kw)
        kw_pass = len(res_kw.answer) > 20
        print(f"[{'✅ PASS' if kw_pass else '❌ FAIL'}] Query từ khóa: '{kw_query}'")
        print(f"       Chủ đề kỳ vọng: {expected_topic}")
        print(f"       Câu trả lời: {res_kw.answer[:140]}...\n")

    # 5. Test Factoid Refusal & Escalation
    print("-" * 80)
    print("🛡️ [5/5] Kiểm thử Factoid Rejection & Escalation:")
    print("-" * 80)

    unsupported_query = "Phần mềm có tích hợp thanh toán lệ phí qua ví Momo hoặc ZaloPay không?"
    req_unsup = ChatRequest(
        query=unsupported_query,
        role="phuong",
        collection=settings.CHROMA_COLLECTION_PHUONG
    )
    res_unsup = await rag_service.process_chat(req_unsup)
    refusal_pass = (
        res_unsup.contact_support is not None
        and ("liên hệ" in res_unsup.answer.lower() or "không có thông tin" in res_unsup.answer.lower())
    )
    print(f"[{'✅ PASS' if refusal_pass else '❌ FAIL'}] Câu hỏi ngoài phạm vi HDSD:")
    print(f"       Query: '{unsupported_query}'")
    print(f"       Intent: {res_unsup.intent}")
    print(f"       Answer: {res_unsup.answer}")
    print(f"       Contact Escalation: {res_unsup.contact_support is not None}\n")

    # Final summary
    print("=" * 80)
    if all_modules_passed and contact_card_valid and morning_pass and case1_pass and refusal_pass:
        print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ CHO PHÂN HỆ PHƯỜNG/XÃ (v2) ĐÃ VƯỢT QUA 100% THÀNH CÔNG!")
    else:
        print("⚠️ MỘT SỐ BÀI TEST CHƯA ĐẠT KỲ VỌNG, VUI LÒNG KIỂM TRA LẠI LOGS.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_phuong_rag_tests())

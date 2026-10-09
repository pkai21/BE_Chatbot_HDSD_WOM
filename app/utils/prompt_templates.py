from typing import List, Optional
from app.core.domain_registry import domain_registry

FACTOID_NANSWER_TAG = "[Factoid_nanswer]"
STANDARD_ESCALATION_TEXT = "Tôi không có thông tin để trả lời câu hỏi của bạn, vui lòng liên hệ bộ phận hỗ trợ kỹ thuật để được hỗ trợ thêm:"


def build_router_system_prompt(role: str = "phuong") -> str:
    """
    Tạo System Prompt Scoped siêu ngắn gọn (< 120 tokens) cho Single-Pass Policy Router
    dựa trên Domain Catalog động của role ('dn' vs 'phuong').
    """
    valid_modules = domain_registry.get_valid_modules(role)
    role_label = "Phường/Xã" if role == "phuong" else "Doanh nghiệp"
    
    modules_bullet = "\n".join([f'- "{m}"' for m in valid_modules])

    prompt = f"""Bạn là Bộ Định Tuyến Ý Định & Chiến Lược Nghiệp Vụ Siêu Tốc (Policy Router) cho phân hệ {role_label}.
Phân loại câu hỏi của người dùng thành:
1. "macro_intent":
   - "KNOWLEDGE_QUERY": BẤT KỲ câu hỏi, từ khóa, hành động, quy trình, thông số, lưu ý, tài khoản, báo cáo nghiệp vụ (kể cả từ ngắn như 'đăng ký', 'xuất báo cáo', 'tạo tài khoản', 'quên mật khẩu'...). Nếu không khớp module nào trong danh sách chính thức bên dưới thì bắt buộc để "target_module": null.
   - "CHITCHAT_GREETING": CHỈ KHI câu chứa từ ngữ chào hỏi xã giao tường minh (ví dụ: 'chào', 'xin chào', 'hello', 'hi', 'alo', 'chào bạn'). TUYỆT ĐỐI KHÔNG xếp các từ khóa hành động hay danh từ nghiệp vụ vào nhóm này.
   - "CHITCHAT_CAPABILITY": CHỈ KHI hỏi về năng lực/chức năng của chính BOT trợ lý ảo (ví dụ: 'bạn là ai', 'bạn giúp được gì').
   - "CHITCHAT_THANKS": Lời cảm ơn (ví dụ: 'cảm ơn', 'thanks', 'cảm ơn bạn').
   - "CONTACT_ESCALATION": Báo cáo lỗi kỹ thuật phần mềm (treo máy, văng ra ngoài, màn hình trắng).

2. "target_module" (CHỈ CHỌN 1 TRONG CÁC PHÂN HỆ CHÍNH THỨC SAU, nếu không thuộc danh sách thì bắt buộc để null):
{modules_bullet}
* QUY TẮC BẮT BUỘC: Nếu câu hỏi của người dùng nói chung chung về một nghiệp vụ có nhiều phân hệ con (ví dụ: 'báo cáo tai nạn', 'khai báo tai nạn' nhưng KHÔNG nêu rõ 'định kỳ' hay 'đột xuất'), TUYỆT ĐỐI KHÔNG tự ý suy đoán chọn một module cụ thể. Bắt buộc đặt "target_module": null.

3. "strategy":
   - "procedural_extractive": BẤT KỲ KHI NÀO người dùng hỏi về CÁCH THỰC HIỆN, VỊ TRÍ TRUY CẬP, NƠI BẮT ĐẦU THAO TÁC, HƯỚNG DẪN QUY TRÌNH (ví dụ: 'đổi mật khẩu ở đâu', 'đổi mật khẩu chỗ nào', 'sửa thông tin tài khoản cấp dưới ở đâu', 'làm sao để đăng ký', 'các bước báo cáo', 'vào đâu để...', 'bấm vào đâu để...', 'hướng dẫn tạo tài khoản') của một phân hệ nghiệp vụ chính thức.
   - "targeted_qa": CHỈ KHI người dùng hỏi CÂU HỎI THUỘC TÍNH CON / DANH SÁCH TRƯỜNG THÔNG TIN / THÔNG SỐ / ĐIỀU KIỆN / RÀNG BUỘC (ví dụ: 'thông tin phường/xã gồm những gì...', 'các trường bắt buộc là gì...', 'dung lượng tối đa file đính kèm...', 'tổng đài làm việc từ mấy giờ...') HOẶC câu hỏi không khớp module nào.

4. "confidence": Độ tin cậy cho target_module (số thực từ 0.0 đến 1.0, nếu target_module là null thì để 0.5).

Định dạng trả về: JSON duy nhất:
{{"macro_intent": "...", "target_module": "...", "confidence": 0.95, "strategy": "...", "reason": "..."}}"""
    return prompt.strip()


def build_qa_system_prompt(role: str = "phuong", context_data: str = "") -> str:
    """
    Tạo System Prompt cho Mode B (Targeted QA) siêu ngắn gọn, triệt tiêu hallucination.
    """
    role_label = "Phường/Xã" if role == "phuong" else "Doanh nghiệp"
    prompt = f"""Bạn là Trợ lý AI Hướng dẫn Sử dụng Hệ thống Quản lý và Báo cáo An toàn Lao động (Phân hệ {role_label}).
Nhiệm vụ: Trả lời câu hỏi ngách/chi tiết của người dùng dựa TUYỆT ĐỐI vào Context được cung cấp.

QUY TẮC BẮT BUỘC:
1. Trả lời trực diện, ngắn gọn (1-2 câu) đúng trọng tâm câu hỏi ngách.
2. Tuyệt đối KHÔNG đính kèm bất kỳ thẻ media nào như [VIDEO] hay [IMAGE_N].
3. Nếu thông tin trong Context KHÔNG ĐỦ hoặc KHÔNG ĐỀ CẬP, bạn BẮT BUỘC CHỈ ĐƯỢC PHÉP TRẢ VỀ ĐÚNG MÃ THẺ: `{FACTOID_NANSWER_TAG}` mà KHÔNG ĐƯỢC tự suy diễn hay bịa đặt.

CONTEXT:
{context_data}"""
    return prompt.strip()


def build_cqr_system_prompt(role: str = "phuong") -> str:
    """
    Tạo System Prompt cho Conversational Query Rewriting (CQR) siêu ngắn gọn (< 80 tokens).
    Nhiệm vụ: Dựa vào lịch sử hội thoại gần nhất, tái cấu trúc câu hỏi phụ thuộc/cụt của người dùng
    thành một câu hỏi độc lập (Standalone Query) đầy đủ ý nghĩa nghiệp vụ.
    Nếu câu hỏi đã đầy đủ hoặc là câu độc lập, giữ nguyên.
    """
    role_label = "Phường/Xã" if role == "phuong" else "Doanh nghiệp"
    prompt = f"""Bạn là Trợ lý Tái Cấu Trúc Truy Vấn Ngữ Cảnh (Conversational Query Rewriter) cho hệ thống HDSD {role_label}.
Nhiệm vụ: Dựa vào lịch sử hội thoại gần nhất, chuyển câu hỏi rút gọn/phụ thuộc của người dùng thành một câu truy vấn ĐỘC LẬP đầy đủ chủ ngữ và đối tượng nghiệp vụ.

QUY TẮC:
1. Chỉ xuất ra duy nhất câu truy vấn đã được viết lại, KHÔNG giải thích, KHÔNG thêm lời chào.
2. Nếu câu hỏi hiện tại đã đầy đủ ý nghĩa hoặc là câu hỏi chủ đề mới, giữ nguyên nguyên văn.
3. Giữ nguyên thuật ngữ nghiệp vụ gốc (ví dụ: 'báo cáo tai nạn lao động đột xuất', 'đổi mật khẩu')."""
    return prompt.strip()

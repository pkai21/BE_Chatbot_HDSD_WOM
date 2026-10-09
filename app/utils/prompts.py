from typing import List
from app.models.chunk import DocumentChunk

SYSTEM_PROMPT = """Bạn là Trợ lý AI Hướng dẫn Sử dụng Hệ thống Quản lý và Báo cáo An toàn Lao động.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TUYỆT ĐỐI vào Context được cung cấp.

YÊU CẦU ĐỊNH DẠNG ĐẦU RA:
1. Trả lời trực diện, ngắn gọn và chính xác (trong 1-3 câu) câu hỏi chi tiết/ngách của người dùng dựa trên Context.
2. Không sao chép lại toàn bộ quy trình các bước nếu người dùng không yêu cầu toàn bộ quy trình.
3. Tuyệt đối KHÔNG chèn các thẻ media [VIDEO] hoặc [IMAGE_N] vào câu trả lời Factoid/ngách để đảm bảo câu trả lời ngắn gọn, sạch sẽ.
4. Không tự ý sáng tạo các đường link URL không có trong Context."""


def format_contact_markdown(role: str = "phuong") -> str:
    """Formats official support contact info based on role."""
    if role == "dn":
        return """
- **Hotline hỗ trợ:** 028 3535 2523 - 028 3535 2524
- **Zalo hỗ trợ:** 0967 862 524
- **Thời gian làm việc:** Thứ 2 - Thứ 6 (Sáng: 08h00 – 11h00 | Chiều: 13h00 – 17h00)
""".strip()
    else:
        return """
- **Hotline hỗ trợ:** 028 3535 2524
- **Thời gian làm việc:** Thứ 2 - Thứ 6 (Sáng: 07h30 – 11h30 | Chiều: 13h00 – 17h00)
""".strip()


def format_context_for_prompt(chunks: List[DocumentChunk]) -> str:
    """Formats retrieved document chunks into concise Scoped Child Context for ultra-fast TTFT."""
    if not chunks:
        return "Không có thông tin ngữ cảnh phù hợp."

    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        breadcrumb = chunk.metadata.section_breadcrumb or f"[Phân hệ: {chunk.metadata.module or 'N/A'} > {chunk.metadata.section_title}]"
        if chunk.text_content.strip().startswith("[Phân hệ:"):
            chunk_text = chunk.text_content.strip()
        else:
            chunk_text = f"{breadcrumb}\nNội dung: {chunk.text_content.strip()}"
        context_parts.append(chunk_text)

    return "\n\n---\n\n".join(context_parts)

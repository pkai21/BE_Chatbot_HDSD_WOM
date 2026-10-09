import re
from typing import List, Optional, Tuple
from app.models.chat import ChatMessage
from app.services.qwen_service import qwen_service
from app.utils.prompt_templates import build_cqr_system_prompt
from app.core.logger import logger


class ConversationalContextResolver:
    """
    Dịch vụ Tái Cấu Trúc Truy Vấn Ngữ Cảnh Đa Lượt (Conversational Query Rewriter - CQR).
    Áp dụng mô hình Hybrid Two-Tier:
    - Tier 1: Fast-Path Slot-Filling (< 1ms) cho các câu trả lời ngắn sau khi được Bot hỏi làm rõ.
    - Tier 2: Neural LLM Rewriter cho các câu hỏi phụ thuộc phức tạp.
    """

    DEPENDENT_TRIGGERS = [
        r"^(đột xuất|dot xuat)$",
        r"^(định kỳ|dinh ky)$",
        r"^(tạo mới|tao moi)$",
        r"^(chỉnh sửa|chinh sua|sửa|sua)$",
        r"^(khôi phục|khoi phuc)$",
        r"^(xóa|xoa)$",
        r"^(cái thứ \d|cái \d|lựa chọn \d|cái trên|cái dưới)$",
        r"^(ở đâu|o dau|như thế nào|nhu the nao|bước mấy|buoc may|làm sao|lam sao)$",
        r"^(thế còn|con|sau đó|sau do|tiếp theo|tiep theo|bước \d|buoc \d)"
    ]

    SLOT_FILL_MAPPINGS = {
        "phuong": [
            {
                "keywords": ["đột xuất", "dot xuat", "báo cáo đột xuất"],
                "target_query": "Hướng dẫn quy trình báo cáo tai nạn lao động đột xuất không theo HĐLĐ"
            },
            {
                "keywords": ["định kỳ", "dinh ky", "báo cáo định kỳ"],
                "target_query": "Hướng dẫn quy trình báo cáo tai nạn lao động định kỳ cho người không có HĐLĐ"
            },
            {
                "keywords": ["tạo mới", "tao moi", "tạo tài khoản", "tao tai khoan"],
                "target_query": "Hướng dẫn tạo mới tài khoản phường xã"
            },
            {
                "keywords": ["chỉnh sửa", "chinh sua", "sửa tài khoản", "sua tai khoan"],
                "target_query": "Hướng dẫn chỉnh sửa tài khoản phường xã"
            },
            {
                "keywords": ["khôi phục", "khoi phuc", "quên mật khẩu", "quen mat khau"],
                "target_query": "Hướng dẫn khôi phục mật khẩu tài khoản phường xã"
            },
            {
                "keywords": ["xóa", "xoa", "xóa tài khoản", "xoa tai khoan"],
                "target_query": "Hướng dẫn xóa tài khoản phường xã"
            },
            {
                "keywords": ["đổi thông tin", "doi thong tin", "thông tin cá nhân", "thong tin ca nhan"],
                "target_query": "Hướng dẫn thay đổi thông tin cá nhân cán bộ phường xã"
            },
            {
                "keywords": ["đổi mật khẩu", "doi mat khau"],
                "target_query": "Hướng dẫn thay đổi mật khẩu tài khoản"
            }
        ],
        "dn": [
            {
                "keywords": ["đột xuất", "dot xuat"],
                "target_query": "Hướng dẫn khai báo tai nạn lao động đột xuất doanh nghiệp"
            },
            {
                "keywords": ["định kỳ", "dinh ky"],
                "target_query": "Hướng dẫn nộp báo cáo định kỳ tai nạn lao động doanh nghiệp"
            },
            {
                "keywords": ["đăng ký", "dang ky", "đăng ký mới"],
                "target_query": "Hướng dẫn các bước đăng ký tài khoản doanh nghiệp mới"
            },
            {
                "keywords": ["đổi mật khẩu", "doi mat khau"],
                "target_query": "Hướng dẫn thay đổi mật khẩu tài khoản doanh nghiệp"
            }
        ]
    }

    def is_dependent_query(self, query: str, history: List[ChatMessage]) -> bool:
        """
        Kiểm tra nhanh xem câu hỏi hiện tại có phụ thuộc vào ngữ cảnh trước đó hay không (< 1ms).
        """
        if not history:
            return False

        clean_query = query.strip().lower()
        words = clean_query.split()

        # Nếu câu bắt đầu bằng các cấu trúc câu độc lập hoàn chỉnh rõ ràng (>= 4 từ)
        standalone_starters = [
            "hướng dẫn", "huong dan", "quy trình", "quy trinh",
            "cách", "cach", "làm sao để", "lam sao de", "làm thế nào để", "lam the nao de",
            "cho tôi xin", "cho toi xin", "xin số", "xin so", "hotline", "tổng đài", "tong dai",
            "thời gian làm việc", "khung giờ"
        ]
        if any(clean_query.startswith(s) for s in standalone_starters) and len(words) >= 4:
            return False

        # Khớp các biểu thức regex phụ thuộc
        for pattern in self.DEPENDENT_TRIGGERS:
            if re.search(pattern, clean_query, re.IGNORECASE):
                return True

        # Câu rất ngắn (<= 3 từ) khi đang có lịch sử hội thoại
        if len(words) <= 3:
            return True

        # Chứa đại từ liên kết / phụ thuộc
        dependent_words = ["ở đâu", "bước mấy", "thế nào", "như nào", "này", "đó", "kia", "sau đó", "tiếp theo", "còn"]
        if any(dw in clean_query for dw in dependent_words) and len(words) <= 6:
            return True

        return False

    async def resolve_context(
        self,
        query: str,
        history: List[ChatMessage],
        role: str = "phuong"
    ) -> Tuple[str, str]:
        """
        Tái cấu trúc câu hỏi đa lượt.
        Trả về: Tuple[resolved_query, resolution_type]
        (resolution_type: 'PASSTHROUGH', 'FAST_SLOT_FILL', 'NEURAL_CQR')
        """
        if not self.is_dependent_query(query, history):
            return query, "PASSTHROUGH"

        clean_query = query.strip().lower()
        role_key = "dn" if role == "dn" else "phuong"

        # Tier 1: Fast-Path Context Slot-Filling (< 1ms)
        mappings = self.SLOT_FILL_MAPPINGS.get(role_key, [])
        for item in mappings:
            for kw in item["keywords"]:
                if kw in clean_query:
                    resolved = item["target_query"]
                    logger.info(f"⚡ [CQR Fast-Path] Resolved '{query}' -> '{resolved}' (matched '{kw}')")
                    return resolved, "FAST_SLOT_FILL"

        # Tier 2: Neural LLM CQR Fallback (cho các câu hỏi phức tạp hơn)
        try:
            cqr_system_prompt = build_cqr_system_prompt(role=role)
            # Chỉ lấy tối đa 4 tin nhắn gần nhất
            scoped_history = history[-4:] if len(history) > 4 else history
            
            logger.info(f"🧠 [CQR Neural Fallback] Invoking Qwen for query: '{query}' with {len(scoped_history)} history items")
            rewritten = await qwen_service.generate_text(
                system_prompt=cqr_system_prompt,
                user_query=query,
                history=scoped_history,
                temperature=0.1,
                max_tokens=60
            )

            rewritten_clean = rewritten.strip().replace('"', '').replace("'", "")
            if rewritten_clean and len(rewritten_clean) >= len(query):
                logger.info(f"🧠 [CQR Neural Result] '{query}' -> '{rewritten_clean}'")
                return rewritten_clean, "NEURAL_CQR"

        except Exception as e:
            logger.error(f"Error in Neural CQR fallback: {e}", exc_info=True)

        return query, "PASSTHROUGH"


cqr_service = ConversationalContextResolver()

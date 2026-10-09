import json
import re
import time
from typing import Optional, Dict, List, Tuple
from app.core.config import settings
from app.models.chat import QuickActionChip
from app.services.qwen_service import qwen_service
from app.core.domain_registry import domain_registry
from app.utils.prompt_templates import build_router_system_prompt
from app.services.semantic_router import semantic_router
from app.core.logger import logger

SECURITY_PATTERNS = [
    r"drop\s+table",
    r"delete\s+from",
    r"insert\s+into",
    r"update\s+set",
    r"alter\s+table",
    r"select\s+.*\s+from",
    r"exec\s*\(",
    r"benchmark\s*\(",
    r"sleep\s*\(",
    r"union\s+select",
    r"('\s*or\s*'?[0-9a-zA-Z]+)",
    r"<script.*?>.*?</script.*?>",
    r"javascript:",
    r"onload=",
    r"onerror=",
    r"system\s*\(",
    r"cmd\.exe",
    r"/bin/sh",
    r"/bin/bash"
]

STANDARD_CAPABILITY_CHIPS_PHUONG = [
    QuickActionChip(id="cap_p_login", label="🔐 Đăng nhập hệ thống", query_text="Hướng dẫn đăng nhập hệ thống cho cán bộ phường xã"),
    QuickActionChip(id="cap_p_user", label="👤 Đổi thông tin cá nhân", query_text="Cách cập nhật thông tin cá nhân của cán bộ"),
    QuickActionChip(id="cap_p_pwd", label="🔑 Đổi mật khẩu", query_text="Làm sao để đổi mật khẩu tài khoản đang đăng nhập?"),
    QuickActionChip(id="cap_p_manage", label="👥 Quản lý tài khoản tuyến dưới", query_text="Giải đáp chức năng tài khoản phường xã gồm những thao tác nào?"),
    QuickActionChip(id="cap_p_dotxuat", label="🚨 Báo cáo TNLĐ đột xuất", query_text="Quy trình phường/xã khai báo vụ tai nạn lao động ngay khi mới xảy ra trên địa bàn"),
    QuickActionChip(id="cap_p_dinhky", label="📊 Báo cáo TNLĐ định kỳ", query_text="Hướng dẫn báo cáo tai nạn lao động định kỳ cho người không có HĐLĐ"),
    QuickActionChip(id="cap_p_contact", label="📞 Hotline hỗ trợ kỹ thuật", query_text="Cho tôi thông tin hotline hỗ trợ kỹ thuật")
]

STANDARD_CAPABILITY_CHIPS_DN = [
    QuickActionChip(id="cap_reg", label="📝 Hướng dẫn đăng ký mới", query_text="Hướng dẫn các bước đăng ký tài khoản doanh nghiệp mới"),
    QuickActionChip(id="cap_login", label="🔑 Hướng dẫn đổi mật khẩu", query_text="Làm thế nào để đổi mật khẩu tài khoản?"),
    QuickActionChip(id="cap_tnld", label="⚠️ Hướng dẫn nộp báo cáo TNLĐ", query_text="Hướng dẫn nộp báo cáo định kỳ tai nạn lao động"),
    QuickActionChip(id="cap_atvsld", label="🛡️ Hướng dẫn nộp báo cáo ATVSLĐ", query_text="Hướng dẫn nộp báo cáo định kỳ An toàn vệ sinh lao động"),
    QuickActionChip(id="cap_stat", label="📊 Tra cứu số liệu thống kê", query_text="Làm thế nào để xem số liệu thống kê tai nạn lao động?")
]

STANDARD_CAPABILITY_CHIPS = STANDARD_CAPABILITY_CHIPS_PHUONG


from app.models.intent import IntentResult


class IntentService:
    """
    Zero-Keyword Single-Pass Dynamic Multi-Aspect Intent & Strategy Router sử dụng Qwen-Flash.
    Định tuyến đồng thời Macro Intent, Target Business Module, Confidence Score và Strategy
    theo Dynamic Scoped Prompt Templates tương ứng với từng Role.
    """

    def __init__(self):
        self._sync_client = None

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(
                api_key=settings.effective_api_key,
                base_url=settings.QWEN_BASE_URL,
                timeout=10.0
            )
        return self._sync_client

    def _parse_router_response(self, raw_content: str, role: str = "phuong") -> Tuple[str, Optional[str], float, str, str]:
        """Trích xuất và chuẩn hóa macro_intent, target_module, confidence, strategy và lý do từ phản hồi JSON."""
        valid_modules = domain_registry.get_valid_modules(role)
        try:
            cleaned_content = raw_content.strip()
            # Extract JSON substring if surrounded by markdown fences or additional text
            json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
            if json_match:
                cleaned_content = json_match.group(0)

            data = json.loads(cleaned_content)
            raw_intent = data.get("macro_intent", data.get("intent", "KNOWLEDGE_QUERY")).upper()
            target_module = data.get("target_module")
            
            # Fuzzy match target_module if needed
            if target_module:
                found_match = False
                for vm in valid_modules:
                    if target_module.strip().lower() == vm.strip().lower() or target_module in vm or vm in target_module:
                        target_module = vm
                        found_match = True
                        break
                if not found_match and target_module not in valid_modules:
                    target_module = None

            # Parse confidence
            raw_conf = data.get("confidence", 1.0)
            try:
                confidence = float(raw_conf)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.95 if target_module else 0.8

            raw_strategy = data.get("strategy", "targeted_qa").lower()
            if "procedural" in raw_strategy and target_module is not None:
                strategy = "procedural_extractive"
            else:
                strategy = "targeted_qa"

            reason = data.get("reason", "Policy-based classification")
            return raw_intent, target_module, confidence, strategy, reason
        except Exception as e:
            logger.warning(f"Error parsing router JSON '{raw_content[:100]}...': {e}")
            return "KNOWLEDGE_QUERY", None, 0.5, "targeted_qa", "JSON parse fallback"

    def _build_intent_result(
        self,
        raw_intent: str,
        target_module: Optional[str],
        confidence: float,
        strategy: str,
        reason: str,
        role: str = "phuong",
        query: str = ""
    ) -> IntentResult:
        """Đóng gói IntentResult tương ứng với nhãn phân loại đa chiều."""
        is_phuong = (role == "phuong")
        if raw_intent == "CHITCHAT_CAPABILITY":
            # Guardrail: Chỉ chấp nhận Capability nếu câu thực sự hỏi về năng lực của Bot hoặc tổng quan chức năng bot
            bot_cap_patterns = [
                r"\b(bạn|bot|em|mày|ai)\b.*(làm|hỗ trợ|giúp|biết|chức năng|năng lực|giới thiệu)",
                r"\b(bạn là ai|bot là ai|làm được gì|giúp được gì|hỗ trợ được gì|chức năng gì|có thể làm gì|hướng dẫn tôi những gì)\b",
                r"\b(tổng quan hệ thống|hệ thống có những chức năng gì|hệ thống gồm những gì|các chức năng chính)\b"
            ]
            norm_q = query.strip().lower()
            is_real_bot_cap = any(re.search(p, norm_q) for p in bot_cap_patterns)
            if not is_real_bot_cap and len(norm_q) > 0:
                # Chuyển về knowledge_query để đi tiếp vào RAG pipeline và kích hoạt Factoid Rejection đúng chuẩn
                return IntentResult(
                    intent="knowledge_query",
                    direct_answer=None,
                    confidence_score=0.5,
                    matched_exemplar=f"capability_guardrail_routed_to_knowledge: {reason}",
                    all_scores={"knowledge_query": 0.5},
                    target_module=target_module,
                    strategy="targeted_qa"
                )

            if is_phuong:
                cap_answer = (
                    "Tôi là **Trợ lý AI Hướng dẫn Sử dụng Hệ thống Quản lý và Báo cáo An toàn Lao động** (Phân hệ Phường/Xã). Tôi có thể hỗ trợ bạn các nghiệp vụ chính sau:\n\n"
                    "1. 🔐 **Đăng nhập hệ thống** cho cán bộ Phường/Xã\n"
                    "2. 👤 **Cập nhật thông tin cá nhân** (Họ tên, Chức danh, Email, SĐT...)\n"
                    "3. 🔑 **Thay đổi mật khẩu** tài khoản đang đăng nhập\n"
                    "4. 👥 **Quản lý tài khoản tuyến dưới** (Tạo mới, Chỉnh sửa, Khôi phục mật khẩu, Xóa tài khoản)\n"
                    "5. 📊 **Báo cáo định kỳ Tai nạn lao động** (Không theo HĐLĐ)\n"
                    "6. 🚨 **Báo cáo đột xuất Tai nạn lao động** (Không theo HĐLĐ)\n"
                    "7. 📞 **Tra cứu thông tin liên hệ & Hotline hỗ trợ kỹ thuật**\n\n"
                    "Bạn cần hướng dẫn thực hiện chức năng nào ở trên?"
                )
                chips = STANDARD_CAPABILITY_CHIPS_PHUONG
            else:
                cap_answer = (
                    "Tôi là **Trợ lý AI Hướng dẫn Sử dụng Hệ thống Quản lý và Báo cáo An toàn Lao động** (Phân hệ Doanh nghiệp). Tôi có thể hỗ trợ bạn các nghiệp vụ chính sau:\n\n"
                    "1. 📝 **Đăng ký tài khoản mới** (Dành cho Doanh nghiệp bằng Mã số thuế)\n"
                    "2. 🔐 **Đăng nhập hệ thống** & Phục hồi/Thay đổi mật khẩu\n"
                    "3. 🏢 **Cập nhật thông tin Doanh nghiệp** (Địa chỉ, Người đại diện, Điện thoại)\n"
                    "4. ⚠️ **Lập & Gửi Báo cáo Định kỳ Tai nạn lao động** (6 tháng / Cả năm)\n"
                    "5. 🛡️ **Lập & Gửi Báo cáo Định kỳ An toàn vệ sinh lao động** (Hàng năm)\n"
                    "6. 📊 **Tra cứu & Xem Thống kê** số liệu tai nạn lao động đã nộp\n\n"
                    "Bạn cần hướng dẫn thực hiện chức năng nào ở trên?"
                )
                chips = STANDARD_CAPABILITY_CHIPS_DN

            return IntentResult(
                intent="chitchat_capability",
                direct_answer=cap_answer,
                quick_action_chips=chips,
                confidence_score=confidence,
                matched_exemplar=reason,
                all_scores={"chitchat_capability": confidence},
                target_module=None,
                strategy="targeted_qa"
            )

        elif raw_intent == "CHITCHAT_GREETING":
            # Guardrail: Chỉ chấp nhận Greeting nếu câu thực sự chứa từ ngữ chào hỏi
            greeting_patterns = [r"\b(chào|xin chào|hello|hi|hey|alo|good morning|chao|kính chào)\b"]
            norm_q = query.strip().lower()
            is_real_greeting = any(re.search(p, norm_q) for p in greeting_patterns)
            if not is_real_greeting and len(norm_q) > 0:
                # Chuyển về knowledge_query để đi tiếp vào RAG pipeline và kích hoạt Factoid Rejection đúng chuẩn
                return IntentResult(
                    intent="knowledge_query",
                    direct_answer=None,
                    confidence_score=0.5,
                    matched_exemplar=f"greeting_guardrail_routed_to_knowledge: {reason}",
                    all_scores={"knowledge_query": 0.5},
                    target_module=target_module,
                    strategy="targeted_qa"
                )

            chips = STANDARD_CAPABILITY_CHIPS_PHUONG[:4] if is_phuong else STANDARD_CAPABILITY_CHIPS_DN[:4]
            return IntentResult(
                intent="chitchat_greeting",
                direct_answer="Xin chào! Tôi là Trợ lý AI Hướng dẫn Sử dụng Hệ thống. Tôi có thể hỗ trợ gì cho bạn trong việc thao tác và tra cứu tài liệu hôm nay?",
                quick_action_chips=chips,
                confidence_score=confidence,
                matched_exemplar=reason,
                all_scores={"chitchat_greeting": confidence},
                target_module=None,
                strategy="targeted_qa"
            )

        elif raw_intent == "CHITCHAT_THANKS":
            return IntentResult(
                intent="chitchat_thanks",
                direct_answer="Rất vui được hỗ trợ bạn! Nếu cần hướng dẫn thêm bất kỳ chức năng nào khác, bạn cứ nhắn cho tôi nhé.",
                confidence_score=confidence,
                matched_exemplar=reason,
                all_scores={"chitchat_thanks": confidence},
                target_module=None,
                strategy="targeted_qa"
            )

        elif raw_intent == "CONTACT_ESCALATION":
            return IntentResult(
                intent="contact_escalation",
                direct_answer=None,
                confidence_score=confidence,
                matched_exemplar=reason,
                all_scores={"contact_escalation": confidence},
                target_module=None,
                strategy="targeted_qa"
            )

        else:
            # Default / KNOWLEDGE_QUERY
            norm_q = query.strip().lower()
            
            # Procedural-First Policy:
            # 1. Action/Location patterns: Các câu hỏi vị trí truy cập hoặc cách bắt đầu thao tác của 1 module
            # (ví dụ: 'đổi mật khẩu ở đâu', 'sửa thông tin tài khoản cấp dưới ở đâu', 'đăng ký ở chỗ nào')
            action_location_patterns = [
                r"\b(ở đâu|chỗ nào|vào đâu|bấm vào đâu|nhấn vào đâu|truy cập ở đâu|tìm ở đâu)\b",
                r"\b(làm sao|làm thế nào|cách nào|hướng dẫn|quy trình|các bước|thao tác thế nào)\b"
            ]
            is_action_location = any(re.search(p, norm_q) for p in action_location_patterns)

            # 2. Targeted Child-Slot patterns: CHỈ KHI hỏi về chi tiết trường dữ liệu con hoặc ràng buộc/thông số
            targeted_child_slot_patterns = [
                r"\b(thông tin gì|điền gì|nhập gì|các trường|trường nào|group nào|mục nào bắt buộc|trường bắt buộc)\b",
                r"\b(đơn vị là gì|triệu đồng hay|đồng hay|bao nhiêu ngày|hạn chót|thời hạn nộp|hạn nộp)\b",
                r"\b(mấy giờ|từ mấy giờ|khung giờ|dung lượng tối đa|định dạng file|cho phép file gì)\b"
            ]
            is_targeted_child_slot = any(re.search(p, norm_q) for p in targeted_child_slot_patterns)

            if target_module is not None:
                if is_action_location and not is_targeted_child_slot:
                    strategy = "procedural_extractive"
                elif is_targeted_child_slot:
                    strategy = "targeted_qa"
            else:
                strategy = "targeted_qa"

            return IntentResult(
                intent="knowledge_query",
                direct_answer=None,
                confidence_score=confidence,
                matched_exemplar=reason,
                all_scores={"knowledge_query": confidence},
                target_module=target_module,
                strategy=strategy
            )

    def _check_deterministic_fast_paths(self, normalized: str, role: str = "phuong") -> Optional[IntentResult]:
        """Tầng 0.5: Fast-path định tuyến siêu tốc (< 0.1ms) cho các trường hợp đặc thù."""
        # 1. Chức năng tài khoản Phường/Xã
        if role == "phuong" and any(k in normalized for k in [
            "chức năng tài khoản phường",
            "chức năng tài khoản",
            "tổng quan chức năng tài khoản",
            "giải đáp chức năng tài khoản",
            "thao tác tài khoản phường",
            "quản lý tài khoản tuyến dưới",
            "chức năng phân hệ phường"
        ]):
            return IntentResult(
                intent="knowledge_query",
                direct_answer=None,
                confidence_score=1.0,
                matched_exemplar="deterministic_account_overview_match",
                target_module="TỔNG QUAN CHỨC NĂNG TÀI KHOẢN PHƯỜNG/XÃ",
                strategy="procedural_extractive"
            )

        # 2. Hotline / Liên hệ hỗ trợ / Khung giờ tổng đài
        if any(k in normalized for k in [
            "hotline", "tổng đài", "số điện thoại hỗ trợ", "liên hệ hỗ trợ",
            "khung giờ làm việc", "giờ làm việc của tổng đài", "buổi sáng tổng đài",
            "buổi chiều tổng đài", "thời gian làm việc của tổng đài", "sáng tổng đài",
            "chiều tổng đài", "thời gian tổng đài", "giờ tổng đài", "giờ làm việc"
        ]):
            is_specific_qa = any(q_word in normalized for q_word in [
                "mấy giờ", "từ mấy giờ", "bao giờ", "thứ mấy", "khi nào", "sáng từ", "chiều từ"
            ])
            return IntentResult(
                intent="knowledge_query",
                direct_answer=None,
                confidence_score=1.0,
                matched_exemplar="deterministic_contact_support_match",
                target_module="LIÊN HỆ HỖ TRỢ",
                strategy="targeted_qa" if is_specific_qa else "procedural_extractive"
            )

        return None

    def _check_ambiguity(self, normalized: str, role: str = "phuong") -> Optional[IntentResult]:
        """
        Tầng 0.6: Phát hiện câu hỏi mơ hồ/thiếu thực thể và kích hoạt làm rõ (Make-Clear).
        Áp dụng mô hình Multi-tier:
        1. Fast-path: Regex Pre-stripping loại bỏ từ đệm giao tiếp + exact match danh sách lõi (< 0.05ms).
        2. Deep Semantic Match: DenseSemanticRouter (AITeamVN Embedding Cosine Similarity).
        """
        clean = normalized.strip()

        # Tầng 0.55: Pre-stripping tiền tố giao tiếp tự nhiên (hỗ trợ bóc tách lồng nhau đa tầng)
        PREFIX_PATTERN = r"^(cho\s+(tôi|em|mình|ad)(\s+(xin|xem|hỏi))?|(tôi|em|mình|ad)?\s*muốn\s+(xem|biết|hỏi|tìm)|làm\s+ơn\s+(cho|hướng\s+dẫn)?|xin|hướng\s+dẫn|quy\s+trình|cách|tìm|về|xem)\s+"
        prev_clean = ""
        core_clean = clean
        while prev_clean != core_clean:
            prev_clean = core_clean
            core_clean = re.sub(PREFIX_PATTERN, "", core_clean).strip()

        # 1. Mơ hồ về Báo cáo Tai nạn lao động (Phường/Xã hoặc Doanh nghiệp)
        tnld_general = [
            "báo cáo tai nạn", "bao cao tai nan",
            "báo cáo tai nạn lao động", "bao cao tai nan lao dong",
            "khai báo tai nạn", "khai bao tai nan",
            "báo cáo tnld", "bao cao tnlđ",
            "tai nạn lao động", "tai nan lao dong", "tnlđ", "tnld"
        ]
        has_general_tnld = any(
            clean == k or core_clean == k or clean == f"hướng dẫn {k}" or clean == f"quy trình {k}" or clean == f"cách {k}"
            for k in tnld_general
        )
        is_specific_type = any(
            t in clean for t in [
                "định kỳ", "dinh ky", "đột xuất", "dot xuat", "sơ lược", "nạn nhân",
                "nghề nghiệp", "phường/xã", "thông tin phường", "kích thước", "file",
                "đính kèm", "dung lượng", "bao nhiêu", "ở đâu"
            ]
        )

        if has_general_tnld and not is_specific_type:
            if role == "phuong":
                return IntentResult(
                    intent="clarification_needed",
                    direct_answer="Bạn đang muốn tìm hiểu về quy trình **Báo cáo tai nạn lao động định kỳ** hay **Báo cáo tai nạn lao động đột xuất**?",
                    quick_action_chips=[
                        QuickActionChip(id="clarify_dotxuat", label="🚨 Báo cáo TNLĐ đột xuất", query_text="Hướng dẫn quy trình báo cáo tai nạn lao động đột xuất không theo HĐLĐ"),
                        QuickActionChip(id="clarify_dinhky", label="📊 Báo cáo TNLĐ định kỳ", query_text="Hướng dẫn quy trình báo cáo tai nạn lao động định kỳ cho người không có HĐLĐ")
                    ],
                    confidence_score=1.0,
                    matched_exemplar="ambiguity_clarification_tnld_phuong",
                    all_scores={"clarification_needed": 1.0},
                    target_module=None,
                    strategy="targeted_qa"
                )
            else:
                return IntentResult(
                    intent="clarification_needed",
                    direct_answer="Bạn đang muốn tìm hiểu về **Báo cáo định kỳ tai nạn lao động** hay **Khai báo tai nạn lao động đột xuất**?",
                    quick_action_chips=[
                        QuickActionChip(id="clarify_dn_dotxuat", label="🚨 Khai báo TNLĐ đột xuất", query_text="Hướng dẫn khai báo tai nạn lao động đột xuất doanh nghiệp"),
                        QuickActionChip(id="clarify_dn_dinhky", label="📊 Báo cáo định kỳ TNLĐ", query_text="Hướng dẫn nộp báo cáo định kỳ tai nạn lao động doanh nghiệp")
                    ],
                    confidence_score=1.0,
                    matched_exemplar="ambiguity_clarification_tnld_dn",
                    all_scores={"clarification_needed": 1.0},
                    target_module=None,
                    strategy="targeted_qa"
                )

        # 2. Mơ hồ về Tài khoản Phường/Xã
        account_general = [
            "tài khoản", "tai khoan",
            "tài khoản phường", "tai khoan phuong",
            "tài khoản phường xã", "tai khoan phuong xa",
            "thao tác tài khoản", "thao tac tai khoan",
            "quản lý tài khoản", "quan ly tai khoan",
            "thông tin tài khoản", "thong tin tai khoan"
        ]
        has_general_acc = any(
            clean == k or core_clean == k or clean == f"hướng dẫn {k}" or clean == f"thao tác {k}"
            for k in account_general
        )
        is_specific_acc = any(
            t in clean for t in [
                "tạo", "tao", "mới", "moi", "sửa", "sua", "chỉnh", "chinh",
                "khôi phục", "khoi phuc", "quên", "quen", "xóa", "xoa",
                "tổng quan", "tong quan", "chức năng", "đổi mật khẩu", "doi mat khau",
                "cá nhân", "ca nhan", "cán bộ", "can bo"
            ]
        )

        if role == "phuong" and has_general_acc and not is_specific_acc:
            return IntentResult(
                intent="clarification_needed",
                direct_answer="Về phân hệ **Tài khoản Phường/xã**, bạn đang cần hướng dẫn thao tác nào dưới đây?",
                quick_action_chips=[
                    QuickActionChip(id="clarify_p_create", label="➕ Tạo mới tài khoản", query_text="Hướng dẫn tạo mới tài khoản phường xã"),
                    QuickActionChip(id="clarify_p_edit", label="✏️ Chỉnh sửa tài khoản", query_text="Hướng dẫn chỉnh sửa tài khoản phường xã"),
                    QuickActionChip(id="clarify_p_reset", label="🔓 Khôi phục mật khẩu", query_text="Hướng dẫn khôi phục mật khẩu tài khoản phường xã"),
                    QuickActionChip(id="clarify_p_del", label="🗑️ Xóa tài khoản", query_text="Hướng dẫn xóa tài khoản phường xã")
                ],
                confidence_score=1.0,
                matched_exemplar="ambiguity_clarification_account_phuong",
                all_scores={"clarification_needed": 1.0},
                target_module=None,
                strategy="targeted_qa"
            )

        # 3. Tầng 0.6: Dense Semantic Router (Deep Embedding Similarity)
        try:
            semantic_result = semantic_router.match_ambiguity(clean, role=role)
            if semantic_result is not None:
                return semantic_result
        except Exception as e:
            logger.error(f"Error checking semantic_router in _check_ambiguity: {e}")

        return None

    async def classify_intent_async(self, query: str, role: str = "phuong") -> IntentResult:
        """Phân loại ý định và chiến lược bất đồng bộ (Native Async Router)."""
        normalized = query.strip().lower()

        # 1. Tầng 0: Kiểm tra An toàn & Mã độc (Security Guardrails Fast-path < 0.1ms)
        for pattern in SECURITY_PATTERNS:
            if re.search(pattern, normalized):
                return IntentResult(
                    intent="security_violation",
                    direct_answer="⚠️ **Từ chối can thiệp kỹ thuật:** Tôi là Trợ lý AI Hướng dẫn Sử dụng (HDSD). Theo quy định an toàn hệ thống, tôi không có quyền can thiệp, chạy lệnh SQL hay thay đổi trực tiếp cơ sở dữ liệu. Vui lòng thực hiện thao tác theo quy trình hoặc liên hệ Ban Quản trị Sở Lao động - TB&XH để được phê duyệt theo thẩm quyền.",
                    confidence_score=1.0,
                    matched_exemplar="security_guardrail_triggered",
                    target_module=None,
                    strategy="targeted_qa"
                )

        # 2. Tầng 0.5: Fast-path định tuyến mẫu nghiệp vụ đặc thù
        fast_result = self._check_deterministic_fast_paths(normalized, role=role)
        if fast_result is not None:
            logger.info(f"Single-Pass Router (FastPath): Intent={fast_result.intent} | Target={fast_result.target_module} | Strategy={fast_result.strategy}")
            return fast_result

        # 2.5. Tầng 0.6: Ambiguity Detector & Clarification (Hỏi lại khi thiếu ngữ cảnh)
        ambiguous_result = self._check_ambiguity(normalized, role=role)
        if ambiguous_result is not None:
            logger.info(f"Single-Pass Router (Ambiguity Clarification): Intent={ambiguous_result.intent} | Options={len(ambiguous_result.quick_action_chips or [])}")
            return ambiguous_result

        # 3. Tầng 1: Single-Pass Multi-Aspect AI Router (Qwen-Flash Fast JSON Mode)
        system_prompt = build_router_system_prompt(role=role)
        try:
            client = qwen_service._get_client()
            response = await client.chat.completions.create(
                model=settings.QWEN_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False}
            )

            raw_content = response.choices[0].message.content.strip()
            raw_intent, target_module, confidence, strategy, reason = self._parse_router_response(raw_content, role=role)
            logger.info(f"Single-Pass Router (Async): Intent={raw_intent} | Target={target_module} | Confidence={confidence} | Strategy={strategy} | Reason='{reason}'")
            return self._build_intent_result(raw_intent, target_module, confidence, strategy, reason, role=role, query=query)

        except Exception as e:
            logger.error(f"Lỗi Policy Router Qwen (Async): {e}. Kích hoạt fallback an toàn về knowledge_query.")
            return IntentResult(
                intent="knowledge_query",
                direct_answer=None,
                confidence_score=0.5,
                matched_exemplar="async_fallback",
                target_module=None,
                strategy="targeted_qa"
            )

    def classify_intent(self, query: str, role: str = "phuong") -> IntentResult:
        """Phân loại ý định và chiến lược đồng bộ (Hỗ trợ CLI và Test Suite)."""
        normalized = query.strip().lower()

        # 1. Tầng 0: Kiểm tra An toàn & Mã độc (Security Guardrails Fast-path < 0.1ms)
        for pattern in SECURITY_PATTERNS:
            if re.search(pattern, normalized):
                return IntentResult(
                    intent="security_violation",
                    direct_answer="⚠️ **Từ chối can thiệp kỹ thuật:** Tôi là Trợ lý AI Hướng dẫn Sử dụng (HDSD). Theo quy định an toàn hệ thống, tôi không có quyền can thiệp, chạy lệnh SQL hay thay đổi trực tiếp cơ sở dữ liệu. Vui lòng thực hiện thao tác theo quy trình hoặc liên hệ Ban Quản trị Sở Lao động - TB&XH để được phê duyệt theo thẩm quyền.",
                    confidence_score=1.0,
                    matched_exemplar="security_guardrail_triggered",
                    target_module=None,
                    strategy="targeted_qa"
                )

        # 2. Tầng 0.5: Fast-path định tuyến mẫu nghiệp vụ đặc thù
        fast_result = self._check_deterministic_fast_paths(normalized, role=role)
        if fast_result is not None:
            logger.info(f"Single-Pass Router (FastPath Sync): Intent={fast_result.intent} | Target={fast_result.target_module} | Strategy={fast_result.strategy}")
            return fast_result

        # 2.5. Tầng 0.6: Ambiguity Detector & Clarification (Hỏi lại khi thiếu ngữ cảnh)
        ambiguous_result = self._check_ambiguity(normalized, role=role)
        if ambiguous_result is not None:
            logger.info(f"Single-Pass Router (Ambiguity Clarification Sync): Intent={ambiguous_result.intent} | Options={len(ambiguous_result.quick_action_chips or [])}")
            return ambiguous_result

        # 3. Tầng 1: Gọi đồng bộ qua OpenAI client
        system_prompt = build_router_system_prompt(role=role)
        try:
            sync_client = self._get_sync_client()
            response = sync_client.chat.completions.create(
                model=settings.QWEN_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False}
            )

            raw_content = response.choices[0].message.content.strip()
            raw_intent, target_module, confidence, strategy, reason = self._parse_router_response(raw_content, role=role)
            logger.info(f"Single-Pass Router (Sync): Intent={raw_intent} | Target={target_module} | Confidence={confidence} | Strategy={strategy} | Reason='{reason}'")
            return self._build_intent_result(raw_intent, target_module, confidence, strategy, reason, role=role, query=query)

        except Exception as e:
            logger.error(f"Lỗi Policy Router Qwen (Sync): {e}. Kích hoạt fallback an toàn về knowledge_query.")
            return IntentResult(
                intent="knowledge_query",
                direct_answer=None,
                confidence_score=0.5,
                matched_exemplar="sync_fallback",
                target_module=None,
                strategy="targeted_qa"
            )


intent_service = IntentService()

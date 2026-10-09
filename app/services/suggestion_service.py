from typing import List, Optional, Dict
from app.models.chat import QuickActionChip
from app.models.chunk import DocumentChunk
from app.core.logger import logger


class SuggestionService:
    """
    Role-Aware Heading Level 1 Follow-up Question Suggestion Service.
    Strictly suggests main procedure workflows (Heading Level 1)
    and strictly enforces Role-Based Isolation between Doanh nghiệp and Phường/Xã.
    """

    ROLE_HEADING_CHIPS: Dict[str, Dict[str, QuickActionChip]] = {
        "dn": {
            "ĐĂNG KÝ": QuickActionChip(
                id="sug_h1_dn_reg",
                label="📝 Đăng ký tài khoản",
                query_text="Hướng dẫn các bước đăng ký tài khoản doanh nghiệp mới"
            ),
            "ĐĂNG NHẬP": QuickActionChip(
                id="sug_h1_dn_login",
                label="🔐 Đăng nhập hệ thống",
                query_text="Hướng dẫn các bước đăng nhập hệ thống"
            ),
            "THAY ĐỔI MẬT KHẨU": QuickActionChip(
                id="sug_h1_dn_pwd",
                label="🔑 Đổi mật khẩu",
                query_text="Hướng dẫn thay đổi mật khẩu tài khoản doanh nghiệp"
            ),
            "THAY ĐỔI THÔNG TIN DOANH NGHIỆP": QuickActionChip(
                id="sug_h1_dn_info",
                label="🏢 Đổi thông tin DN",
                query_text="Hướng dẫn thay đổi thông tin doanh nghiệp"
            ),
            "BÁO CÁO ĐỊNH KỲ - 1. Tai nạn lao động": QuickActionChip(
                id="sug_h1_dn_tnld",
                label="⚠️ Báo cáo TNLĐ",
                query_text="Hướng dẫn nộp báo cáo định kỳ tai nạn lao động"
            ),
            "BÁO CÁO ĐỊNH KỲ - 2. An toàn vệ sinh lao động": QuickActionChip(
                id="sug_h1_dn_atvsld",
                label="🛡️ Báo cáo ATVSLĐ",
                query_text="Hướng dẫn nộp báo cáo định kỳ An toàn vệ sinh lao động"
            ),
            "THỐNG KÊ": QuickActionChip(
                id="sug_h1_dn_stat",
                label="📊 Xem số liệu thống kê",
                query_text="Làm thế nào để xem biểu đồ và số liệu thống kê?"
            ),
            "LIÊN HỆ HỖ TRỢ": QuickActionChip(
                id="sug_h1_dn_help",
                label="📞 Hotline & Zalo hỗ trợ",
                query_text="Cho tôi thông tin hotline và zalo hỗ trợ kỹ thuật"
            )
        },
        "phuong": {
            "ĐĂNG NHẬP": QuickActionChip(
                id="sug_h1_p_login",
                label="🔐 Đăng nhập hệ thống",
                query_text="Hướng dẫn đăng nhập hệ thống cho cán bộ phường xã"
            ),
            "THAY ĐỔI THÔNG TIN CÁ NHÂN": QuickActionChip(
                id="sug_h1_p_user",
                label="👤 Đổi thông tin cán bộ",
                query_text="Hướng dẫn thay đổi thông tin cá nhân cán bộ"
            ),
            "THAY ĐỔI MẬT KHẨU": QuickActionChip(
                id="sug_h1_p_pwd",
                label="🔑 Đổi mật khẩu",
                query_text="Làm sao để đổi mật khẩu tài khoản đang đăng nhập?"
            ),
            "TẠO MỚI TÀI KHOẢN PHƯỜNG/XÃ": QuickActionChip(
                id="sug_h1_p_create",
                label="➕ Tạo tài khoản Phường/Xã",
                query_text="Hướng dẫn tạo mới tài khoản phường xã"
            ),
            "CHỈNH SỬA TÀI KHOẢN PHƯỜNG/XÃ": QuickActionChip(
                id="sug_h1_p_edit",
                label="✏️ Sửa tài khoản tuyến dưới",
                query_text="Cách sửa thông tin tài khoản tuyến dưới"
            ),
            "KHÔI PHỤC MẬT KHẨU TÀI KHOẢN PHƯỜNG/XÃ": QuickActionChip(
                id="sug_h1_p_reset",
                label="🔓 Khôi phục mật khẩu",
                query_text="Cách khôi phục mật khẩu tài khoản phường xã"
            ),
            "XÓA TÀI KHOẢN PHƯỜNG/XÃ": QuickActionChip(
                id="sug_h1_p_del",
                label="🗑️ Xóa tài khoản tuyến dưới",
                query_text="Làm sao để xóa tài khoản phường xã?"
            ),
            "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỊNH KỲ KHÔNG THEO HĐLĐ": QuickActionChip(
                id="sug_h1_p_dinhky",
                label="📊 Báo cáo TNLĐ định kỳ",
                query_text="Hướng dẫn báo cáo tai nạn lao động định kỳ cho người không có HĐLĐ"
            ),
            "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỘT XUẤT KHÔNG THEO HĐLĐ": QuickActionChip(
                id="sug_h1_p_dotxuat",
                label="🚨 Báo cáo TNLĐ đột xuất",
                query_text="Quy trình phường/xã khai báo vụ tai nạn lao động ngay khi mới xảy ra trên địa bàn"
            ),
            "LIÊN HỆ HỖ TRỢ": QuickActionChip(
                id="sug_h1_p_help",
                label="📞 Hotline hỗ trợ",
                query_text="Cho tôi thông tin hotline hỗ trợ kỹ thuật"
            )
        }
    }

    def get_suggested_chips(
        self,
        query: str,
        role: str = "dn",
        target_module: Optional[str] = None,
        primary_chunk: Optional[DocumentChunk] = None,
        limit: int = 3
    ) -> List[QuickActionChip]:
        """
        Determines suggested follow-up question action chips strictly based on Heading Level 1s
        belonging to the given role, excluding the active heading and avoiding cross-role pollution.
        """
        active_role = "phuong" if str(role).lower() == "phuong" else "dn"
        role_chips_dict = self.ROLE_HEADING_CHIPS[active_role]

        # Determine active module
        active_module = target_module
        if not active_module and primary_chunk and primary_chunk.metadata:
            active_module = primary_chunk.metadata.module or primary_chunk.metadata.section_title

        matched_active_key: Optional[str] = None
        if active_module:
            for h_key in role_chips_dict:
                if h_key == active_module or h_key in active_module or active_module in h_key:
                    matched_active_key = h_key
                    break

        norm_query = query.lower().strip()
        filtered_chips: List[QuickActionChip] = []

        # 1. Primary pass: Add chips from same role that are not active heading and not query-redundant
        for h_key, chip in role_chips_dict.items():
            if matched_active_key and h_key == matched_active_key:
                continue

            chip_q = chip.query_text.lower()
            common_words = set(w for w in chip_q.split() if len(w) > 3).intersection(
                set(w for w in norm_query.split() if len(w) > 3)
            )
            is_same_intent = len(common_words) >= 4 or chip_q in norm_query or norm_query in chip_q
            if is_same_intent:
                continue

            filtered_chips.append(chip)
            if len(filtered_chips) >= limit:
                break

        # 2. Fallback pass: Pad with remaining Heading 1s of the SAME role if fewer than limit
        if len(filtered_chips) < limit:
            for h_key, chip in role_chips_dict.items():
                if not any(c.id == chip.id for c in filtered_chips):
                    if not (matched_active_key and h_key == matched_active_key):
                        filtered_chips.append(chip)
                if len(filtered_chips) >= limit:
                    break

        return filtered_chips[:limit]


suggestion_service = SuggestionService()

from app.models.chat import ContactSupportInfo

DEFAULT_SUPPORT_INFO = ContactSupportInfo(
    title="THÔNG TIN LIÊN HỆ HỖ TRỢ KỸ THUẬT",
    working_hours="Thứ 2 - Thứ 6\n- Sáng: 08h00 – 11h00\n- Chiều: 13h00 – 17h00",
    hotlines=["028 3535 2523", "028 3535 2524"],
    zalo="0967 862 524"
)

def format_contact_markdown(info: ContactSupportInfo = DEFAULT_SUPPORT_INFO) -> str:
    hotline_str = " - ".join(info.hotlines)
    return f"""
> 📞 **{info.title}**
>
> **Thời gian làm việc:** {info.working_hours}
> - **Hotline hỗ trợ:** {hotline_str}
> - **Zalo hỗ trợ:** {info.zalo}
"""

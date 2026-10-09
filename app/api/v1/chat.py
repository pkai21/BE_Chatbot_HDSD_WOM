import asyncio
import json
import os
import re
import uuid
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from app.core.config import settings
from app.models.chat import ChatRequest, ChatResponse, QuickActionChip
from app.models.common import ApiResponse
from app.services.rag_service import rag_service
from app.services.chat_history_service import chat_history_service
from app.core.logger import logger

# Import Autonomous Warehouse Agent and Scope Guard
try:
    from IPGov_Chatbot.modules.mod03_router.warehouse_langgraph_agent import warehouse_agent
except ImportError as err:
    logger.warning(f"Could not import warehouse_agent from IPGov_Chatbot: {err}")
    warehouse_agent = None

try:
    from IPGov_Chatbot.modules.mod02_guardrails.scope_prechecker import ScopePrechecker
except ImportError as err:
    logger.warning(f"Could not import ScopePrechecker: {err}")
    ScopePrechecker = None

router = APIRouter(prefix="/chat", tags=["Chat"])

DWH_ROLES = {
    "dwh", "warehouse", "lanhdao", "so", "so_noivu", "tinh", 
    "chuyenvien", "leader", "lamdong_leader", "lamdong_phong_kinhte", "phong"
}

DWH_COLLECTIONS = {"dwh", "warehouse", "vna_wom_dev"}

HDSD_PROCEDURAL_PATTERNS = [
    r"\b(hướng dẫn|huong dan|cách|cach)\s+(đăng nhập|dang nhap|đổi mật khẩu|doi mat khau|quên mật khẩu|quen mat khau|đăng ký|dang ky|tạo tài khoản|tao tai khoan|cập nhật thông tin|cap nhat thong tin)\b",
    r"\b(hotline|liên hệ|lien he|số điện thoại|so dien thoai|tổng đài|tong dai)\b",
    r"\b(quên mật khẩu|quen mat khau|đổi mật khẩu|doi mat khau|lấy lại mật khẩu|lay lai mat khau)\b",
    r"\b(cán bộ mới|can bo moi|bắt đầu|bat dau)\b",
    r"\bquy trình (nộp|khai báo|báo cáo)\b.*(trên phần mềm|trên hệ thống|như thế nào|thế nào)",
]

DWH_METRIC_PATTERNS = [
    r"\b(chỉ tiêu|chi tieu|kinh phí|kinh phi|khuyến công|khuyen cong|giải ngân|giai ngan|dự toán|du toan|ngân sách|ngan sach|sản xuất muối|san xuat muoi|ocop)\b",
    r"\b(thống kê|thong ke|số liệu|so lieu|tổng số|tong so|bao nhiêu|bao nhieu|xếp hạng|xep hang)\b",
    r"\b(tai nạn lao động|tai nan lao dong|tnlđ|tnld)\b",
    r"\b(fact_|criteria|collection_form|user_mission|office_mission|bảng fact|bang fact)\b",
    r"\b(báo cáo đã duyệt|trạng thái approved|status approved)\b",
    r"\b(sở công thương|so cong thuong|sở xây dựng|so xay dung|phòng kinh tế|phong kinh te|sở nội vụ|so noi vu)\b",
    r"\b(dwh|kho dữ liệu|kho du lieu|truy vấn csdl|truy van csdl|truy vấn dwh)\b",
    r"\b(năm 202[0-9]|nam 202[0-9])\b",
]


def is_warehouse_query(request: ChatRequest) -> bool:
    """
    Phân định thông minh giữa:
    - HDSD Document RAG (hướng dẫn sử dụng phần mềm, đổi mk, hotline, thủ tục công...)
    - DWH Autonomous Warehouse Agent (truy vấn số liệu, chỉ tiêu, kinh phí, SQL...).
    """
    raw_text = request.effective_query
    if not raw_text:
        return False
    text = raw_text.lower()

    # 1. Kiểm tra sớm xem câu hỏi có thuộc nhóm ngoài phạm vi DWH (thời tiết, thủ tục CCCD, chính trị, sáng tác, định tính...)
    if ScopePrechecker:
        is_out_of_scope, _, _ = ScopePrechecker.check_scope(raw_text)
        if is_out_of_scope:
            return False

    # 2. Ưu tiên kiểm tra câu hỏi thao tác phần mềm HDSD (đăng nhập, mật khẩu, hotline...) -> Luôn đi vào HDSD Document RAG
    if any(re.search(p, text) for p in HDSD_PROCEDURAL_PATTERNS):
        return False

    # 3. Nếu chỉ định collection là dwh/warehouse
    col = (request.collection or "").lower().strip()
    if col in DWH_COLLECTIONS:
        return True

    # 4. Nếu câu hỏi có từ khóa/mẫu hỏi số liệu kho DWH
    if any(re.search(p, text) for p in DWH_METRIC_PATTERNS):
        return True

    # 5. Nếu role chỉ định chuyên biệt cho DWH
    role = (request.role or "").lower().strip()
    if role in DWH_ROLES and not any(k in text for k in ["hdsd", "huong dan", "hướng dẫn"]):
        return True

    return False


async def stream_warehouse_agent(request: ChatRequest, session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream SSE các stage và tokens từ WarehouseLangGraphAgent,
    đồng thời lưu vết vào Supabase/SQLite chat history khi hoàn tất.
    """
    user_context = {
        "role_level": request.level if request.level is not None else 0,
        "tenant_code": request.tenant_code or "68",
        "department_code": request.department_code,
    }
    history = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
    full_answer_parts = []

    async for chunk in warehouse_agent.run_stream(
        query=request.effective_query,
        session_id=session_id,
        user_context=user_context,
        history=history,
    ):
        yield chunk
        if "event: done" in chunk:
            try:
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "full_answer" in data:
                            full_answer_parts.append(data["full_answer"])
            except Exception:
                pass

    if full_answer_parts:
        try:
            bot_resp = ChatResponse(
                session_id=session_id,
                answer=full_answer_parts[0],
                intent="dwh_autonomous_query",
                source_chunks=[],
                images=[],
                youtube_links=[],
                quick_action_chips=[]
            )
            asyncio.create_task(
                chat_history_service.save_chat_interaction_async(
                    session_id=session_id,
                    user_query=request.effective_query,
                    bot_response=bot_resp,
                    role=request.role or "lanhdao"
                )
            )
        except Exception as e:
            logger.warning(f"Could not persist warehouse chat to history: {e}")


@router.post("", response_model=ApiResponse[ChatResponse])
async def chat_endpoint(request: ChatRequest):
    try:
        if warehouse_agent is not None and is_warehouse_query(request):
            session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
            user_context = {
                "role_level": request.level if request.level is not None else 0,
                "tenant_code": request.tenant_code or "68",
                "department_code": request.department_code,
            }
            history = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
            result = warehouse_agent.run(
                query=request.effective_query,
                session_id=session_id,
                user_context=user_context,
                history=history,
            )
            raw_chips = result.get("quick_action_chips") or []
            chips = []
            for i, c in enumerate(raw_chips):
                if isinstance(c, dict):
                    chips.append(QuickActionChip(
                        id=c.get("id", f"chip_{i}"),
                        label=str(c.get("label", "")),
                        query_text=str(c.get("query_text", ""))
                    ))
                elif isinstance(c, QuickActionChip):
                    chips.append(c)
                else:
                    chips.append(QuickActionChip(id=f"chip_{i}", label=str(c), query_text=str(c)))
            answer = result.get("answer") or "Đã hoàn thành tra cứu số liệu."
            chat_response = ChatResponse(
                session_id=session_id,
                answer=answer,
                source_chunks=[],
                images=[],
                youtube_links=[],
                quick_action_chips=chips,
                intent="dwh_autonomous_query"
            )
            try:
                asyncio.create_task(
                    chat_history_service.save_chat_interaction_async(
                        session_id=session_id,
                        user_query=request.effective_query,
                        bot_response=chat_response,
                        role=request.role or "lanhdao"
                    )
                )
            except Exception as e:
                logger.warning(f"Could not persist warehouse chat to history: {e}")

            return ApiResponse(
                success=True,
                message="Generated response successfully",
                data=chat_response
            )

        response = await rag_service.process_chat(request)
        return ApiResponse(
            success=True,
            message="Generated response successfully",
            data=response
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams metadata, real-time stage updates, and tokens in real-time.
    """
    try:
        if warehouse_agent is not None and is_warehouse_query(request):
            session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
            return StreamingResponse(
                stream_warehouse_agent(request, session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        return StreamingResponse(
            rag_service.process_chat_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"Chat stream endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/audit/export")
async def export_audit_log():
    """
    Direct download endpoint for the JSONL audit log file.
    """
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, settings.CHAT_AUDIT_LOG_FILE)
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            pass  # Create empty file
    return FileResponse(
        path=log_file,
        filename="chat_audit.jsonl",
        media_type="application/x-jsonlines"
    )


@router.get("/history/{session_id}")
async def get_session_history(session_id: str):
    """
    Retrieves message history for a specific session directly from Supabase PostgreSQL.
    """
    try:
        messages = chat_history_service.get_session_messages(session_id)
        return ApiResponse(
            success=True,
            message=f"Retrieved {len(messages)} messages for session {session_id}",
            data=messages
        )
    except Exception as e:
        logger.error(f"Failed to fetch session history for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def get_all_sessions(limit: int = 50):
    """
    Retrieves latest chat sessions summary directly from Supabase PostgreSQL.
    """
    try:
        sessions = chat_history_service.get_all_sessions(limit=limit)
        return ApiResponse(
            success=True,
            message=f"Retrieved {len(sessions)} sessions",
            data=sessions
        )
    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



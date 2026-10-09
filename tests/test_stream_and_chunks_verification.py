import asyncio
import sys
from pathlib import Path

# UTF-8 encoding support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chat import ChatRequest
from app.services.rag_service import rag_service
from app.core.logger import logger


async def main():
    query = "cần nhập thông tin gì ở phần Thông tin Phường/Xã khi khai báo cáo tai nạn lao động đột xuất"
    print("=" * 80)
    print(f"🔍 KIỂM THỬ TOÀN PIPELINE VỚI CÂU HỎI:")
    print(f"   \"{query}\"")
    print("=" * 80)

    # 1. Warm up RAG Service
    print("\n⏳ [1/3] Đang khởi động RAG Service...")
    await rag_service.warmup()

    request = ChatRequest(
        query=query,
        role="phuong",
        session_id="test_verification_session"
    )

    # 2. Test Non-streaming (process_chat)
    print("\n" + "-" * 80)
    print("🧪 [2/3] KIỂM THỬ LUỒNG NON-STREAMING (process_chat):")
    print("-" * 80)
    res_non_stream = await rag_service.process_chat(request)
    print(f"\n📊 Kết quả Non-streaming:")
    print(f"   - Intent: {res_non_stream.intent}")
    print(f"   - Số lượng chunks đính kèm response: {len(res_non_stream.source_chunks)}")
    for i, c in enumerate(res_non_stream.source_chunks, 1):
        cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else str(c))
        sec = getattr(getattr(c, "metadata", None), "section_title", None) or (c.get("metadata", {}).get("section_title") if isinstance(c, dict) else "")
        print(f"     [{i}] ID: {cid} | Section: {sec}")
    print(f"   - Câu trả lời sinh ra từ LLM:\n     >>> {res_non_stream.answer}")

    # 3. Test Streaming (process_chat_stream)
    print("\n" + "-" * 80)
    print("🧪 [3/3] KIỂM THỬ LUỒNG SSE STREAMING (process_chat_stream):")
    print("-" * 80)
    stream_events = []
    tokens = []
    async for event in rag_service.process_chat_stream(request):
        stream_events.append(event)
        if "event: token" in event:
            import json
            for line in event.split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    tokens.append(data.get("content", ""))

    full_stream_answer = "".join(tokens).strip()
    print(f"\n📊 Kết quả Streaming:")
    print(f"   - Số lượng sự kiện SSE: {len(stream_events)}")
    print(f"   - Câu trả lời ghép từ Stream:\n     >>> {full_stream_answer}")

    print("\n" + "=" * 80)
    # Verification assertions
    assert len(res_non_stream.source_chunks) == 4, f"Expected 4 chunks in non-stream, got {len(res_non_stream.source_chunks)}"
    assert "Phường/xã" in res_non_stream.answer or "phường/xã" in res_non_stream.answer, "Non-stream answer missing expected fields"
    assert len(full_stream_answer) > 0, "Stream answer is empty"
    assert "từ chối" not in full_stream_answer.lower() and "hỗ trợ kỹ thuật" not in full_stream_answer.lower(), "Stream answer wrongly rejected"
    print("🎉 TẤT CẢ CÁC BƯỚC KIỂM THỬ STREAMING VÀ NON-STREAMING ĐÃ THÀNH CÔNG 100%!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

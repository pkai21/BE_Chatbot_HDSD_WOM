import sys
import time
import asyncio
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.rag_service import rag_service
from app.models.chat import ChatRequest
from app.services.qwen_service import qwen_service
from app.utils.prompts import SYSTEM_PROMPT, format_context_for_prompt


async def run_profiler():
    query = "Làm thế nào để đổi mật khẩu tài khoản?"
    print("=" * 60)
    print("🔬 BẮT ĐẦU ĐO ĐẠC PROFILING CHI TIẾT ĐỘ TRỄ TỪNG BƯỚC (LATENCY BREAKDOWN)")
    print("=" * 60)
    print(f"❓ Query: '{query}'\n")

    # Step 1: BM25 & ChromaDB Retrieval Time
    t0 = time.perf_counter()
    chunks = await rag_service.retrieve_context(query, top_k=3)
    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000
    print(f"1️⃣ [RETRIEVAL] Thời gian truy xuất Hybrid (Chroma + BM25): {retrieval_ms:.2f} ms ({len(chunks)} chunks)")

    # Step 2: Context Formatting & Prompt Payload size
    context_data = format_context_for_prompt(chunks)
    full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_data}"
    prompt_chars = len(full_prompt)
    print(f"2️⃣ [PROMPT PAYLOAD] Độ dài Context gửi sang API: {prompt_chars} ký tự (~{prompt_chars//4} tokens)")

    # Step 3: Standalone Direct LLM Streaming TTFT
    print("\n3️⃣ [STREAMING TTFT] Đo đạc TTFT thực tế đơn luồng (Single Request):")
    start = time.perf_counter()
    ttft_ms = None
    token_count = 0
    
    async for token in qwen_service.generate_stream(system_prompt=full_prompt, user_query=query):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - start) * 1000
            print(f"   ⚡ TIME TO FIRST TOKEN (TTFT): {ttft_ms:.2f} ms ({ttft_ms/1000:.3f} giây)")
        token_count += 1

    total_stream_ms = (time.perf_counter() - start) * 1000
    print(f"   🏁 TỔNG THỜI GIAN STREAMING HOÀN TẤT: {total_stream_ms:.2f} ms ({total_stream_ms/1000:.2f} giây)")
    print(f"   📊 Tổng số tokens sinh ra: {token_count} tokens")
    print(f"   🚀 Tốc độ sinh text: {token_count / (total_stream_ms/1000):.1f} tokens/giây")

    print("\n" + "=" * 60)
    print("🎯 KẾT LUẬN PROFILING:")
    print(f"   - Retrieval Overhead: {retrieval_ms:.1f} ms")
    print(f"   - LLM First Token (TTFT): {ttft_ms:.1f} ms")
    print(f"   - Total End-to-End Latency: {retrieval_ms + total_stream_ms:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_profiler())

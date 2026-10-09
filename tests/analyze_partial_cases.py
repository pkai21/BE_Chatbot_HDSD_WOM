import os
import sys
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
from tests.run_ragas_evaluation import TEST_SUITE


async def run_diagnostics():
    print("=" * 80)
    print("🔍 PHÂN TÍCH CHI TIẾT TỪNG TEST CASE CHƯA ĐẠT ĐIỂM TỐI ĐA")
    print("=" * 80)

    for tc in TEST_SUITE:
        req = ChatRequest(query=tc["query"], top_k=4)
        resp = await rag_service.process_chat(req)
        
        answer_text = resp.answer.lower()
        retrieved_text = " ".join([c.text_content.lower() for c in resp.source_chunks]) if resp.source_chunks else ""
        evaluated_text = (answer_text + " " + retrieved_text).strip()

        matched_keywords = [k for k in tc["expected_keywords"] if k.lower() in evaluated_text]
        critical_matched = [c for c in tc["critical_notes"] if c.lower() in evaluated_text]

        kw_score = len(matched_keywords) / len(tc["expected_keywords"]) if tc["expected_keywords"] else 1.0
        crit_score = len(critical_matched) / len(tc["critical_notes"]) if tc["critical_notes"] else 1.0

        # Check if Partial or Fail
        if kw_score < 0.6 or crit_score < 1.0:
            print(f"\n📌 TEST CASE: {tc['id']} - {tc['name']}")
            print(f"📂 Nhóm tình huống: {tc['category']}")
            print(f"❓ Câu hỏi User: \"{tc['query']}\"")
            print(f"🎯 Ground Truth: {tc.get('ground_truth')}")
            print(f"\n🔑 Phân tích Khớp từ khóa & Điểm:")
            print(f"   - Từ khóa mong đợi: {tc['expected_keywords']}")
            print(f"   - Từ khóa đã khớp: {matched_keywords} ({kw_score*100:.1f}%)")
            print(f"   - Lưu ý cốt lõi (Critical): {tc['critical_notes']}")
            print(f"   - Lưu ý đã khớp: {critical_matched} ({crit_score*100:.1f}%)")
            print(f"   - Intent nhận diện: {resp.intent}")
            
            print(f"\n🤖 Câu trả lời thực tế của Chatbot:")
            print(f"{resp.answer}")

            print(f"\n📚 Các Chunks tài liệu được trích xuất ({len(resp.source_chunks)} chunks):")
            for idx, c in enumerate(resp.source_chunks):
                print(f"   [{idx+1}] [{c.metadata.module} -> {c.metadata.section_title}] (Source: {c.metadata.document_source})")
                print(f"       Nội dung: {c.text_content[:200].replace(chr(10), ' ')}...\n")
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())

import asyncio
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import psycopg2
from app.core.config import settings
from app.models.chat import ChatRequest, ChatResponse, QuickActionChip, ContactSupportInfo
from app.services.chat_history_service import chat_history_service
from app.services.rag_service import rag_service


def test_direct_service_save():
    print("\n--- 1. Testing ChatHistoryService Direct Save ---")
    test_session_id = str(uuid.uuid4())
    test_query = "Kiểm tra lưu lịch sử hội thoại vào Supabase"
    test_response = ChatResponse(
        session_id=test_session_id,
        answer="Đây là câu trả lời kiểm thử tự động ghi vào Supabase PostgreSQL.",
        intent="test_intent",
        images=["https://chatbothdsd-production.up.railway.app/static/images/hdsd_atl_phuong_img_38_rId8.jpg"],
        youtube_links=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        quick_action_chips=[QuickActionChip(id="chip_test_1", label="Chi tiết", query_text="Chi tiết quy trình")],
        contact_support=ContactSupportInfo.for_phuong()
    )

    success = chat_history_service._save_interaction_sync(
        session_id=test_session_id,
        user_query=test_query,
        bot_response=test_response,
        role="phuong",
        user_identifier="test_admin"
    )
    assert success, "Direct save failed!"
    print(f"✅ Direct save succeeded for session: {test_session_id}")

    # Verify query
    messages = chat_history_service.get_session_messages(test_session_id)
    assert len(messages) == 2, f"Expected 2 messages (user + bot), got {len(messages)}"
    print(f"✅ Retrieved {len(messages)} messages from Supabase:")
    for m in messages:
        print(f"   [{m['role'].upper()}]: {m['content'][:60]}... (Images: {len(m['images'])})")


async def test_full_rag_pipeline_persistence():
    print("\n--- 2. Testing Full RAG Pipeline Persistence ---")
    test_session_id = str(uuid.uuid4())
    request = ChatRequest(
        query="Hướng dẫn tôi cách đăng nhập hệ thống an toàn lao động",
        role="phuong",
        session_id=test_session_id,
        stream=False
    )
    resp = await rag_service.process_chat(request)
    print(f"✅ RAG processed. Answer length: {len(resp.answer)}")

    # Allow async background task to complete write
    await asyncio.sleep(1.5)

    # Check Supabase records
    messages = chat_history_service.get_session_messages(test_session_id)
    assert len(messages) >= 2, f"Expected at least 2 messages in Supabase, got {len(messages)}"
    print(f"✅ Verified RAG conversation recorded in Supabase! ({len(messages)} messages in DB)")


def verify_tables_status():
    print("\n--- 3. Verifying Overall Supabase Database Counts ---")
    conn = psycopg2.connect(settings.DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chat_sessions;")
        sessions_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chat_messages;")
        messages_count = cur.fetchone()[0]
        print(f"📊 Total `chat_sessions` in Supabase: {sessions_count}")
        print(f"📊 Total `chat_messages` in Supabase: {messages_count}")
    conn.close()


async def main():
    print("=== STARTING SUPABASE CHAT PERSISTENCE VERIFICATION ===")
    test_direct_service_save()
    await test_full_rag_pipeline_persistence()
    verify_tables_status()
    print("\n🎉 ALL SUPABASE PERSISTENCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(main())

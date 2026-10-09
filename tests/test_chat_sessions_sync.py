import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.chat_history_service import chat_history_service


client = TestClient(app)


def test_service_get_all_sessions():
    """Kiểm tra hàm get_all_sessions trong ChatHistoryService trả về danh sách phiên hợp lệ."""
    sessions = chat_history_service.get_all_sessions(limit=10)
    assert isinstance(sessions, list)
    if len(sessions) > 0:
        s = sessions[0]
        assert "id" in s
        assert "role" in s
        assert "title" in s
        assert "createdAt" in s
        assert "updatedAt" in s
        assert "messages" in s


def test_api_get_sessions_endpoint():
    """Kiểm tra endpoint GET /api/v1/chat/sessions trả về 200 và cấu trúc ApiResponse chuẩn."""
    response = client.get("/api/v1/chat/sessions?limit=5")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data.get("success") is True
    assert isinstance(json_data.get("data"), list)


def test_service_and_api_get_session_messages_has_timestamp():
    """Kiểm tra cả Service và API lấy messages của 1 session đều trả về key timestamp."""
    sessions = chat_history_service.get_all_sessions(limit=1)
    if len(sessions) > 0:
        session_id = sessions[0]["id"]
        # 1. Service check
        messages = chat_history_service.get_session_messages(session_id)
        assert isinstance(messages, list)
        if len(messages) > 0:
            assert "timestamp" in messages[0]
            assert "created_at" in messages[0]

        # 2. Endpoint check
        res = client.get(f"/api/v1/chat/history/{session_id}")
        assert res.status_code == 200
        data = res.json().get("data", [])
        if len(data) > 0:
            assert "timestamp" in data[0]

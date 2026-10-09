import pytest
from app.services.intent_service import IntentService


@pytest.fixture
def intent_service():
    return IntentService()


class TestProceduralFirstRouter:
    """
    Test suite kiểm thử toàn diện cho Procedural-First Hierarchical Intent Engine (PF-HIE).
    Đảm bảo 100% các câu hỏi vị trí thao tác / bắt đầu quy trình được route đúng vào Mode A (procedural_extractive).
    """

    def test_phuong_change_password_location(self, intent_service):
        """Test Case 1 (User Issue): 'đổi mật khẩu chỗ nào ?' ở role Phường/Xã."""
        query = "đổi mật khẩu chỗ nào ?"
        result = intent_service.classify_intent(query, role="phuong")
        
        assert result.intent == "knowledge_query"
        assert result.target_module == "THAY ĐỔI MẬT KHẨU"
        assert result.strategy == "procedural_extractive", f"Expected procedural_extractive, got {result.strategy}"

    def test_phuong_edit_sub_account_location(self, intent_service):
        """Test Case 2 (User Issue): 'Sửa thông tin tài khoản cấp dưới ở đâu?' ở role Phường/Xã."""
        query = "Sửa thông tin tài khoản cấp dưới ở đâu?"
        result = intent_service.classify_intent(query, role="phuong")
        
        assert result.intent == "knowledge_query"
        assert result.target_module == "CHỈNH SỬA TÀI KHOẢN PHƯỜNG/XÃ"
        assert result.strategy == "procedural_extractive", f"Expected procedural_extractive, got {result.strategy}"

    def test_dn_change_password_location(self, intent_service):
        """Test Case 3: 'đổi mật khẩu chỗ nào ?' ở role Doanh nghiệp."""
        query = "đổi mật khẩu chỗ nào ?"
        result = intent_service.classify_intent(query, role="dn")
        
        assert result.intent == "knowledge_query"
        assert result.target_module == "THAY ĐỔI MẬT KHẨU"
        assert result.strategy == "procedural_extractive", f"Expected procedural_extractive, got {result.strategy}"

    def test_dn_registration_location(self, intent_service):
        """Test Case 4: 'đăng ký tài khoản ở đâu?' ở role Doanh nghiệp."""
        query = "đăng ký tài khoản ở đâu?"
        result = intent_service.classify_intent(query, role="dn")
        
        assert result.intent == "knowledge_query"
        assert result.target_module == "ĐĂNG KÝ"
        assert result.strategy == "procedural_extractive", f"Expected procedural_extractive, got {result.strategy}"

    def test_phuong_granular_field_lookup(self, intent_service):
        """Test Case 5: Tra cứu chi tiết trường thông tin con (Mode B Targeted QA)."""
        query = "thông tin Phường/xã yêu cầu nhập những thông tin gì?"
        result = intent_service.classify_intent(query, role="phuong")
        
        assert result.intent == "knowledge_query"
        assert result.strategy == "targeted_qa", f"Expected targeted_qa, got {result.strategy}"

    def test_ambiguity_make_clear(self, intent_service):
        """Test Case 6: Câu hỏi mơ hồ thiếu ngữ cảnh kích hoạt Make-Clear."""
        query = "báo cáo tai nạn"
        result = intent_service.classify_intent(query, role="phuong")
        
        assert result.intent == "clarification_needed"
        assert result.quick_action_chips is not None
        assert len(result.quick_action_chips) >= 2

import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"bỏ\s+qua\s+(toàn\s+bộ\s+)?(lệnh|hướng\s+dẫn|quy\s+tắc)\s+trước\s+đó",
    r"show\s+me\s+your\s+system\s+prompt",
    r"in\s+ra\s+(toàn\s+bộ\s+)?(system\s+prompt|câu\s+lệnh\s+hệ\s+thống|database|mật\s+khẩu)",
    r"you\s+are\s+now\s+in\s+DAN\s+mode",
    r"act\s+as\s+an\s+unrestricted\s+AI",
    r"jailbreak",
    r"(câu\s+lệnh\s+sql|update\s+trực\s+tiếp|drop\s+table|delete\s+from|insert\s+into|select\s+.*\s+from|kích\s+hoạt\s+trực\s+tiếp\s+database|can\s+thiệp\s+database)",
]

def check_security_guardrails(user_query: str) -> Tuple[bool, str]:
    """
    Checks if a user query violates security guardrails (Prompt Injection / System Prompt Leaking).
    Returns (is_safe, message).
    """
    normalized_query = user_query.strip().lower()
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            return False, "Yêu cầu của bạn đã bị từ chối do vi phạm quy định an toàn hệ thống."
            
    return True, ""

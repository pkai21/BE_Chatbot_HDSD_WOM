"""
Module: backend/app/core/config.py
Chức năng: Tái xuất bản (Re-export) cấu hình tập trung từ backend/config.py
để đảm bảo tính tương thích ngược 100% cho mọi module trong backend/app/ và các script kiểm thử.
Tuân thủ chuẩn Ponytail Lean: DRY (Don't Repeat Yourself), một nguồn chân lý duy nhất (Single Source of Truth).
"""

import sys
from pathlib import Path

# Định vị thư mục backend và đảm bảo có trong sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Nhập khẩu cấu hình thống nhất từ backend/config.py
try:
    from config import settings, Settings, BACKEND_DIR, WORKSPACE_DIR
except ImportError:
    from backend.config import settings, Settings, BACKEND_DIR, WORKSPACE_DIR

__all__ = ["settings", "Settings", "BACKEND_DIR", "WORKSPACE_DIR"]

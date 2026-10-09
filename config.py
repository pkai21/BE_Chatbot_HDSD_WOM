"""
Module: backend/config.py
Chức năng: Quản lý cấu hình tập trung và các API Keys được phép sử dụng cho dự án IPGov_Chatbot
(kết hợp kế thừa tương thích các cấu hình từ Chatbot HDSD).
Tuân thủ chuẩn Arc42, IEEE Std 1016-2009 và Triết lý Tinh Giản Ponytail Lean.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Đường dẫn thư mục gốc backend
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BACKEND_DIR.parent

# Tự động nạp biến môi trường từ .env trước khi khởi tạo Settings
from dotenv import load_dotenv
load_dotenv(WORKSPACE_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


class Settings(BaseSettings):
    # ==============================================================================
    # 1. THÔNG TIN CHUNG DỰ ÁN (PROJECT & ENVIRONMENT METADATA)
    # ==============================================================================
    PROJECT_NAME: str = "IPGov Chatbot Assistant"
    PROJECT_CODE: str = "IPGov_Chatbot"
    APP_NAME: str = "IPGov Chatbot Assistant"
    APP_VERSION: str = "1.0.0"
    BACKEND_DIR: Path = BACKEND_DIR
    WORKSPACE_DIR: Path = WORKSPACE_DIR
    IPGOV_DIR: Path = WORKSPACE_DIR / "IPGov_Chatbot"
    BASE_DIR: Path = WORKSPACE_DIR / "IPGov_Chatbot"
    FRONTEND_DIR: Path = WORKSPACE_DIR / "IPGov_Chatbot" / "frontend"
    BENCH_HTML_PATH: Path = WORKSPACE_DIR / "IPGov_Chatbot" / "frontend" / "role_selector_bench.html"
    ENVIRONMENT: str = "development"  # "development", "staging", "production"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ==============================================================================
    # ==============================================================================
    # 2. CÁC API KEYS MÔ HÌNH NGÔN NGỮ ĐƯỢC PHÉP SỬ DỤNG CHO IPGOV_CHATBOT (LLM APIS)
    # ==============================================================================
    # 2.1. Alibaba DashScope / Qwen API Gateway
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    QWEN_MODEL_NAME: str = os.getenv("QWEN_MODEL_NAME", "qwen3.7-flash")

    # 2.2. Vô hiệu hóa Groq API (Không phù hợp hạn mức TPM/RPM)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_PRIMARY_MODEL: str = ""
    GROQ_ALLOWED_MODELS: List[str] = []

    # 2.3. OpenRouter API Gateway (Hỗ trợ truy cập đa mô hình Gemini, DeepSeek, Qwen...)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_HEAVY_MODEL: str = os.getenv("OPENROUTER_HEAVY_MODEL", "deepseek/deepseek-v4.1-flash")
    OPENROUTER_LIGHT_MODEL: str = os.getenv("OPENROUTER_LIGHT_MODEL", "google/gemini-2.5-flash-lite")
    OPENROUTER_PRIMARY_MODEL: str = os.getenv("OPENROUTER_PRIMARY_MODEL", "google/gemini-2.5-flash-lite")
    OPENROUTER_MODEL_NAME: str = os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-2.5-flash-lite")
    ROUTER_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.7"))
    ENABLE_OPENROUTER_RESPONSE_CACHE: bool = os.getenv("ENABLE_OPENROUTER_RESPONSE_CACHE", "true").lower() in ("true", "1", "yes")

    # 2.4. Google GenAI API Gateway (via shopaikey proxy)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_BASE_URL: str = os.getenv("GOOGLE_BASE_URL", "https://api.shopaikey.com")
    GEMINI_HEAVY_MODEL: str = os.getenv("GEMINI_HEAVY_MODEL", "gemini-3.7-flash")
    GEMINI_LIGHT_MODEL: str = os.getenv("GEMINI_LIGHT_MODEL", "gemini-3.5-flash-lite")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
    ACTIVE_LLM_PROVIDER: str = os.getenv("ACTIVE_LLM_PROVIDER", "openrouter")

    # 2.5. Tham số điều khiển suy luận Router (Tất định, Ponytail Lean, Low Budget Tokens)
    ROUTER_TEMPERATURE: float = 0.0
    ROUTER_MAX_TOKENS: int = 800
    ROUTER_ENABLE_THINKING: bool = False  # Tắt reasoning tokens để tối ưu chi phí & tốc độ

    # 2.6. Tham số điều khiển suy luận LLM chung (SQL Generation & Synthesis)
    LLM_TEMPERATURE: float = 0.0  # T=0.0 để loại trừ tính ngẫu nhiên khi sinh SQL/Slot
    LLM_TOP_P: float = 0.95
    LLM_MAX_TOKENS: int = 2048

    # 2.7. Task-based API Keys cho Autonomous Warehouse Agent (ADR-001)
    AGENT_DECISION_LLM_API_KEY: Optional[str] = os.getenv("AGENT_DECISION_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")
    SQL_GENERATION_LLM_API_KEY: Optional[str] = os.getenv("SQL_GENERATION_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")
    RESPONSE_SYNTHESIS_LLM_API_KEY: Optional[str] = os.getenv("RESPONSE_SYNTHESIS_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY")

    AGENT_PRIMARY_MODEL: str = os.getenv("AGENT_PRIMARY_MODEL", "google/gemini-2.5-flash-lite")
    AGENT_FALLBACK_MODEL_1: str = os.getenv("AGENT_FALLBACK_MODEL_1", "google/gemini-3.5-flash-lite")
    AGENT_FALLBACK_MODEL_2: str = os.getenv("AGENT_FALLBACK_MODEL_2", "deepseek/deepseek-v4.1-flash")
    AGENT_MEMORY_DB_PATH: str = os.getenv("AGENT_MEMORY_DB_PATH", "data/agent_memory.db")


    @property
    def effective_llm_api_key(self) -> str:
        """Trả về API key ưu tiên (OpenRouter hoặc Google GenAI)."""
        if self.ACTIVE_LLM_PROVIDER == "openrouter" and self.OPENROUTER_API_KEY:
            return self.OPENROUTER_API_KEY
        if self.ACTIVE_LLM_PROVIDER == "google" and self.GOOGLE_API_KEY:
            return self.GOOGLE_API_KEY
        return self.OPENROUTER_API_KEY or self.GOOGLE_API_KEY

    @property
    def effective_api_key(self) -> str:
        """Alias tương thích ngược cho effective_llm_api_key."""
        return self.effective_llm_api_key

    @property
    def allowed_llm_providers(self) -> List[str]:
        """Danh sách các nhà cung cấp LLM đã được nạp API key hợp lệ."""
        providers = []
        if self.OPENROUTER_API_KEY:
            providers.append("openrouter")
        if self.GOOGLE_API_KEY:
            providers.append("google")
        return providers

    # ==============================================================================
    # 3. KHO DỮ LIỆU DWH POSTGRESQL (GROUND TRUTH: vna_wom_dev)
    # ==============================================================================
    DWH_HOST: str = os.getenv("DWH_HOST", "104.248.155.6")
    DWH_PORT: int = int(os.getenv("DWH_PORT", 5432))
    DWH_USER: str = os.getenv("DWH_USER", "vna_wom_dev")
    DWH_PASSWORD: str = os.getenv("DWH_PASSWORD", "1d4ed4dd-d5df-4081-809f-884b7ca16cbd")
    DWH_DB: str = os.getenv("DWH_DB", "vna_wom_dev")
    DWH_DATABASE_URL: Optional[str] = None
    DWH_POOL_MIN_SIZE: int = 5
    DWH_POOL_MAX_SIZE: int = 25
    DWH_TIMEOUT_SECONDS: float = 5.0

    @property
    def effective_dwh_url(self) -> str:
        """Trả về URL kết nối asyncpg PostgreSQL chuẩn cho kho DWH."""
        if self.DWH_DATABASE_URL:
            return self.DWH_DATABASE_URL
        return f"postgresql+asyncpg://{self.DWH_USER}:{self.DWH_PASSWORD}@{self.DWH_HOST}:{self.DWH_PORT}/{self.DWH_DB}"

    @property
    def sync_dwh_url(self) -> str:
        """Trả về URL kết nối đồng bộ (psycopg2) cho benchmark / script testing."""
        return f"postgresql://{self.DWH_USER}:{self.DWH_PASSWORD}@{self.DWH_HOST}:{self.DWH_PORT}/{self.DWH_DB}"

    @property
    def sqlalchemy_sync_dwh_url(self) -> str:
        """Trả về URL kết nối đồng bộ chuẩn SQLAlchemy (psycopg2)."""
        return f"postgresql+psycopg2://{self.DWH_USER}:{self.DWH_PASSWORD}@{self.DWH_HOST}:{self.DWH_PORT}/{self.DWH_DB}"

    # ==============================================================================
    # 4. IN-MEMORY DUCKDB & METADATA CATALOG SETTINGS
    # ==============================================================================
    DUCKDB_PATH: str = ":memory:"  # Chạy thuần trong RAM để đạt tốc độ < 2ms
    CATALOG_AUTO_SYNC_MINUTES: int = 5
    ENABLE_RAPIDFUZZ_MATCHING: bool = True
    RAPIDFUZZ_THRESHOLD: float = 75.0

    # ==============================================================================
    # 5. BẢO MẬT XÁC THỰC, PHÂN QUYỀN HBAC & AST GUARDRAILS
    # ==============================================================================
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "ipgov_super_secret_jwt_key_development_only_change_in_prod")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_SECONDS: int = 3600 * 8
    INTERNAL_SYNC_SECRET: str = os.getenv("INTERNAL_SYNC_SECRET", "ipgov_internal_sync_secret_token_change_in_prod")
    ENFORCE_HBAC_STRICT: bool = True
    SQL_MAX_LIMIT: int = 500

    # ==============================================================================
    # 6. HỆ THỐNG QUAN SÁT & GIÁM SÁT CỤC BỘ (LOCAL OBSERVABILITY & TRACING)
    # ==============================================================================
    OTEL_ENABLED: bool = True
    OTEL_SERVICE_NAME: str = "ipgov-chatbot-backend"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"  # Local Jaeger / OTel collector
    
    # Langfuse Tracing (Cục bộ hoặc Cloud)
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"  # Mặc định Local Langfuse Docker
    
    # Dead-Letter-Queue (DLQ) Incident Store
    ENABLE_DLQ_STORAGE: bool = True
    DLQ_TABLE_NAME: str = "public.chatbot_dlq_incidents"

    # ==============================================================================
    # 7. CẤU HÌNH KẾ THỪA TỪ CHATBOT HDSD (TƯƠNG THÍCH NGƯỢC)
    # ==============================================================================
    EMBEDDING_MODEL_NAME: str = "AITeamVN/Vietnamese_Embedding_v2"
    EMBEDDING_DIMENSION: int = 1024
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    USE_REMOTE_EMBEDDING: bool = os.getenv("USE_REMOTE_EMBEDDING", "true").lower() in ("true", "1", "yes")
    HF_EMBEDDING_SPACE_URL: str = os.getenv("HF_EMBEDDING_SPACE_URL", "https://nato1306-vietnamese-embedding-api.hf.space")

    DATABASE_URL: Optional[str] = None
    SUPABASE_URL: Optional[str] = "https://tykwgiubhnxedlpxszdn.supabase.co"
    SUPABASE_PASS: str = os.getenv("SUPABASE_PASS", "")
    SUPABASE_ANON_KEY: Optional[str] = None

    BASE_IMAGE_CDN_URL: Optional[str] = None

    @property
    def effective_image_cdn_url(self) -> str:
        """Trả về URL CDN phục vụ ảnh: localhost trong dev hoặc domain public trong production."""
        if self.BASE_IMAGE_CDN_URL:
            return self.BASE_IMAGE_CDN_URL
        return f"http://localhost:{self.PORT}/static/images"

    @property
    def effective_database_url(self) -> Optional[str]:
        """Trả về URL cơ sở dữ liệu Supabase/PostgreSQL cho Chatbot HDSD (None trong dev để dùng SQLite)."""
        return self.DATABASE_URL

    VECTOR_DB_TYPE: str = "chroma"
    CHROMA_PERSIST_DIRECTORY: str = str(BACKEND_DIR / "chroma_data")
    CHROMA_COLLECTION_PHUONG: str = "hdsd_phuong"
    CHROMA_COLLECTION_DN: str = "hdsd_dn"
    DEFAULT_COLLECTION: str = "hdsd_phuong"
    IMAGE_STORAGE_PATH: str = str(BACKEND_DIR / "data" / "extracted_images")

    # 2.3. Redis Session & Semantic Decision Cache Settings
    REDIS_ENABLED: bool = True
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_SOCKET_TIMEOUT: float = 2.0
    REDIS_MAX_CONNECTIONS: int = 20

    SESSION_TTL_SECONDS: int = 1800        # 30 phút trượt
    SESSION_MAX_HISTORY_TURNS: int = 3     # Giữ 3 lượt gần nhất (6 tin nhắn)
    ENABLE_DECISION_CACHE: bool = True     # Cache quyết định định tuyến Lượt 1
    DECISION_CACHE_TTL_SECONDS: int = 3600 # 1 giờ cache

    LOG_DIR: str = str(BACKEND_DIR / "data" / "logs")
    CHAT_AUDIT_LOG_FILE: str = "chat_audit.jsonl"
    CHAT_ERROR_LOG_FILE: str = "chat_errors.jsonl"

    CORS_ORIGINS: List[str] = [
        "*",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://chatbot-hdsd.vercel.app",
        "https://ipgov-chatbot-frontend.onrender.com",
    ]

    model_config = SettingsConfigDict(
        env_file=(
            ".env",
            "../.env",
            "backend/.env",
            "IPGov_Chatbot/.env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Khởi tạo singleton instance cấu hình
settings = Settings()

# Tự động nạp token HuggingFace nếu có
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

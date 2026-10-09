import os
import sys
from pathlib import Path

# Đảm bảo thư mục backend và workspace root luôn có trong sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_WORKSPACE_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.logger import logger
from app.api.v1.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.ingestion import router as ingestion_router
from app.services.rag_service import rag_service
from app.services.qwen_service import qwen_service
from app.services.semantic_router import semantic_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager:
    Automatically warms up Vector Embedding, BM25 Index, and HTTP/2 API session pool on startup.
    Gracefully cleans up network sessions on shutdown.
    """
    logger.info("🚀 [Startup] Đang khởi động toàn bộ hệ thống HDSD Chatbot...")
    
    async def _do_warmup():
        try:
            # 1. Warm up Vector Retrieval (ChromaDB ONNX + BM25 Sparse Index)
            await rag_service.warmup()
            # 2. Warm up Semantic Router (Embedding Anchors) in background thread
            await asyncio.to_thread(semantic_router.warmup)
            # 3. Warm up Qwen LLM API (HTTP/2 Connection Pool & SSL Handshake)
            await qwen_service.warmup()
            logger.info("🔥 [Startup] Hoàn tất warm-up hệ thống! Sẵn sàng phục vụ yêu cầu với tốc độ tối đa (TTFT < 1s).")
        except Exception as e:
            logger.warning(f"⚠️ [Startup] Có lỗi trong quá trình warm-up: {e}")

    # Launch warmup in background so port binds immediately (< 500ms)
    asyncio.create_task(_do_warmup())

    yield

    logger.info("🛑 [Shutdown] Đang đóng các kết nối nền...")
    await qwen_service.close()
    logger.info("✅ [Shutdown] Hệ thống đã tắt an toàn.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal HDSD Chatbot Assistant API powered by Qwen-3.7-Flash & Vector RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure Static Image Directory Exists and Mount Static Files
os.makedirs(settings.IMAGE_STORAGE_PATH, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=settings.IMAGE_STORAGE_PATH), name="images")

# Register API Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "api_v1_prefix": "/api/v1"
    }


if __name__ == "__main__":
    reload_kwargs = {}
    if settings.DEBUG:
        reload_kwargs = {
            "reload": True,
            "reload_dirs": [str(_BACKEND_DIR)],
            "reload_excludes": [
                "*.db", "*.db-wal", "*.db-shm", "*.sqlite", "*.sqlite3",
                "*.parquet", "*.log", "*.jsonl",
                "*.tmp", "*.pyc", "__pycache__/*",
                ".next/*", "*/.next/*",
                "node_modules/*", "*/node_modules/*",
                "data/*", "*/data/*",
                "tests/logs/*", "*/tests/logs/*",
                ".git/*", "*/.git/*"
            ]
        }
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, **reload_kwargs)

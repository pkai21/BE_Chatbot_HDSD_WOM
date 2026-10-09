import os
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.core.logger import logger
from app.models.chunk import DocumentChunk


class AuditLogger:
    def __init__(self):
        self.log_dir = getattr(settings, "LOG_DIR", "./data/logs")
        self.audit_log_file = os.path.join(self.log_dir, getattr(settings, "CHAT_AUDIT_LOG_FILE", "chat_audit.jsonl"))
        self.error_log_file = os.path.join(self.log_dir, getattr(settings, "CHAT_ERROR_LOG_FILE", "chat_errors.jsonl"))
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log directory {self.log_dir}: {e}")

    def log_interaction(
        self,
        session_id: str,
        user_query: str,
        answer: str,
        intent: str,
        status: str = "SUCCESS",
        execution_time_ms: float = 0.0,
        ttft_ms: Optional[float] = None,
        source_chunks: Optional[List[DocumentChunk]] = None,
        images: Optional[List[str]] = None,
        youtube_links: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs a full chat interaction (question + answer + context + metrics) to a JSON Lines file.
        Preserves Vietnamese UTF-8 characters for easy analysis.
        """
        if not getattr(settings, "ENABLE_AUDIT_LOG", True):
            return

        self._ensure_log_dir()

        # Format retrieved chunks summary for analysis
        chunks_summary = []
        if source_chunks:
            for chunk in source_chunks:
                chunks_summary.append({
                    "chunk_id": chunk.id,
                    "document_source": chunk.metadata.document_source,
                    "section_title": chunk.metadata.section_title,
                    "module": chunk.metadata.module,
                    "sub_module": chunk.metadata.sub_module,
                    "content_preview": chunk.text_content[:150] + ("..." if len(chunk.text_content) > 150 else "")
                })

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user_query": user_query,
            "intent": intent,
            "status": status,
            "execution_time_ms": round(execution_time_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "chunks_retrieved_count": len(chunks_summary),
            "source_chunks": chunks_summary,
            "images_count": len(images) if images else 0,
            "images": images or [],
            "youtube_links_count": len(youtube_links) if youtube_links else 0,
            "youtube_links": youtube_links or [],
            "bot_answer": answer,
            "error_message": error_message,
            "extra_data": extra_data or {}
        }

        # 1. Append to main audit log
        self._write_jsonl(self.audit_log_file, log_record)

        # 2. If status is ERROR or intent indicates software error / security block, also write to error log file
        if status in ("ERROR", "SECURITY_BLOCKED") or error_message or intent == "software_error":
            self._write_jsonl(self.error_log_file, log_record)

    def _write_jsonl(self, file_path: str, record: dict):
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log to {file_path}: {e}")


audit_logger = AuditLogger()

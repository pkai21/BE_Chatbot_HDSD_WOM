import os
import json
import uuid
import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.core.logger import logger
from app.models.chat import ChatResponse, QuickActionChip, ContactSupportInfo
from app.models.chunk import DocumentChunk

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    psycopg2 = None
    Json = None


class ChatHistoryService:
    """
    Service responsible for persisting chat sessions and chat messages.
    Supports Supabase PostgreSQL with an automatic, seamless fallback to Local SQLite
    when running locally without cloud database access.
    """

    def __init__(self):
        self.db_url = settings.effective_database_url
        self.sqlite_db_path = Path(settings.BACKEND_DIR) / "data" / "local_chat_history.db"
        self._use_sqlite_fallback = (self.db_url is None)
        self._supabase_warned = False
        self._init_sqlite()

    def _init_sqlite(self):
        """Initializes local SQLite database schema for local offline chat persistence."""
        try:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.sqlite_db_path)) as conn:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        user_identifier TEXT,
                        role TEXT,
                        title TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        intent TEXT,
                        source_chunks TEXT,
                        images TEXT,
                        youtube_links TEXT,
                        quick_action_chips TEXT,
                        contact_support TEXT,
                        created_at TEXT,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                    );
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize local SQLite database: {e}")

    def _get_connection(self):
        """Attempts connection to PostgreSQL/Supabase. Returns None if unreachable or fallback is active."""
        if self._use_sqlite_fallback or not self.db_url or psycopg2 is None:
            return None

        try:
            return psycopg2.connect(self.db_url, connect_timeout=3)
        except Exception as e:
            if not self._supabase_warned:
                logger.info(f"ℹ️ [ChatHistory] Supabase cloud is unreachable ({e}). Seamlessly switching to local SQLite database (data/{self.sqlite_db_path.name}).")
                self._supabase_warned = True
            self._use_sqlite_fallback = True
            return None

    def _format_chunk_for_db(self, chunk: DocumentChunk) -> dict:
        return {
            "id": chunk.id,
            "document_source": chunk.metadata.document_source,
            "section_title": chunk.metadata.section_title,
            "module": chunk.metadata.module,
            "sub_module": chunk.metadata.sub_module,
            "text_preview": chunk.text_content[:200] + ("..." if len(chunk.text_content) > 200 else "")
        }

    def _save_interaction_sync(
        self,
        session_id: str,
        user_query: str,
        bot_response: ChatResponse,
        role: str = "phuong",
        user_identifier: Optional[str] = None
    ) -> bool:
        """
        Synchronous worker executed in a background worker thread.
        Persists chat interactions to Supabase PostgreSQL, with automatic fallback to Local SQLite.
        """
        try:
            try:
                valid_session_uuid = str(uuid.UUID(session_id))
            except Exception:
                valid_session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))

            title = user_query.strip()[:100]
            now_iso = datetime.now(timezone.utc).isoformat()

            source_chunks_data = [self._format_chunk_for_db(c) for c in bot_response.source_chunks] if bot_response.source_chunks else []
            images_data = bot_response.images if bot_response.images else []
            youtube_links_data = bot_response.youtube_links if bot_response.youtube_links else []
            chips_data = [c.model_dump() for c in bot_response.quick_action_chips] if bot_response.quick_action_chips else []
            contact_data = bot_response.contact_support.model_dump() if bot_response.contact_support else None

            conn = self._get_connection()
            if conn:
                try:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO chat_sessions (id, user_identifier, role, title, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, NOW(), NOW())
                                ON CONFLICT (id) DO UPDATE 
                                SET updated_at = NOW(),
                                    role = EXCLUDED.role,
                                    title = COALESCE(chat_sessions.title, EXCLUDED.title);
                            """, (valid_session_uuid, user_identifier or "anonymous_user", role, title))

                            cur.execute("""
                                INSERT INTO chat_messages (id, session_id, role, content, created_at)
                                VALUES (%s, %s, 'user', %s, NOW());
                            """, (str(uuid.uuid4()), valid_session_uuid, user_query))

                            cur.execute("""
                                INSERT INTO chat_messages (
                                    id, session_id, role, content, intent,
                                    source_chunks, images, youtube_links,
                                    quick_action_chips, contact_support, created_at
                                )
                                VALUES (%s, %s, 'assistant', %s, %s, %s, %s, %s, %s, %s, NOW());
                            """, (
                                str(uuid.uuid4()),
                                valid_session_uuid,
                                bot_response.answer,
                                bot_response.intent or "knowledge_query",
                                Json(source_chunks_data) if Json else json.dumps(source_chunks_data),
                                Json(images_data) if Json else json.dumps(images_data),
                                Json(youtube_links_data) if Json else json.dumps(youtube_links_data),
                                Json(chips_data) if Json else json.dumps(chips_data),
                                Json(contact_data) if (contact_data and Json) else (json.dumps(contact_data) if contact_data else None)
                            ))
                    logger.info(f"Successfully saved chat interaction to Supabase (Session: {valid_session_uuid})")
                    return True
                except Exception as e:
                    logger.warning(f"Error persisting to Supabase ({e}), falling back to local SQLite.")
                    self._use_sqlite_fallback = True

            # Local SQLite Fallback
            with sqlite3.connect(str(self.sqlite_db_path)) as sconn:
                scur = sconn.cursor()
                scur.execute("""
                    INSERT INTO chat_sessions (id, user_identifier, role, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET updated_at = ?, role = excluded.role, title = COALESCE(chat_sessions.title, excluded.title);
                """, (valid_session_uuid, user_identifier or "anonymous_user", role, title, now_iso, now_iso, now_iso))

                scur.execute("""
                    INSERT INTO chat_messages (id, session_id, role, content, created_at)
                    VALUES (?, ?, 'user', ?, ?)
                """, (str(uuid.uuid4()), valid_session_uuid, user_query, now_iso))

                scur.execute("""
                    INSERT INTO chat_messages (
                        id, session_id, role, content, intent,
                        source_chunks, images, youtube_links,
                        quick_action_chips, contact_support, created_at
                    )
                    VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    valid_session_uuid,
                    bot_response.answer,
                    bot_response.intent or "knowledge_query",
                    json.dumps(source_chunks_data),
                    json.dumps(images_data),
                    json.dumps(youtube_links_data),
                    json.dumps(chips_data),
                    json.dumps(contact_data) if contact_data else None,
                    now_iso
                ))
                sconn.commit()
            logger.info(f"💾 Saved chat interaction to local SQLite (Session: {valid_session_uuid})")
            return True
        except Exception as e:
            logger.error(f"Failed to persist chat interaction: {e}")
            return False

    async def save_chat_interaction_async(
        self,
        session_id: str,
        user_query: str,
        bot_response: ChatResponse,
        role: str = "phuong",
        user_identifier: Optional[str] = None
    ):
        """Non-blocking async wrapper using asyncio.to_thread."""
        try:
            await asyncio.to_thread(
                self._save_interaction_sync,
                session_id=session_id,
                user_query=user_query,
                bot_response=bot_response,
                role=role,
                user_identifier=user_identifier
            )
        except Exception as e:
            logger.error(f"Error in background save_chat_interaction_async: {e}")

    def get_session_messages(self, session_id: str) -> List[dict]:
        """Fetch all messages for a session from Supabase, or fallback to local SQLite."""
        try:
            try:
                valid_session_uuid = str(uuid.UUID(session_id))
            except Exception:
                valid_session_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))

            conn = self._get_connection()
            if conn:
                try:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT id, role, content, intent, images, youtube_links, quick_action_chips, contact_support, created_at
                                FROM chat_messages
                                WHERE session_id = %s
                                ORDER BY created_at ASC;
                            """, (valid_session_uuid,))
                            rows = cur.fetchall()
                            messages = []
                            for r in rows:
                                created_iso = r[8].isoformat() if r[8] else None
                                messages.append({
                                    "id": str(r[0]),
                                    "role": r[1],
                                    "content": r[2],
                                    "intent": r[3],
                                    "images": r[4] or [],
                                    "youtube_links": r[5] or [],
                                    "quick_action_chips": r[6] or [],
                                    "contact_support": r[7],
                                    "created_at": created_iso,
                                    "timestamp": created_iso
                                })
                            return messages
                except Exception as e:
                    logger.warning(f"Error querying Supabase messages ({e}), using local SQLite.")
                    self._use_sqlite_fallback = True

            # Local SQLite fallback
            if self.sqlite_db_path.exists():
                with sqlite3.connect(str(self.sqlite_db_path)) as sconn:
                    scur = sconn.cursor()
                    scur.execute("""
                        SELECT id, role, content, intent, images, youtube_links, quick_action_chips, contact_support, created_at
                        FROM chat_messages
                        WHERE session_id = ?
                        ORDER BY created_at ASC;
                    """, (valid_session_uuid,))
                    rows = scur.fetchall()
                    messages = []
                    for r in rows:
                        messages.append({
                            "id": str(r[0]),
                            "role": r[1],
                            "content": r[2],
                            "intent": r[3],
                            "images": json.loads(r[4]) if r[4] else [],
                            "youtube_links": json.loads(r[5]) if r[5] else [],
                            "quick_action_chips": json.loads(r[6]) if r[6] else [],
                            "contact_support": json.loads(r[7]) if r[7] else None,
                            "created_at": r[8],
                            "timestamp": r[8]
                        })
                    return messages
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve session messages for {session_id}: {e}")
            return []

    def get_all_sessions(self, limit: int = 50) -> List[dict]:
        """Fetch latest chat sessions summary from Supabase, or fallback to local SQLite."""
        try:
            conn = self._get_connection()
            if conn:
                try:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT id, role, title, created_at, updated_at
                                FROM chat_sessions
                                ORDER BY updated_at DESC
                                LIMIT %s;
                            """, (limit,))
                            rows = cur.fetchall()
                            sessions = []
                            for r in rows:
                                sessions.append({
                                    "id": str(r[0]),
                                    "role": r[1] or "dn",
                                    "title": r[2] or "Cuộc hội thoại",
                                    "createdAt": r[3].isoformat() if r[3] else None,
                                    "updatedAt": r[4].isoformat() if r[4] else None,
                                    "messages": []
                                })
                            return sessions
                except Exception as e:
                    logger.warning(f"Error querying Supabase sessions ({e}), using local SQLite.")
                    self._use_sqlite_fallback = True

            # Local SQLite fallback
            if self.sqlite_db_path.exists():
                with sqlite3.connect(str(self.sqlite_db_path)) as sconn:
                    scur = sconn.cursor()
                    scur.execute("""
                        SELECT id, role, title, created_at, updated_at
                        FROM chat_sessions
                        ORDER BY updated_at DESC
                        LIMIT ?;
                    """, (limit,))
                    rows = scur.fetchall()
                    sessions = []
                    for r in rows:
                        sessions.append({
                            "id": str(r[0]),
                            "role": r[1] or "dn",
                            "title": r[2] or "Cuộc hội thoại",
                            "createdAt": r[3],
                            "updatedAt": r[4],
                            "messages": []
                        })
                    return sessions
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve sessions: {e}")
            return []


chat_history_service = ChatHistoryService()

import asyncio
import json
import time
import uuid
import re
from typing import AsyncGenerator, List, Tuple, Optional, Any, Dict
from app.core.config import settings
from app.models.chat import ChatRequest, ChatResponse, ContactSupportInfo
from app.models.chunk import DocumentChunk
from app.vectorstore.hybrid_retriever import HybridRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from app.core.guardrails import check_security_guardrails
from app.services.intent_service import intent_service
from app.services.intent_router import (
    intent_router,
    STRATEGY_PROCEDURAL_EXTRACTIVE,
    STRATEGY_TARGETED_QA
)
from app.services.suggestion_service import suggestion_service
from app.services.qwen_service import qwen_service
from app.services.cqr_service import cqr_service
from app.services.chat_history_service import chat_history_service
from app.core.audit_logger import audit_logger
from app.utils.prompts import format_context_for_prompt
from app.utils.prompt_templates import build_qa_system_prompt, FACTOID_NANSWER_TAG, STANDARD_ESCALATION_TEXT
from app.core.logger import logger

MODE_A_RETRIEVAL_TOP_K = 1
MODE_B_RETRIEVAL_TOP_K = 4


def clean_display_content(text: str) -> str:
    """Lọc sạch các thẻ tiêu đề nội bộ, từ khóa tìm kiếm, ghi chú AI và thẻ tên file ảnh [[IMG:...]]."""
    cleaned = re.sub(r"^###\s+[^\n]+\n+", "", text)
    cleaned = re.sub(r"\[Từ khóa tìm kiếm:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*):[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[(?:Ghi chú AI|Ghi chú dành cho Trợ lý ảo[^\]]*)\](?:\s*:\s*[^\n]+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\[IMG:[^\]]*\]\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_cdn_image_url(url: str) -> str:
    """Chuẩn hóa mọi image URL sang BASE_IMAGE_CDN_URL với filename an toàn."""
    if not url:
        return ""
    import urllib.parse
    clean_url = url.strip()
    if "/" in clean_url:
        filename = clean_url.split("/")[-1]
    else:
        filename = clean_url
    unquoted = urllib.parse.unquote(filename)
    safe_filename = urllib.parse.quote(unquoted)
    return f"{settings.effective_image_cdn_url}/{safe_filename}"


class RAGService:
    """
    Streamlined Hierarchical RAG Service (Two-Tier Architecture):
    - Mode A (procedural_extractive): Trả về trực tiếp Parent Chunk (< 50ms, bypass LLM, full hình ảnh & video).
    - Mode B (targeted_qa): Tìm kiếm trên Child Chunks, truyền Scoped Child Context (< 100 tokens) vào Qwen LLM.
    """

    def __init__(self):
        self.retrievers = {
            settings.CHROMA_COLLECTION_PHUONG: HybridRetriever(chroma_store=ChromaVectorStore(collection_name=settings.CHROMA_COLLECTION_PHUONG)),
            settings.CHROMA_COLLECTION_DN: HybridRetriever(chroma_store=ChromaVectorStore(collection_name=settings.CHROMA_COLLECTION_DN)),
        }
        self.default_collection = settings.DEFAULT_COLLECTION

    def get_retriever(self, collection_name: Optional[str] = None, role: Optional[str] = None) -> HybridRetriever:
        if role == "dn":
            collection_name = settings.CHROMA_COLLECTION_DN
        elif role == "phuong":
            collection_name = settings.CHROMA_COLLECTION_PHUONG
        elif not collection_name:
            collection_name = self.default_collection

        if collection_name not in self.retrievers:
            self.retrievers[collection_name] = HybridRetriever(chroma_store=ChromaVectorStore(collection_name=collection_name))
        return self.retrievers[collection_name]

    def _dispatch_save_history(
        self,
        session_id: str,
        user_query: str,
        answer: str,
        intent: str,
        role: str = "phuong",
        images: Optional[List[str]] = None,
        youtube_links: Optional[List[str]] = None,
        quick_action_chips: Optional[List[Any]] = None,
        contact_support: Optional[ContactSupportInfo] = None,
        source_chunks: Optional[List[Any]] = None
    ):
        """Dispatches non-blocking async persistence to Supabase."""
        try:
            resp = ChatResponse(
                session_id=session_id,
                answer=answer,
                intent=intent,
                images=images or [],
                youtube_links=youtube_links or [],
                quick_action_chips=quick_action_chips or [],
                contact_support=contact_support,
                source_chunks=[c.model_dump() if hasattr(c, "model_dump") else c for c in (source_chunks or [])]
            )
            asyncio.create_task(
                chat_history_service.save_chat_interaction_async(
                    session_id=session_id,
                    user_query=user_query,
                    bot_response=resp,
                    role=role
                )
            )
        except Exception as e:
            logger.error(f"Failed to dispatch chat history persistence: {e}")

    async def warmup(self):
        """Warms up embedding, vectorstores, and sparse index."""
        logger.info("Initializing RAGService warm-up...")
        for coll_name, retriever in self.retrievers.items():
            await retriever.warmup()
        await qwen_service.warmup()
        logger.info("RAGService warm-up completed successfully.")

    async def retrieve_context(
        self,
        query: str,
        target_module: Optional[str] = None,
        confidence_score: float = 1.0,
        strategy: str = STRATEGY_TARGETED_QA,
        top_k: int = 2,
        collection: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Truy xuất ngữ cảnh theo Mode:
        - Mode A (procedural_extractive): Lấy Parent Chunk (ưu tiên direct match qua module name).
        - Mode B (targeted_qa): Lấy Top 1-2 Child Chunks trúng đích.
        """
        retriever = self.get_retriever(collection_name=collection, role=role)

        if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE and target_module:
            parent_chunk = retriever.get_parent_chunk_by_module(target_module, query=query)
            if parent_chunk:
                return [parent_chunk]

        prefer_child = True if strategy == STRATEGY_TARGETED_QA else False
        chunks = await retriever.search(
            query=query,
            top_k=top_k,
            target_module=target_module,
            confidence_score=confidence_score,
            prefer_child=prefer_child
        )

        # Fallback: if prefer_child was True but no child chunks found, retrieve any matching chunks
        if not chunks and prefer_child:
            chunks = await retriever.search(
                query=query,
                top_k=top_k,
                target_module=target_module,
                confidence_score=confidence_score,
                prefer_child=None
            )

        return chunks

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Alias for process_chat_message."""
        return await self.process_chat_message(request)

    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        start_time = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())
        raw_query = request.query.strip()
        role = request.role or ("dn" if request.collection == settings.CHROMA_COLLECTION_DN else "phuong")
        contact_info = ContactSupportInfo.for_phuong() if role == "phuong" else ContactSupportInfo.for_dn()

        # 0. Conversational Context Resolution (CQR Đa Lượt)
        query, resolution_type = await cqr_service.resolve_context(
            query=raw_query,
            history=request.history,
            role=role
        )
        if resolution_type != "PASSTHROUGH":
            logger.info(f"🔄 [CQR Multi-turn] '{raw_query}' -> '{query}' (Type: {resolution_type})")

        logger.info(f"Processing chat message for session {session_id} (Role: {role}): '{query}'")

        try:
            # 1. Security Guardrails Check
            is_safe, refusal_reason = check_security_guardrails(query)
            if not is_safe:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=refusal_reason,
                    intent="security_violation",
                    status="REFUSED_SAFETY",
                    execution_time_ms=elapsed_ms
                )
                suggested_chips = suggestion_service.get_suggested_chips(query=query, role=role, limit=3)
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=refusal_reason,
                    intent="security_violation",
                    role=role,
                    quick_action_chips=suggested_chips
                )
                return ChatResponse(
                    session_id=session_id,
                    answer=refusal_reason,
                    intent="security_violation",
                    quick_action_chips=suggested_chips
                )

            # 2. Single-Pass Intent & Strategy Service Check
            intent_res = await intent_service.classify_intent_async(query, role=role)
            if intent_res.direct_answer:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=intent_res.direct_answer,
                    intent=intent_res.intent,
                    status="SUCCESS",
                    execution_time_ms=elapsed_ms
                )
                chips = intent_res.quick_action_chips or suggestion_service.get_suggested_chips(query=query, role=role, limit=3)
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=intent_res.direct_answer,
                    intent=intent_res.intent,
                    role=role,
                    quick_action_chips=chips
                )
                return ChatResponse(
                    session_id=session_id,
                    answer=intent_res.direct_answer,
                    quick_action_chips=chips,
                    intent=intent_res.intent
                )

            # 3. Contact Escalation / Software Error intent
            if intent_res.intent in ["contact_escalation", "software_error"]:
                answer = STANDARD_ESCALATION_TEXT
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=answer,
                    intent=intent_res.intent,
                    status="FALLBACK_SUPPORT",
                    execution_time_ms=elapsed_ms
                )
                suggested_chips = suggestion_service.get_suggested_chips(
                    query=query,
                    role=role,
                    target_module="LIÊN HỆ HỖ TRỢ",
                    limit=3
                )
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=answer,
                    intent=intent_res.intent,
                    role=role,
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips
                )
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips,
                    intent=intent_res.intent
                )

            # 4. Strategy & Target Module from Single-Pass Router
            strategy = intent_res.strategy
            target_module = intent_res.target_module
            confidence_score = intent_res.confidence_score

            chunks = await self.retrieve_context(
                query=query,
                target_module=target_module,
                confidence_score=confidence_score,
                strategy=strategy,
                top_k=MODE_B_RETRIEVAL_TOP_K if strategy == STRATEGY_TARGETED_QA else MODE_A_RETRIEVAL_TOP_K,
                collection=request.collection,
                role=role
            )

            # Collect media with strict deterministic order preservation
            image_urls = []
            youtube_links = []
            for chunk in chunks:
                if chunk.metadata.image_urls:
                    for img in chunk.metadata.image_urls:
                        normalized_img = normalize_cdn_image_url(img)
                        if normalized_img and normalized_img not in image_urls:
                            image_urls.append(normalized_img)
                if chunk.metadata.youtube_info:
                    yt_link = chunk.metadata.youtube_info.display_link
                    if yt_link not in youtube_links:
                        youtube_links.append(yt_link)

            # 5. Determine 3 Suggested Follow-up Action Chips
            suggested_chips = suggestion_service.get_suggested_chips(
                query=query,
                role=role,
                target_module=target_module,
                primary_chunk=chunks[0] if chunks else None,
                limit=3
            )

            is_contact_inquiry = target_module == "LIÊN HỆ HỖ TRỢ" or (chunks and chunks[0].metadata.section_title == "LIÊN HỆ HỖ TRỢ")

            # MODE A: Procedural Extractive Verbatim (< 50ms) -> Bypass LLM hoàn toàn, kèm Ảnh & Video
            if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE and chunks:
                if is_contact_inquiry:
                    answer = ""
                else:
                    primary_chunk = chunks[0]
                    raw_content = primary_chunk.text_content
                    cleaned_body = clean_display_content(raw_content)
                    intro = f"Dưới đây là hướng dẫn chi tiết quy trình **{primary_chunk.metadata.section_title}** trên hệ thống:\n\n"
                    answer = intro + cleaned_body

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=answer,
                    intent="knowledge_query_extractive",
                    status="SUCCESS",
                    execution_time_ms=elapsed_ms
                )
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=answer,
                    intent="knowledge_query",
                    role=role,
                    images=image_urls,
                    youtube_links=youtube_links,
                    quick_action_chips=suggested_chips,
                    contact_support=contact_info if is_contact_inquiry else None,
                    source_chunks=chunks
                )
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    intent="knowledge_query",
                    images=image_urls,
                    youtube_links=youtube_links,
                    quick_action_chips=suggested_chips,
                    contact_support=contact_info if is_contact_inquiry else None,
                    source_chunks=[c.model_dump() for c in chunks]
                )

            # MODE B: Targeted Generative QA via Qwen LLM với Scoped Child Context (< 100 tokens)
            context_data = format_context_for_prompt(chunks) if chunks else "Không có tài liệu phù hợp."
            qa_system_prompt = build_qa_system_prompt(role=role, context_data=context_data)

            # Ghi log chi tiết các chunks gửi đi cho LLM
            logger.info(
                f"🚀 [LLM Context Delivery - Non-Streaming] Sending {len(chunks)} chunks to Qwen LLM for query: '{query}'\n"
                + "\n".join([f"   [{i+1}] ID: {c.id} | Section: {c.metadata.section_title}\n       Preview: {c.text_content[:120].strip()}..." for i, c in enumerate(chunks)])
            )

            raw_answer = await qwen_service.generate_response(
                system_prompt=qa_system_prompt,
                user_query=query,
                history=request.history
            )

            # Dọn dẹp thẻ [VIDEO] nếu model lỡ sinh ra
            cleaned_answer = re.sub(r"\[VIDEO\]", "", raw_answer).strip()

            # Kiểm tra trường hợp Factoid_nanswer (Không có thông tin trong HDSD)
            if FACTOID_NANSWER_TAG.lower() in raw_answer.lower() or "[factoid_nanswer]" in raw_answer.lower() or not chunks:
                logger.info(f"Triggered Factoid_nanswer rejection for query: '{query}'")
                nanswer_text = STANDARD_ESCALATION_TEXT
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=nanswer_text,
                    intent="factoid_nanswer",
                    status="FALLBACK_SUPPORT",
                    execution_time_ms=elapsed_ms
                )
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=nanswer_text,
                    intent="factoid_nanswer",
                    role=role,
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips,
                    source_chunks=chunks
                )
                return ChatResponse(
                    session_id=session_id,
                    answer=nanswer_text,
                    intent="factoid_nanswer",
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips,
                    source_chunks=[chunk.model_dump() for chunk in chunks]
                )

            # Câu trả lời Factoid thành công (Text ngắn gọn, không kèm Video to)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            audit_logger.log_interaction(
                session_id=session_id,
                user_query=query,
                answer=cleaned_answer,
                intent="knowledge_query_targeted_qa",
                status="SUCCESS",
                execution_time_ms=elapsed_ms
            )

            self._dispatch_save_history(
                session_id=session_id,
                user_query=raw_query,
                answer=cleaned_answer,
                intent="knowledge_query",
                role=role,
                quick_action_chips=suggested_chips,
                contact_support=contact_info if is_contact_inquiry else None,
                source_chunks=chunks
            )

            return ChatResponse(
                session_id=session_id,
                answer=cleaned_answer,
                intent="knowledge_query",
                images=[],
                youtube_links=[],
                quick_action_chips=suggested_chips,
                contact_support=contact_info if is_contact_inquiry else None,
                source_chunks=[chunk.model_dump() for chunk in chunks]
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            fallback_answer = "Hệ thống đang tạm thời gián đoạn xử lý. Vui lòng thử lại sau giây lát hoặc liên hệ bộ phận hỗ trợ kỹ thuật:"
            audit_logger.log_interaction(
                session_id=session_id,
                user_query=query,
                answer=fallback_answer,
                intent="error_fallback",
                status="ERROR",
                execution_time_ms=elapsed_ms
            )
            self._dispatch_save_history(
                session_id=session_id,
                user_query=raw_query,
                answer=fallback_answer,
                intent="error_fallback",
                role=role,
                contact_support=contact_info
            )
            return ChatResponse(
                session_id=session_id,
                answer=fallback_answer,
                contact_support=contact_info,
                intent="error_fallback"
            )

    async def process_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Processes a chat query with Server-Sent Events (SSE) streaming."""
        start_time = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())
        raw_query = request.query.strip()
        role = request.role or ("dn" if request.collection == settings.CHROMA_COLLECTION_DN else "phuong")
        contact_info = ContactSupportInfo.for_phuong() if role == "phuong" else ContactSupportInfo.for_dn()
        accumulated_answer: List[str] = []

        # 0. Conversational Context Resolution (CQR Đa Lượt)
        query, resolution_type = await cqr_service.resolve_context(
            query=raw_query,
            history=request.history,
            role=role
        )
        if resolution_type != "PASSTHROUGH":
            logger.info(f"🔄 [CQR Multi-turn Streaming] '{raw_query}' -> '{query}' (Type: {resolution_type})")

        logger.info(f"Processing streaming chat for session {session_id} (Role: {role}): '{query}'")

        def sse_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # 1. Guardrails Check
            is_safe, guardrail_msg = check_security_guardrails(query)
            if not is_safe:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=guardrail_msg,
                    intent="guardrail_triggered",
                    status="BLOCKED",
                    execution_time_ms=elapsed_ms
                )
                yield sse_event("metadata", {
                    "session_id": session_id,
                    "intent": "guardrail_triggered"
                })
                yield sse_event("token", {"content": guardrail_msg})
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=guardrail_msg,
                    intent="guardrail_triggered",
                    role=role
                )
                yield sse_event("done", {
                    "session_id": session_id,
                    "full_answer": guardrail_msg
                })
                return

            # 2. Single-Pass Intent & Strategy Service Check
            intent_res = await intent_service.classify_intent_async(query, role=role)
            if intent_res.direct_answer:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=intent_res.direct_answer,
                    intent=intent_res.intent,
                    status="SUCCESS",
                    execution_time_ms=elapsed_ms
                )
                meta_payload = {
                    "session_id": session_id,
                    "intent": intent_res.intent
                }
                if intent_res.quick_action_chips:
                    meta_payload["quick_action_chips"] = [c.model_dump() for c in intent_res.quick_action_chips]

                yield sse_event("metadata", meta_payload)

                lines = intent_res.direct_answer.split("\n")
                for line in lines:
                    yield sse_event("token", {"content": line + "\n"})
                    await asyncio.sleep(0.005)

                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=intent_res.direct_answer,
                    intent=intent_res.intent,
                    role=role,
                    quick_action_chips=intent_res.quick_action_chips
                )

                yield sse_event("done", {
                    "session_id": session_id,
                    "full_answer": intent_res.direct_answer,
                    "metrics": {
                        "ttft_ms": 15.0,
                        "total_time_ms": elapsed_ms
                    }
                })
                return

            # 3. Contact Escalation / Software Error intent
            if intent_res.intent in ["contact_escalation", "software_error"]:
                answer = STANDARD_ESCALATION_TEXT
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=answer,
                    intent=intent_res.intent,
                    status="FALLBACK_SUPPORT",
                    execution_time_ms=elapsed_ms
                )
                suggested_chips = suggestion_service.get_suggested_chips(
                    query=query,
                    role=role,
                    target_module="LIÊN HỆ HỖ TRỢ",
                    limit=3
                )
                yield sse_event("metadata", {
                    "session_id": session_id,
                    "intent": intent_res.intent,
                    "contact_support": contact_info.model_dump(),
                    "quick_action_chips": [c.model_dump() for c in suggested_chips]
                })
                yield sse_event("token", {"content": answer})
                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=answer,
                    intent=intent_res.intent,
                    role=role,
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips
                )
                yield sse_event("done", {
                    "session_id": session_id,
                    "full_answer": answer
                })
                return

            # 4. Strategy & Target Module from Single-Pass Router
            strategy = intent_res.strategy
            target_module = intent_res.target_module
            confidence_score = intent_res.confidence_score

            chunks = await self.retrieve_context(
                query=query,
                target_module=target_module,
                confidence_score=confidence_score,
                strategy=strategy,
                top_k=MODE_B_RETRIEVAL_TOP_K if strategy == STRATEGY_TARGETED_QA else MODE_A_RETRIEVAL_TOP_K,
                collection=request.collection,
                role=role
            )

            # Collect media with strict deterministic order preservation
            image_urls = []
            youtube_links = []
            for chunk in chunks:
                if chunk.metadata.image_urls:
                    for img in chunk.metadata.image_urls:
                        normalized_img = normalize_cdn_image_url(img)
                        if normalized_img and normalized_img not in image_urls:
                            image_urls.append(normalized_img)
                if chunk.metadata.youtube_info:
                    yt_link = chunk.metadata.youtube_info.display_link
                    if yt_link not in youtube_links:
                        youtube_links.append(yt_link)

            # 5. Determine 3 Suggested Follow-up Action Chips
            suggested_chips = suggestion_service.get_suggested_chips(
                query=query,
                role=role,
                target_module=target_module,
                primary_chunk=chunks[0] if chunks else None,
                limit=3
            )

            is_contact_inquiry = target_module == "LIÊN HỆ HỖ TRỢ" or (chunks and chunks[0].metadata.section_title == "LIÊN HỆ HỖ TRỢ")

            # Phân biệt Metadata Media: Chỉ gửi Ảnh & Video khi là Mode A (Procedural)
            media_images = image_urls if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE else []
            media_videos = youtube_links if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE else []

            # Send initial Metadata event
            yield sse_event("metadata", {
                "session_id": session_id,
                "intent": "knowledge_query",
                "images": media_images,
                "youtube_links": media_videos,
                "quick_action_chips": [c.model_dump() for c in suggested_chips],
                "contact_support": contact_info.model_dump() if is_contact_inquiry else None,
                "source_chunks": [chunk.model_dump() for chunk in chunks]
            })

            # MODE A: Procedural Extractive Verbatim Stream (< 50ms) -> Trả về trực tiếp, bypass LLM
            if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE and chunks:
                if is_contact_inquiry:
                    final_answer = ""
                else:
                    primary_chunk = chunks[0]
                    raw_content = primary_chunk.text_content
                    cleaned_body = clean_display_content(raw_content)
                    intro = f"Dưới đây là hướng dẫn chi tiết quy trình **{primary_chunk.metadata.section_title}** trên hệ thống:\n\n"
                    full_text = intro + cleaned_body

                    lines = full_text.split("\n")
                    for line in lines:
                        accumulated_answer.append(line + "\n")
                        yield sse_event("token", {"content": line + "\n"})
                        await asyncio.sleep(0.008)

                    final_answer = "".join(accumulated_answer).strip()

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=final_answer,
                    intent="knowledge_query_extractive",
                    status="SUCCESS",
                    execution_time_ms=elapsed_ms
                )

                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=final_answer,
                    intent="knowledge_query",
                    role=role,
                    images=image_urls,
                    youtube_links=youtube_links,
                    quick_action_chips=suggested_chips,
                    contact_support=contact_info if is_contact_inquiry else None,
                    source_chunks=chunks
                )

                yield sse_event("done", {
                    "session_id": session_id,
                    "full_answer": final_answer,
                    "contact_support": contact_info.model_dump() if is_contact_inquiry else None,
                    "metrics": {
                        "ttft_ms": 20.0,
                        "total_time_ms": elapsed_ms
                    }
                })
                return

            # MODE B: Targeted Generative QA via Qwen LLM Stream với Scoped Child Context (< 100 tokens)
            context_data = format_context_for_prompt(chunks) if chunks else "Không có tài liệu phù hợp."
            qa_system_prompt = build_qa_system_prompt(role=role, context_data=context_data)

            # Ghi log chi tiết các chunks gửi đi cho LLM
            logger.info(
                f"🚀 [LLM Context Delivery - Streaming] Sending {len(chunks)} chunks to Qwen LLM for query: '{query}'\n"
                + "\n".join([f"   [{i+1}] ID: {c.id} | Section: {c.metadata.section_title}\n       Preview: {c.text_content[:120].strip()}..." for i, c in enumerate(chunks)])
            )

            ttft_ms = None
            is_refusal = False
            is_streaming_active = False
            prefix_buffer = ""
            accumulated_tokens: List[str] = []

            async for token in qwen_service.generate_stream(
                system_prompt=qa_system_prompt,
                user_query=query,
                history=request.history
            ):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start_time) * 1000

                accumulated_tokens.append(token)

                if not is_streaming_active:
                    prefix_buffer += token
                    # Kiểm tra dấu hiệu từ chối [Factoid_nanswer]
                    if FACTOID_NANSWER_TAG.lower() in prefix_buffer.lower() or "[factoid" in prefix_buffer.lower():
                        is_refusal = True
                        break
                    elif len(prefix_buffer) >= 20:
                        # Đã qua vùng an toàn (không phải refusal) -> Kích hoạt stream trực tiếp tức thì
                        is_streaming_active = True
                        cleaned_prefix = re.sub(r"\[VIDEO\]", "", prefix_buffer)
                        yield sse_event("token", {"content": cleaned_prefix})
                else:
                    # Stream thời gian thực trực tiếp từng token từ LLM
                    if "[VIDEO]" not in token:
                        yield sse_event("token", {"content": token})

            if not chunks:
                is_refusal = True

            # Kiểm tra trường hợp kết thúc câu trả lời ngắn (< 20 ký tự) chưa kích hoạt streaming
            if not is_streaming_active and not is_refusal:
                if FACTOID_NANSWER_TAG.lower() in prefix_buffer.lower() or "[factoid" in prefix_buffer.lower():
                    is_refusal = True
                else:
                    cleaned_prefix = re.sub(r"\[VIDEO\]", "", prefix_buffer).strip()
                    if cleaned_prefix:
                        yield sse_event("token", {"content": cleaned_prefix})

            # Xử lý trường hợp Factoid_nanswer (Từ chối)
            if is_refusal:
                logger.info(f"Triggered Factoid_nanswer stream rejection for query: '{query}'")
                nanswer_text = STANDARD_ESCALATION_TEXT
                yield sse_event("token", {"content": nanswer_text})

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audit_logger.log_interaction(
                    session_id=session_id,
                    user_query=query,
                    answer=nanswer_text,
                    intent="factoid_nanswer",
                    status="FALLBACK_SUPPORT",
                    execution_time_ms=elapsed_ms
                )

                self._dispatch_save_history(
                    session_id=session_id,
                    user_query=raw_query,
                    answer=nanswer_text,
                    intent="factoid_nanswer",
                    role=role,
                    contact_support=contact_info,
                    quick_action_chips=suggested_chips,
                    source_chunks=chunks
                )

                yield sse_event("done", {
                    "session_id": session_id,
                    "full_answer": nanswer_text,
                    "contact_support": contact_info.model_dump(),
                    "metrics": {
                        "ttft_ms": ttft_ms or elapsed_ms,
                        "total_time_ms": elapsed_ms
                    }
                })
                return

            raw_answer = "".join(accumulated_tokens).strip()
            cleaned_answer = re.sub(r"\[VIDEO\]", "", raw_answer).strip()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            audit_logger.log_interaction(
                session_id=session_id,
                user_query=query,
                answer=cleaned_answer,
                intent="knowledge_query_targeted_qa",
                status="SUCCESS",
                execution_time_ms=elapsed_ms
            )

            self._dispatch_save_history(
                session_id=session_id,
                user_query=raw_query,
                answer=cleaned_answer,
                intent="knowledge_query",
                role=role,
                quick_action_chips=suggested_chips,
                contact_support=contact_info if is_contact_inquiry else None,
                source_chunks=chunks
            )

            yield sse_event("done", {
                "session_id": session_id,
                "full_answer": cleaned_answer,
                "contact_support": contact_info.model_dump() if is_contact_inquiry else None,
                "metrics": {
                    "ttft_ms": ttft_ms or elapsed_ms,
                    "total_time_ms": elapsed_ms
                }
            })

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            fallback_answer = "Hệ thống đang tạm thời gián đoạn xử lý. Vui lòng thử lại sau giây lát hoặc liên hệ bộ phận hỗ trợ kỹ thuật:"
            self._dispatch_save_history(
                session_id=session_id,
                user_query=raw_query,
                answer=fallback_answer,
                intent="error_fallback",
                role=role,
                contact_support=contact_info
            )
            yield sse_event("token", {"content": fallback_answer})
            yield sse_event("done", {
                "session_id": session_id,
                "full_answer": fallback_answer,
                "contact_support": contact_info.model_dump(),
                "metrics": {"total_time_ms": elapsed_ms}
            })


rag_service = RAGService()

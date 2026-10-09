import os
from typing import List, AsyncGenerator, Optional
import httpx
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logger import logger
from app.models.chat import ChatMessage


class QwenService:
    def __init__(self):
        self.base_url = settings.QWEN_BASE_URL
        self.model_name = settings.QWEN_MODEL_NAME
        self.client: Optional[AsyncOpenAI] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> AsyncOpenAI:
        api_key = settings.effective_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("⚠️ Chưa cấu hình DASHSCOPE_API_KEY hoặc QWEN_API_KEY trong file `.env`.")
        
        if self.client is None:
            # Persistent HTTP/2 Session Pooling & Keep-Alive
            self._http_client = httpx.AsyncClient(
                http2=True,
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                    keepalive_expiry=120.0
                ),
                timeout=httpx.Timeout(60.0, connect=10.0)
            )
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                http_client=self._http_client
            )
            logger.info("Initialized Qwen AsyncOpenAI client with persistent HTTP/2 Session Pooling & Keep-Alive.")
        return self.client

    async def warmup(self):
        """Warm up connection pool to eliminate initial TLS/TCP handshake latency."""
        try:
            client = self._get_client()
            await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            logger.info("Qwen API connection pool warmed up successfully.")
        except Exception as e:
            logger.warning(f"Qwen API warmup ping skipped or failed: {e}")

    async def close(self):
        """Closes the underlying HTTP/2 client session gracefully."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.info("Closed Qwen HTTP/2 persistent client session.")
        self.client = None
        self._http_client = None

    async def generate_response(
        self,
        system_prompt: str,
        user_query: str,
        history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Calls Qwen API to generate complete response."""
        try:
            client = self._get_client()
        except ValueError as e:
            return str(e)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error calling Qwen API: {e}", exc_info=True)
            return f"❌ Đã xảy ra lỗi khi gọi Qwen API: {str(e)}"

    async def generate_stream(
        self,
        system_prompt: str,
        user_query: str,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[str, None]:
        """Streams response tokens from Qwen API."""
        try:
            client = self._get_client()
        except ValueError as e:
            yield str(e)
            return

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Stream error from Qwen API: {e}", exc_info=True)
            yield f"\n❌ Lỗi truyền luồng: {str(e)}"


qwen_service = QwenService()

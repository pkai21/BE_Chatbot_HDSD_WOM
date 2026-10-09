import re
import json
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from app.models.chunk import DocumentChunk, ChunkMetadata, YouTubeInfo
from app.vectorstore.chroma_store import ChromaVectorStore
from app.core.logger import logger


VI_NORMALIZATION_MAP = {
    r"\bkí\b": "ký", r"\bkì\b": "kỳ", r"\bkỉ\b": "kỷ", r"\bkĩ\b": "kỹ", r"\bkị\b": "kỵ",
    r"\blí\b": "lý", r"\blì\b": "lỳ", r"\blỉ\b": "lỷ", r"\blĩ\b": "lỹ", r"\blị\b": "lỵ",
    r"\bmí\b": "mỹ", r"\bmì\b": "mỳ", r"\bmỉ\b": "mỷ", r"\bmĩ\b": "mỹ", r"\bmị\b": "mỵ",
    r"\bsí\b": "sỹ", r"\bsì\b": "sỳ", r"\bsỉ\b": "sỷ", r"\bsĩ\b": "sỹ", r"\bsị\b": "sỵ",
    r"\btí\b": "tỷ", r"\btì\b": "tỳ", r"\btỉ\b": "tỷ", r"\btĩ\b": "tỹ", r"\btị\b": "tỵ",
    r"\bquí\b": "quý", r"\bquì\b": "quỳ", r"\bquỉ\b": "quỷ", r"\bquĩ\b": "quỹ", r"\bquị\b": "quỵ",
}


def normalize_vietnamese_text(text: str) -> str:
    """Normalizes common Vietnamese spelling variations (e.g. 'đăng kí' -> 'đăng ký')."""
    lower = text.lower()
    for pattern, repl in VI_NORMALIZATION_MAP.items():
        lower = re.sub(pattern, repl, lower)
    return lower


def tokenize_vi(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer with Vietnamese spelling normalization."""
    normalized = normalize_vietnamese_text(text)
    cleaned = re.sub(r"[^\w\s]", " ", normalized)
    return [w for w in cleaned.split() if w]


CONVERSATIONAL_STOPWORDS: List[str] = [
    "cách", "để", "trong", "app", "làm", "sao", "hướng", "dẫn", "cho",
    "tôi", "hỏi", "với", "như", "thế", "nào", "ở", "đâu", "phần", "mềm",
    "trên", "hệ", "thống", "vào", "được", "không", "giúp", "với"
]


def strip_conversational_noise(query: str) -> str:
    """Strips conversational filler tokens for BM25 search."""
    words = query.lower().split()
    cleaned_words = [w for w in words if w not in CONVERSATIONAL_STOPWORDS]
    cleaned_query = " ".join(cleaned_words)
    return cleaned_query if cleaned_query else query


class HybridRetriever:
    """
    Confidence-Weighted Dynamic Hybrid Retriever (Dense ChromaDB + BM25Okapi + Reciprocal Rank Fusion)
    Hỗ trợ tìm kiếm phân cấp 2 tầng (Parent Chunks cho Mode A vs Child Chunks cho Mode B).
    """

    def __init__(self, chroma_store: ChromaVectorStore):
        self.chroma_store = chroma_store
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[DocumentChunk] = []
        self._is_indexed = False

    def _ensure_bm25_index(self):
        """Loads all chunks from ChromaDB and initializes BM25 index."""
        if self._is_indexed:
            return

        try:
            all_items = self.chroma_store.collection.get()
            if not all_items or not all_items.get("ids"):
                return

            chunks = []
            for idx in range(len(all_items["ids"])):
                chunk_id = all_items["ids"][idx]
                doc_text = all_items["documents"][idx] if all_items.get("documents") else ""
                meta = all_items["metadatas"][idx] if all_items.get("metadatas") else {}

                raw_imgs = meta.get("image_urls", "[]")
                image_urls = json.loads(raw_imgs) if isinstance(raw_imgs, str) else raw_imgs

                raw_yt = meta.get("youtube_info")
                yt_dict = json.loads(raw_yt) if (isinstance(raw_yt, str) and raw_yt) else raw_yt
                youtube_info = YouTubeInfo(**yt_dict) if yt_dict else None

                raw_extra = meta.get("extra_attributes", "{}")
                extra_attrs = json.loads(raw_extra) if (isinstance(raw_extra, str) and raw_extra) else (raw_extra or {})

                metadata = ChunkMetadata(
                    module=meta.get("module"),
                    sub_module=meta.get("sub_module"),
                    section_title=meta.get("section_title", ""),
                    heading_level=meta.get("heading_level", 1),
                    document_source=meta.get("document_source", ""),
                    parent_id=meta.get("parent_id") or None,
                    section_breadcrumb=meta.get("section_breadcrumb") or None,
                    is_child=bool(meta.get("is_child", False)),
                    image_urls=image_urls,
                    youtube_info=youtube_info,
                    extra_attributes=extra_attrs
                )
                chunks.append(DocumentChunk(id=chunk_id, text_content=doc_text, metadata=metadata))

            self.corpus_chunks = chunks
            tokenized_corpus = [
                tokenize_vi(f"{c.text_content} {c.metadata.section_breadcrumb or ''} {' '.join(c.metadata.extra_attributes.get('keywords', []))}")
                for c in chunks
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_indexed = True
            logger.info(f"Initialized BM25 index with {len(chunks)} documents for collection '{self.chroma_store.collection_name}'")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}", exc_info=True)

    async def warmup(self):
        """Pre-indexes BM25 and triggers a dummy search to warm up ONNX/Embedding weights."""
        self._ensure_bm25_index()
        try:
            await self.chroma_store.search("warmup test", top_k=1)
            logger.info("HybridRetriever (ChromaDB + BM25) warmed up successfully.")
        except Exception as e:
            logger.warning(f"HybridRetriever warmup exception: {e}")

    async def search(
        self,
        query: str,
        top_k: int = 2,
        alpha: float = 0.3,
        target_module: Optional[str] = None,
        confidence_score: float = 1.0,
        prefer_child: Optional[bool] = None
    ) -> List[DocumentChunk]:
        """
        Confidence-Weighted Dynamic Hybrid Search:
        1. Dense Search (ChromaDB Vector Store).
        2. Sparse Search (BM25Okapi trên cleaned keywords).
        3. Reciprocal Rank Fusion (RRF).
        4. Confidence-Weighted Dynamic Boost:
           Score(d) = RRF_base(d) + (C_router * Max_RRF * I(d in TargetModule))
        5. Lọc và ưu tiên phân cấp (Child Chunks cho Mode B vs Parent Chunks cho Mode A).
        """
        norm_query = normalize_vietnamese_text(query)
        cleaned_sparse_query = strip_conversational_noise(norm_query)

        # 1. Dense Search (ChromaDB)
        dense_results = await self.chroma_store.search(query=norm_query, top_k=top_k * 3)

        # 2. Sparse Search (BM25) on cleaned action keywords
        self._ensure_bm25_index()
        sparse_results: List[Tuple[DocumentChunk, float]] = []
        if self.bm25 and self.corpus_chunks:
            tokens = tokenize_vi(cleaned_sparse_query)
            if tokens:
                scores = self.bm25.get_scores(tokens)
                max_bm25 = max(scores) if scores.any() and max(scores) > 0 else 1.0
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]
                sparse_results = [(self.corpus_chunks[i], float(scores[i] / max_bm25)) for i in top_indices if scores[i] > 0]

        # 3. Reciprocal Rank Fusion (RRF) with Score-Salience Weighting
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        k = 60

        for rank, chunk in enumerate(dense_results):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (alpha / (k + rank + 1))

        for rank, (chunk, norm_salience) in enumerate(sparse_results):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (((1.0 - alpha) / (k + rank + 1)) * (1.0 + norm_salience))

        # Also add any chunks from corpus that might match target_module
        if target_module:
            for chunk in self.corpus_chunks:
                chunk_map[chunk.id] = chunk
                if chunk.id not in rrf_scores:
                    rrf_scores[chunk.id] = 0.0

        if not rrf_scores:
            return []

        # 4. Confidence-Weighted Dynamic Boost
        # Base max RRF for dynamic scale
        max_rrf = max(rrf_scores.values()) if rrf_scores and max(rrf_scores.values()) > 0 else (1.0 / (k + 1))
        boost_weight = max(0.1, min(1.0, confidence_score)) * max_rrf * 10.0

        if target_module:
            norm_target = target_module.strip().lower()
            for chunk_id, chunk in chunk_map.items():
                sec_title = (chunk.metadata.section_title or "").strip().lower()
                mod_name = (chunk.metadata.module or "").strip().lower()
                breadcrumb = (chunk.metadata.section_breadcrumb or "").strip().lower()

                is_match = (
                    norm_target == sec_title or
                    norm_target == mod_name or
                    norm_target in sec_title or
                    norm_target in breadcrumb
                )

                if is_match:
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + boost_weight

        # 5. Phân tách theo Parent / Child Chunks nếu prefer_child được chỉ định
        candidate_ids = list(rrf_scores.keys())
        if prefer_child is True:
            # Mode B: Ưu tiên các Child Chunks
            child_ids = [cid for cid in candidate_ids if chunk_map[cid].metadata.is_child]
            if child_ids:
                candidate_ids = child_ids
        elif prefer_child is False:
            # Mode A: Ưu tiên các Parent Chunks
            parent_ids = [cid for cid in candidate_ids if not chunk_map[cid].metadata.is_child]
            if parent_ids:
                candidate_ids = parent_ids

        # Sort candidate IDs by merged RRF score
        sorted_ids = sorted(candidate_ids, key=lambda cid: rrf_scores[cid], reverse=True)
        if not sorted_ids:
            return []

        max_score = rrf_scores[sorted_ids[0]]

        # Dynamic Cutoff: Only keep secondary chunk if score is >= 60% of top score
        filtered_ids = [
            cid for cid in sorted_ids[:top_k]
            if rrf_scores[cid] >= 0.6 * max_score or cid == sorted_ids[0]
        ]

        result_chunks = [chunk_map[cid] for cid in filtered_ids if cid in chunk_map]
        logger.info(f"Hybrid search returned {len(result_chunks)} chunks for query '{query}' (Target: {target_module}, Conf: {confidence_score:.2f}, PreferChild: {prefer_child})")
        return result_chunks

    def get_parent_chunk_by_module(self, module_name: str, query: Optional[str] = None) -> Optional[DocumentChunk]:
        """
        Trích xuất trực tiếp Parent Chunk của một module cho Mode A (< 1ms).
        Tự động bỏ qua các heading/mục lục rỗng (< 100 ký tự) và phân giải nhánh con chính xác theo query.
        """
        self._ensure_bm25_index()
        norm_mod = module_name.strip().lower()
        candidates: List[DocumentChunk] = []

        for chunk in self.corpus_chunks:
            if not chunk.metadata.is_child:
                sec_title = (chunk.metadata.section_title or "").strip().lower()
                mod_name = (chunk.metadata.module or "").strip().lower()
                if norm_mod == sec_title or norm_mod == mod_name or norm_mod in sec_title:
                    # Bỏ qua các heading rỗng chỉ có vài chục ký tự
                    if len(chunk.text_content.strip()) > 100:
                        candidates.append(chunk)

        if not candidates:
            return None

        if len(candidates) == 1 or not query:
            return candidates[0]

        # Phân giải đa nhánh (ví dụ: Báo cáo TNLĐ vs Báo cáo ATVSLĐ) dựa trên độ trùng khớp từ khóa trong query
        norm_query = query.strip().lower()
        scored = []
        for c in candidates:
            sec_title = (c.metadata.section_title or "").strip().lower()
            keywords = [w for w in sec_title.replace("-", " ").replace(".", " ").split() if len(w) > 2]
            score = sum(1 for kw in keywords if kw in norm_query)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger
from app.models.intent import IntentResult, QuickActionChip


class DenseSemanticRouter:
    """
    Dense Semantic Router (Aurelio AI Pattern) sử dụng Sentence Transformer
    (mô hình 'settings.EMBEDDING_MODEL_NAME' - AITeamVN/Vietnamese_Embedding_v2).

    Quản lý các Anchor Utterances cho các ý định mơ hồ (Ambiguous Intents)
    và kích hoạt phản hồi làm rõ (Make-Clear) kèm Quick Action Chips.
    Tích hợp Discriminative Negative Filter để triệt tiêu hoàn toàn False Positives.
    """

    SIMILARITY_THRESHOLD: float = 0.76

    # Danh sách các câu Anchor mẫu đại diện cho ý định mơ hồ
    AMBIGUITY_ANCHORS: Dict[str, Dict[str, List[str]]] = {
        "phuong": {
            "tnld": [
                "báo cáo tai nạn lao động",
                "báo cáo tai nạn",
                "hướng dẫn tai nạn lao động",
                "khai báo tai nạn",
                "làm báo cáo tai nạn",
                "cho tôi báo cáo tai nạn",
                "cho tôi báo cáo tai nạn lao động",
                "tôi muốn xem báo cáo tai nạn lao động",
                "tôi muốn xem báo cáo tai nạn",
                "xem báo cáo tai nạn lao động",
                "quy trình báo cáo tai nạn",
                "báo cáo tnld",
                "tai nạn lao động"
            ],
            "account": [
                "tài khoản phường",
                "tài khoản phường xã",
                "thao tác tài khoản",
                "quản lý tài khoản",
                "hướng dẫn tài khoản phường",
                "thông tin tài khoản",
                "cho mình xem thông tin tài khoản"
            ]
        },
        "dn": {
            "tnld": [
                "báo cáo tai nạn lao động",
                "khai báo tai nạn",
                "báo cáo tai nạn lao động doanh nghiệp",
                "báo cáo định kỳ hoặc đột xuất tai nạn",
                "khai báo tai nạn lao động",
                "cho tôi báo cáo tai nạn lao động",
                "báo cáo tnld"
            ]
        }
    }

    # Bộ lọc loại trừ phân định (Discriminative Negative Keywords)
    # Nếu câu hỏi chứa bất kỳ từ nào dưới đây -> ĐÃ RÕ THỰC THỂ, KHÔNG kích hoạt mơ hồ
    DISCRIMINATIVE_FILTERS: Dict[str, List[str]] = {
        "tnld": [
            "định kỳ", "dinh ky", "đột xuất", "dot xuat",
            "sơ lược", "so luoc", "nạn nhân", "nan nhan",
            "nghề nghiệp", "nghe nghiep", "kích thước", "kich thuoc",
            "file", "tệp", "đính kèm", "dinh kem", "dung lượng", "dung luong",
            "tối đa", "toi da", "bao nhiêu", "bao nhieu", "hạn nộp", "han nop",
            "ở đâu", "o dau", "thời hạn", "thoi han"
        ],
        "account": [
            "tạo", "tao", "mới", "moi", "sửa", "sua",
            "chỉnh", "chinh", "khôi phục", "khoi phuc", "quên", "quen",
            "xóa", "xoa", "đổi mật khẩu", "doi mat khau", "mật khẩu", "mat khau",
            "thay đổi thông tin cá nhân", "thay doi thong tin", "cán bộ", "can bo",
            "cá nhân", "ca nhan"
        ]
    }

    def __init__(self):
        self._model = None
        self._is_warmed_up = False
        # Cached normalized embeddings: { "role:intent_key": (anchors_list, embeddings_matrix_2d) }
        self._anchor_embeddings: Dict[str, Tuple[List[str], np.ndarray]] = {}

    def _get_model(self):
        """Lazy load SentenceTransformer embedding model with local cache fast-path and HF token."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading DenseSemanticRouter embedding model: '{settings.EMBEDDING_MODEL_NAME}'")
            token = settings.HF_TOKEN if settings.HF_TOKEN else None
            try:
                # Fast path: load directly from local cache to bypass network HEAD requests and speed up startup (< 2s)
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    token=token,
                    model_kwargs={"local_files_only": True}
                )
            except Exception as e:
                logger.info(f"Local-only load bypassed ({e}), loading with HF Hub check...")
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, token=token)
        return self._model

    def warmup(self):
        """Pre-computes and caches normalized embeddings for all ambiguity anchors with instant disk cache."""
        if self._is_warmed_up:
            return

        from pathlib import Path
        cache_file = Path(settings.BACKEND_DIR) / "data" / "anchor_embeddings_cache.npz"

        # Fast-path 1: Instant load pre-computed vector matrices from .npz disk cache (< 20ms, 0 CPU compute)
        if cache_file.exists():
            try:
                loaded = np.load(str(cache_file))
                all_present = True
                for role, categories in self.AMBIGUITY_ANCHORS.items():
                    for cat, anchors in categories.items():
                        key = f"{role}:{cat}"
                        if key in loaded:
                            self._anchor_embeddings[key] = (anchors, loaded[key])
                        else:
                            all_present = False
                if all_present and len(self._anchor_embeddings) > 0:
                    self._is_warmed_up = True
                    logger.info(f"⚡ DenseSemanticRouter anchor cache loaded from '{cache_file.name}' (< 20ms).")
                    # Eagerly load model weights into RAM in background thread so user queries are instant
                    self._get_model()
                    logger.info("🔥 DenseSemanticRouter model loaded into RAM. Ready for instant inference.")
                    return
            except Exception as e:
                logger.warning(f"Failed to load anchor embeddings cache, will re-compute: {e}")

        # Fallback 2: Compute via model and persist cache
        try:
            model = self._get_model()
            cache_dict = {}
            for role, categories in self.AMBIGUITY_ANCHORS.items():
                for cat, anchors in categories.items():
                    key = f"{role}:{cat}"
                    vectors = model.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
                    self._anchor_embeddings[key] = (anchors, vectors)
                    cache_dict[key] = vectors
            self._is_warmed_up = True
            logger.info(f"DenseSemanticRouter warmed up successfully with {len(self._anchor_embeddings)} anchor groups.")
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(cache_file), **cache_dict)
                logger.info(f"💾 Saved pre-computed anchor vector cache to '{cache_file.name}'.")
            except Exception as e:
                logger.warning(f"Could not persist anchor cache: {e}")
        except Exception as e:
            logger.error(f"Error during DenseSemanticRouter warmup: {e}", exc_info=True)

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Encodes query and returns normalized 1D numpy vector."""
        try:
            model = self._get_model()
            vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            return vec[0]
        except Exception as e:
            logger.error(f"Error embedding query in DenseSemanticRouter: {e}")
            return None

    def match_ambiguity(self, query: str, role: str = "phuong") -> Optional[IntentResult]:
        """
        Kiểm tra câu hỏi người dùng có thuộc ý định mơ hồ hay không bằng Cosine Similarity.
        Nếu Cosine >= SIMILARITY_THRESHOLD và không chứa thực thể phân định -> Kích hoạt Make-Clear.
        """
        clean = query.strip().lower()
        if not self._is_warmed_up:
            self.warmup()

        query_vec = self._embed_query(clean)
        if query_vec is None:
            return None

        # 1. Kiểm tra nhóm Tai nạn lao động (TNLĐ)
        key_tnld = f"{role}:tnld"
        if key_tnld in self._anchor_embeddings:
            anchors, anchor_matrix = self._anchor_embeddings[key_tnld]
            # Cosine similarity for L2-normalized vectors is simply the dot product
            scores = np.dot(anchor_matrix, query_vec)
            max_idx = int(np.argmax(scores))
            max_score = float(scores[max_idx])

            has_discriminative = any(kw in clean for kw in self.DISCRIMINATIVE_FILTERS["tnld"])

            if max_score >= self.SIMILARITY_THRESHOLD and not has_discriminative:
                logger.info(
                    f"DenseSemanticRouter triggered TNLĐ Clarification: role={role} | "
                    f"score={max_score:.4f} >= {self.SIMILARITY_THRESHOLD} | matched_anchor='{anchors[max_idx]}'"
                )
                if role == "phuong":
                    return IntentResult(
                        intent="clarification_needed",
                        direct_answer="Bạn đang muốn tìm hiểu về quy trình **Báo cáo tai nạn lao động định kỳ** hay **Báo cáo tai nạn lao động đột xuất**?",
                        quick_action_chips=[
                            QuickActionChip(id="clarify_dotxuat", label="🚨 Báo cáo TNLĐ đột xuất", query_text="Hướng dẫn quy trình báo cáo tai nạn lao động đột xuất không theo HĐLĐ"),
                            QuickActionChip(id="clarify_dinhky", label="📊 Báo cáo TNLĐ định kỳ", query_text="Hướng dẫn quy trình báo cáo tai nạn lao động định kỳ cho người không có HĐLĐ")
                        ],
                        confidence_score=float(round(max_score, 4)),
                        matched_exemplar=f"semantic_router_{anchors[max_idx]}",
                        all_scores={"clarification_needed": float(round(max_score, 4))},
                        target_module=None,
                        strategy="targeted_qa"
                    )
                else:
                    return IntentResult(
                        intent="clarification_needed",
                        direct_answer="Bạn đang muốn tìm hiểu về **Báo cáo định kỳ tai nạn lao động** hay **Khai báo tai nạn lao động đột xuất**?",
                        quick_action_chips=[
                            QuickActionChip(id="clarify_dn_dotxuat", label="🚨 Khai báo TNLĐ đột xuất", query_text="Hướng dẫn khai báo tai nạn lao động đột xuất doanh nghiệp"),
                            QuickActionChip(id="clarify_dn_dinhky", label="📊 Báo cáo định kỳ TNLĐ", query_text="Hướng dẫn nộp báo cáo định kỳ tai nạn lao động doanh nghiệp")
                        ],
                        confidence_score=float(round(max_score, 4)),
                        matched_exemplar=f"semantic_router_{anchors[max_idx]}",
                        all_scores={"clarification_needed": float(round(max_score, 4))},
                        target_module=None,
                        strategy="targeted_qa"
                    )

        # 2. Kiểm tra nhóm Tài khoản (Account) cho role Phường/Xã
        if role == "phuong":
            key_acc = "phuong:account"
            if key_acc in self._anchor_embeddings:
                anchors_acc, anchor_matrix_acc = self._anchor_embeddings[key_acc]
                scores_acc = np.dot(anchor_matrix_acc, query_vec)
                max_idx_acc = int(np.argmax(scores_acc))
                max_score_acc = float(scores_acc[max_idx_acc])

                has_discriminative_acc = any(kw in clean for kw in self.DISCRIMINATIVE_FILTERS["account"])

                if max_score_acc >= self.SIMILARITY_THRESHOLD and not has_discriminative_acc:
                    logger.info(
                        f"DenseSemanticRouter triggered Account Clarification: role={role} | "
                        f"score={max_score_acc:.4f} >= {self.SIMILARITY_THRESHOLD} | matched_anchor='{anchors_acc[max_idx_acc]}'"
                    )
                    return IntentResult(
                        intent="clarification_needed",
                        direct_answer="Về phân hệ **Tài khoản Phường/xã**, bạn đang cần hướng dẫn thao tác nào dưới đây?",
                        quick_action_chips=[
                            QuickActionChip(id="clarify_p_create", label="➕ Tạo mới tài khoản", query_text="Hướng dẫn tạo mới tài khoản phường xã"),
                            QuickActionChip(id="clarify_p_edit", label="✏️ Chỉnh sửa tài khoản", query_text="Hướng dẫn chỉnh sửa tài khoản phường xã"),
                            QuickActionChip(id="clarify_p_reset", label="🔓 Khôi phục mật khẩu", query_text="Hướng dẫn khôi phục mật khẩu tài khoản phường xã"),
                            QuickActionChip(id="clarify_p_del", label="🗑️ Xóa tài khoản", query_text="Hướng dẫn xóa tài khoản phường xã")
                        ],
                        confidence_score=float(round(max_score_acc, 4)),
                        matched_exemplar=f"semantic_router_{anchors_acc[max_idx_acc]}",
                        all_scores={"clarification_needed": float(round(max_score_acc, 4))},
                        target_module=None,
                        strategy="targeted_qa"
                    )

        return None


# Global singleton instance
semantic_router = DenseSemanticRouter()

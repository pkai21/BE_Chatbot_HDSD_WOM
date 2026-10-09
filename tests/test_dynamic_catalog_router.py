import pytest
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.domain_registry import domain_registry
from app.utils.prompt_templates import build_router_system_prompt, build_qa_system_prompt
from app.services.intent_service import intent_service
from app.vectorstore.hybrid_retriever import HybridRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from app.core.config import settings


@pytest.mark.asyncio
async def test_domain_registry_catalogs():
    """Kiểm tra Domain Registry load chính xác catalogs cho dn và phuong."""
    dn_catalog = domain_registry.get_catalog("dn")
    assert len(dn_catalog) >= 7, f"Expected at least 7 modules for DN, got {len(dn_catalog)}"
    
    dn_modules = domain_registry.get_valid_modules("dn")
    assert "ĐĂNG KÝ" in dn_modules or "ĐĂNG NHẬP" in dn_modules
    assert "BÁO CÁO ĐỊNH KỲ" in dn_modules

    phuong_catalog = domain_registry.get_catalog("phuong")
    assert len(phuong_catalog) >= 10, f"Expected at least 10 modules for Phuong, got {len(phuong_catalog)}"
    
    phuong_modules = domain_registry.get_valid_modules("phuong")
    assert "ĐĂNG NHẬP" in phuong_modules
    assert "TỔNG QUAN CHỨC NĂNG TÀI KHOẢN PHƯỜNG/XÃ" in phuong_modules
    assert "BÁO CÁO TAI NẠN LAO ĐỘNG ĐỊNH KỲ KHÔNG THEO HĐLĐ" in phuong_modules


def test_scoped_prompt_templates():
    """Kiểm tra Prompt Templates được sinh động và scoped theo role."""
    prompt_dn = build_router_system_prompt("dn")
    assert "Doanh nghiệp" in prompt_dn
    assert "BÁO CÁO ĐỊNH KỲ" in prompt_dn

    prompt_phuong = build_router_system_prompt("phuong")
    assert "Phường/Xã" in prompt_phuong
    assert "TỔNG QUAN CHỨC NĂNG TÀI KHOẢN PHƯỜNG/XÃ" in prompt_phuong


@pytest.mark.asyncio
async def test_intent_router_confidence_and_routing():
    """Kiểm tra Intent Service trả về target_module, strategy và confidence_score."""
    # Test deterministic fast path
    res = await intent_service.classify_intent_async("Hotline hỗ trợ kỹ thuật số mấy?", role="phuong")
    assert res.intent == "knowledge_query"
    assert res.target_module == "LIÊN HỆ HỖ TRỢ"
    assert res.confidence_score >= 0.8

    # Test procedural fast path
    res_p = await intent_service.classify_intent_async("Tổng quan chức năng tài khoản phường xã", role="phuong")
    assert res_p.intent == "knowledge_query"
    assert res_p.target_module == "TỔNG QUAN CHỨC NĂNG TÀI KHOẢN PHƯỜNG/XÃ"
    assert res_p.strategy == "procedural_extractive"


@pytest.mark.asyncio
async def test_confidence_weighted_dynamic_boost():
    """Kiểm tra công thức Confidence-Weighted Dynamic Boost trong HybridRetriever."""
    retriever = HybridRetriever(chroma_store=ChromaVectorStore(collection_name=settings.CHROMA_COLLECTION_PHUONG))
    await retriever.warmup()

    # Search with target module and high confidence
    results = await retriever.search(
        query="Hướng dẫn đăng nhập hệ thống",
        top_k=2,
        target_module="ĐĂNG NHẬP",
        confidence_score=1.0
    )
    assert len(results) > 0
    top_chunk = results[0]
    sec = (top_chunk.metadata.section_title or "") + " " + (top_chunk.metadata.module or "")
    assert "đăng nhập" in sec.lower()

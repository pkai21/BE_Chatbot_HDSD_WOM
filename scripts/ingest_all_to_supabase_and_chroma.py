import os
import sys
import json
import asyncio
from pathlib import Path
import psycopg2
from psycopg2.extras import Json

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logger import logger
from app.ingestion.md_parser import md_parser
from app.ingestion.docx_parser import docx_parser
from app.ingestion.chunker import chunker
from app.ingestion.catalog_generator import catalog_generator
from app.vectorstore.chroma_store import ChromaVectorStore
from sentence_transformers import SentenceTransformer

# Load embedding model (1024 dimensions)
print("⏳ Loading embedding model:", settings.EMBEDDING_MODEL_NAME)
try:
    embedding_model = SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME,
        token=settings.HF_TOKEN if settings.HF_TOKEN else None,
        model_kwargs={"local_files_only": True}
    )
except Exception:
    embedding_model = SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME,
        token=settings.HF_TOKEN if settings.HF_TOKEN else None
    )

SUPABASE_CONN_STR = settings.DATABASE_URL or "postgresql://postgres:cEzQV7AuXRXnmxGb@db.tykwgiubhnxedlpxszdn.supabase.co:5432/postgres"

def ingest_to_supabase(chunks, role: str, doc_title: str):
    print(f"\n🐘 Đang đồng bộ {len(chunks)} chunks vào Supabase PostgreSQL (Table: document_chunks, Role: {role})...")
    conn = psycopg2.connect(SUPABASE_CONN_STR)
    cur = conn.cursor()
    
    # Ensure pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Ensure table exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id VARCHAR PRIMARY KEY,
        document_title VARCHAR NOT NULL,
        target_role VARCHAR NOT NULL,
        heading_hierarchy TEXT[],
        content TEXT NOT NULL,
        extracted_images JSONB DEFAULT '[]'::jsonb,
        youtube_ref JSONB DEFAULT '{}'::jsonb,
        step_keywords TEXT[],
        embedding vector(1024),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)
    conn.commit()

    # Prepare batch text for embedding
    texts = [c.text_content for c in chunks]
    print(f"  🧠 Đang tính toán 1024-dim dense embeddings cho {len(texts)} chunks...")
    vectors = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    upsert_query = """
    INSERT INTO document_chunks (
        id, document_title, target_role, heading_hierarchy, content, extracted_images, youtube_ref, step_keywords, embedding
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        document_title = EXCLUDED.document_title,
        target_role = EXCLUDED.target_role,
        heading_hierarchy = EXCLUDED.heading_hierarchy,
        content = EXCLUDED.content,
        extracted_images = EXCLUDED.extracted_images,
        youtube_ref = EXCLUDED.youtube_ref,
        step_keywords = EXCLUDED.step_keywords,
        embedding = EXCLUDED.embedding,
        created_at = NOW();
    """

    for i, c in enumerate(chunks):
        img_list = c.metadata.image_urls if (c.metadata and c.metadata.image_urls) else []
        yt_dict = c.metadata.youtube_info.model_dump() if (c.metadata and c.metadata.youtube_info) else {}
        vec = vectors[i].tolist()
        
        headings = []
        if c.metadata:
            if c.metadata.module:
                headings.append(c.metadata.module)
            if c.metadata.sub_module:
                headings.append(c.metadata.sub_module)
            if c.metadata.section_title:
                headings.append(c.metadata.section_title)
        
        keywords = []
        if c.metadata and c.metadata.extra_attributes:
            keywords = c.metadata.extra_attributes.get("keywords", [])

        cur.execute(upsert_query, (
            c.id,
            doc_title,
            role,
            headings,
            c.text_content,
            Json(img_list),
            Json(yt_dict),
            keywords,
            vec
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"  ✅ Đã lưu thành công {len(chunks)} chunks vào Supabase PostgreSQL!")


async def process_and_ingest_all():
    project_root = backend_dir.parent
    
    # 1. DOANH NGHIỆP (DN)
    dn_docx = project_root / "Bussiness_Rules" / "docs" / "_AI_HDSD_ATLĐ (DN).v1_HCM_2026 (1).docx"
    dn_md = project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_DN_v1_2026.md"
    dn_target = dn_docx if dn_docx.exists() else dn_md
    
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU NẠP DỮ LIỆU DOANH NGHIỆP (DN)")
    print("=" * 60)
    
    raw_dn_chunks = docx_parser.parse_docx(str(dn_target))
    dn_chunks = chunker.split_chunks(raw_dn_chunks)
    print(f"✅ Trích xuất {len(dn_chunks)} chunks cho Doanh nghiệp.")
    
    catalog_generator.generate_catalog_from_chunks(dn_chunks, role="dn", save_file=True)
    
    # ChromaDB
    chroma_dn = ChromaVectorStore(collection_name=settings.CHROMA_COLLECTION_DN)
    await chroma_dn.delete_collection()
    await chroma_dn.add_chunks(dn_chunks)
    print(f"✅ Đã nạp {len(dn_chunks)} chunks vào ChromaDB collection '{settings.CHROMA_COLLECTION_DN}'.")
    
    # Supabase
    ingest_to_supabase(dn_chunks, role="dn", doc_title="HDSD ATLĐ (Doanh nghiệp)")
    
    # 2. PHƯỜNG / XÃ (PHUONG)
    phuong_docx = project_root / "Bussiness_Rules" / "docs" / "HDSD ATLĐ (Phường).docx"
    phuong_md = project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_Phuong_v2_2026.md"
    phuong_target = phuong_docx if phuong_docx.exists() else phuong_md
    
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU NẠP DỮ LIỆU PHƯỜNG / XÃ (PHUONG)")
    print("=" * 60)
    
    raw_phuong_chunks = docx_parser.parse_docx(str(phuong_target))
    phuong_chunks = chunker.split_chunks(raw_phuong_chunks)
    print(f"✅ Trích xuất {len(phuong_chunks)} chunks cho Phường/Xã.")
    
    catalog_generator.generate_catalog_from_chunks(phuong_chunks, role="phuong", save_file=True)
    
    # ChromaDB
    chroma_phuong = ChromaVectorStore(collection_name=settings.CHROMA_COLLECTION_PHUONG)
    await chroma_phuong.delete_collection()
    await chroma_phuong.add_chunks(phuong_chunks)
    print(f"✅ Đã nạp {len(phuong_chunks)} chunks vào ChromaDB collection '{settings.CHROMA_COLLECTION_PHUONG}'.")
    
    # Supabase
    ingest_to_supabase(phuong_chunks, role="phuong", doc_title="HDSD ATLĐ (Phường/Xã)")

    print("\n🎉 TOÀN BỘ DỮ LIỆU ĐÃ ĐƯỢC NẠP THÀNH CÔNG VÀO CHROMADB & SUPABASE POSTGRESQL!")

if __name__ == "__main__":
    asyncio.run(process_and_ingest_all())

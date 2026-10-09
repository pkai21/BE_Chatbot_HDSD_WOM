import os
import sys
import asyncio
from pathlib import Path

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
from app.ingestion.pdf_parser import pdf_parser
from app.ingestion.chunker import chunker
from app.ingestion.catalog_generator import catalog_generator
from app.vectorstore.chroma_store import ChromaVectorStore


import argparse


async def ingest_collection(doc_paths: list, collection_name: str, doc_label: str, role: str):
    print("\n" + "=" * 60)
    print(f"🚀 BẮT ĐẦU NẠP DỮ LIỆU {doc_label.upper()} VÀO CHROMADB ({collection_name})")
    print("=" * 60)

    all_chunks = []
    for doc_path in doc_paths:
        if not doc_path.exists():
            print(f"⚠️ Không tìm thấy file: {doc_path}")
            continue

        print(f"\n📄 Đang xử lý tài liệu: {doc_path.name}")
        print(f"📁 Đường dẫn: {doc_path}")

        if doc_path.suffix.lower() == ".md":
            parsed_chunks = md_parser.parse_markdown(str(doc_path))
            parent_count = len([c for c in parsed_chunks if not c.metadata.is_child])
            child_count = len([c for c in parsed_chunks if c.metadata.is_child])
            print(f"  ✅ [Markdown 2 Tầng] Đã trích xuất {len(parsed_chunks)} chunks ({parent_count} Parent Chunks, {child_count} Child Chunks).")
            all_chunks.extend(parsed_chunks)
        elif doc_path.suffix.lower() in [".docx", ".doc"]:
            raw_chunks = docx_parser.parse_docx(str(doc_path))
            print(f"  ✅ Đã trích xuất {len(raw_chunks)} đoạn mục chính từ tài liệu docx.")
            refined_chunks = chunker.split_chunks(raw_chunks)
            print(f"  ✅ Đã chuẩn hóa thành {len(refined_chunks)} vector chunks (kèm metadata).")
            all_chunks.extend(refined_chunks)
        elif doc_path.suffix.lower() == ".pdf":
            raw_chunks = pdf_parser.parse_pdf(str(doc_path))
            print(f"  ✅ Đã trích xuất {len(raw_chunks)} vector chunks từ tài liệu PDF.")
            all_chunks.extend(raw_chunks)

    if not all_chunks:
        print(f"❌ Không có dữ liệu chunks nào cho {doc_label}.")
        return False

    # Tự động đồng bộ và sinh Domain Catalog JSON
    print(f"\n📋 Đang tự động tạo Domain Catalog cho phân hệ '{role}'...")
    catalog = catalog_generator.generate_catalog_from_chunks(all_chunks, role=role, save_file=True)
    print(f"  ✅ Đã cập nhật Domain Catalog ({len(catalog)} phân hệ chính thức).")

    print(f"\n🗄️ Khởi tạo ChromaDB collection '{collection_name}' tại: {settings.CHROMA_PERSIST_DIRECTORY}")
    chroma_store = ChromaVectorStore(collection_name=collection_name)
    print("  🔄 Đang làm sạch và cập nhật bộ chỉ mục ChromaDB...")
    await chroma_store.delete_collection()

    print(f"  📥 Đang nạp {len(all_chunks)} chunks vào ChromaDB...")
    success = await chroma_store.add_chunks(all_chunks)

    if success:
        print(f"🎉 NẠP THÀNH CÔNG COLLECTION '{collection_name}' ({len(all_chunks)} chunks)!")
    return success


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB collections")
    parser.add_argument("--mode", choices=["all", "dn", "phuong"], default="all", help="Select target collection to ingest")
    args = parser.parse_args()

    project_root = backend_dir.parent
    
    # Ưu tiên file Docx gốc để trích xuất đầy đủ media hình ảnh vào static CDN
    dn_docx = project_root / "Bussiness_Rules" / "docs" / "_AI_HDSD_ATLĐ (DN).v1_HCM_2026 (1).docx"
    dn_md = project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_DN_v1_2026.md"
    dn_target = dn_docx if dn_docx.exists() else dn_md

    phuong_docx = project_root / "Bussiness_Rules" / "docs" / "HDSD ATLĐ (Phường).docx"
    phuong_md = project_root / "Bussiness_Rules" / "docs" / "markdown" / "HDSD_Phuong_v2_2026.md"
    phuong_target = phuong_docx if phuong_docx.exists() else phuong_md

    if args.mode in ["all", "dn"]:
        await ingest_collection([dn_target], settings.CHROMA_COLLECTION_DN, "Doanh nghiệp (v1)", role="dn")

    if args.mode in ["all", "phuong"]:
        await ingest_collection([phuong_target], settings.CHROMA_COLLECTION_PHUONG, "Phường/Xã (v2)", role="phuong")


if __name__ == "__main__":
    asyncio.run(main())

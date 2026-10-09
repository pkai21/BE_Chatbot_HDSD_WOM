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
from app.ingestion.docx_parser import docx_parser
from app.ingestion.chunker import chunker
from app.vectorstore.chroma_store import ChromaVectorStore


async def run_ingestion_phuong():
    print("\n" + "=" * 65)
    print("🚀 BẮT ĐẦU NẠP DỮ LIỆU HDSD PHƯỜNG/XÃ (V2) VÀO CHROMADB")
    print("=" * 65)

    # 1. Locate Phường docx file
    project_root = backend_dir.parent
    phuong_doc_path = project_root / "Bussiness_Rules" / "docs" / "HDSD WOM v1.0.docx"

    if not phuong_doc_path.exists():
        print(f"❌ Không tìm thấy file tài liệu: {phuong_doc_path}")
        return

    print(f"\n📄 Đang xử lý tài liệu: {phuong_doc_path.name}")
    print(f"📁 Đường dẫn: {phuong_doc_path}")

    # 2. Parse docx
    raw_chunks = docx_parser.parse_docx(str(phuong_doc_path))
    print(f"  ✅ Đã trích xuất {len(raw_chunks)} phân hệ nghiệp vụ từ tài liệu HDSD Phường/Xã.")

    # 3. Refine / Chunking
    refined_chunks = chunker.split_chunks(raw_chunks)
    print(f"  ✅ Đã chuẩn hóa thành {len(refined_chunks)} vector chunks (kèm metadata & từ khóa).")

    if not refined_chunks:
        print("❌ Không có dữ liệu chunks nào để nạp vào ChromaDB.")
        return

    # 4. Initialize ChromaDB Vector Store for Phường
    collection_name = settings.CHROMA_COLLECTION_PHUONG
    print(f"\n🗄️ Khởi tạo ChromaDB Collection '{collection_name}' tại: {settings.CHROMA_PERSIST_DIRECTORY}")
    chroma_store = ChromaVectorStore(collection_name=collection_name)

    # Reset collection to ensure clean fresh index
    print("  🔄 Đang làm sạch và khởi tạo lại bộ chỉ mục collection...")
    await chroma_store.delete_collection()

    # 5. Add Chunks to ChromaDB
    print(f"  📥 Đang nạp {len(refined_chunks)} chunks vào ChromaDB...")
    success = await chroma_store.add_chunks(refined_chunks)

    if success:
        print("\n" + "=" * 65)
        print(f"🎉 NẠP DỮ LIỆU VÀO COLLECTION '{collection_name}' THÀNH CÔNG!")
        print("=" * 65)
        print(f"📊 Tổng số Chunks: {len(refined_chunks)}")

        # Count total images & keywords
        total_images = sum(len(c.metadata.image_urls) for c in refined_chunks)
        total_kws = sum(len(c.metadata.extra_attributes.get("keywords", [])) for c in refined_chunks)
        print(f"🖼️ Tổng số ảnh UI đã ánh xạ: {total_images} ảnh (lưu tại {settings.IMAGE_STORAGE_PATH})")
        print(f"🏷️ Tổng số từ khóa tìm kiếm đã trích xuất: {total_kws} từ khóa")

        # 6. Verify Test Queries
        print("\n🔍 CHẠY KIỂM THỬ TRUY VẤN (VERIFICATION TEST - PHÂN HỆ PHƯỜNG/XÃ):")
        test_queries = [
            "Làm sao để đổi mật khẩu tài khoản đang đăng nhập?",
            "Quy trình phường/xã khai báo vụ tai nạn lao động ngay khi mới xảy ra trên địa bàn",
            "Hướng dẫn tạo mới tài khoản cho phường xã tuyến dưới",
            "Cách cập nhật thông tin cá nhân của cán bộ?",
            "Hotline hỗ trợ kỹ thuật của phường xã"
        ]

        for q in test_queries:
            print(f"\n👉 Test Query: '{q}'")
            results = await chroma_store.search(query=q, top_k=2)
            for r_idx, chunk in enumerate(results, 1):
                kws = chunk.metadata.extra_attributes.get("keywords", [])
                print(f"   [{r_idx}] Module: {chunk.metadata.module} | Mục: {chunk.metadata.section_title}")
                print(f"       Ảnh đính kèm: {len(chunk.metadata.image_urls)} ảnh -> {chunk.metadata.image_urls[:2]}")
                print(f"       Từ khóa: {kws[:2]}")
                print(f"       Nội dung tóm tắt: {chunk.text_content[:120]}...\n")
    else:
        print(f"❌ Có lỗi xảy ra khi nạp chunks vào ChromaDB collection '{collection_name}'.")


if __name__ == "__main__":
    asyncio.run(run_ingestion_phuong())

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.models.chunk import DocumentChunk
from app.core.logger import logger

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "catalogs"


class CatalogGenerator:
    """
    Tự động hóa trích xuất danh mục phân hệ nghiệp vụ (Domain Catalog)
    từ danh sách Parent Chunks và lưu ra JSON Registry.
    """

    def __init__(self, catalog_dir: Optional[Path] = None):
        self.catalog_dir = catalog_dir or CATALOG_DIR
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    def generate_catalog_from_chunks(
        self,
        chunks: List[DocumentChunk],
        role: str,
        save_file: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất danh mục phân hệ từ các Parent Chunks (hoặc tất cả Chunks).
        """
        # Lấy tất cả parent chunks (hoặc các chunk độc lập nếu không có parent)
        parent_chunks = [c for c in chunks if not c.metadata.is_child]
        if not parent_chunks:
            parent_chunks = chunks

        modules_map: Dict[str, Dict[str, Any]] = {}

        for idx, p_chunk in enumerate(parent_chunks):
            mod_name = p_chunk.metadata.module or p_chunk.metadata.section_title
            if not mod_name:
                continue

            # Lấy mô tả ngắn từ nội dung chunk (1-2 câu đầu)
            raw_text = p_chunk.text_content
            clean_text = re.sub(r"^###\s+[^\n]+\n+", "", raw_text).strip()
            first_para = clean_text.split("\n\n")[0] if clean_text else ""
            desc = first_para[:200] + ("..." if len(first_para) > 200 else "")

            # Tìm các child chunks của parent này để lấy danh sách sub_modules
            p_id = p_chunk.id
            sub_chunks = [c for c in chunks if c.metadata.parent_id == p_id or (c.metadata.module == mod_name and c.metadata.is_child)]
            sub_modules = []
            for sc in sub_chunks:
                st = sc.metadata.sub_module or sc.metadata.section_title
                if st and st not in sub_modules and st != mod_name:
                    sub_modules.append(st)

            if mod_name not in modules_map:
                modules_map[mod_name] = {
                    "id": f"{role}_{len(modules_map) + 1:02d}",
                    "name": mod_name,
                    "description": desc,
                    "sub_modules": sub_modules,
                    "document_source": p_chunk.metadata.document_source
                }
            else:
                for sm in sub_modules:
                    if sm not in modules_map[mod_name]["sub_modules"]:
                        modules_map[mod_name]["sub_modules"].append(sm)

        catalog = list(modules_map.values())

        if save_file:
            self.save_catalog(catalog, role)

        logger.info(f"Generated catalog for role '{role}' with {len(catalog)} official modules")
        return catalog

    def save_catalog(self, catalog: List[Dict[str, Any]], role: str) -> str:
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.catalog_dir / f"domain_catalog_{role}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved domain catalog to {file_path}")
        return str(file_path)

    def load_catalog(self, role: str) -> List[Dict[str, Any]]:
        file_path = self.catalog_dir / f"domain_catalog_{role}.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading domain catalog from {file_path}: {e}")
            return []


catalog_generator = CatalogGenerator()

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.core.logger import logger

CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "catalogs"


class DomainRegistry:
    """
    Quản lý danh mục nghiệp vụ (Domain Catalog) động theo role (Doanh nghiệp vs Phường/Xã).
    Tự động load và cache các file JSON catalog được sinh ra từ Ingestion pipeline.
    """

    def __init__(self, catalog_dir: Optional[Path] = None):
        self.catalog_dir = catalog_dir or CATALOG_DIR
        self._catalogs: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all_catalogs()

    def _load_all_catalogs(self):
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        for role in ["dn", "phuong"]:
            file_path = self.catalog_dir / f"domain_catalog_{role}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._catalogs[role] = json.load(f)
                    logger.info(f"Loaded {len(self._catalogs[role])} modules from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {e}")
                    self._catalogs[role] = []
            else:
                self._catalogs[role] = []

    def reload(self):
        """Reloads all catalogs from disk."""
        self._load_all_catalogs()

    def get_catalog(self, role: str) -> List[Dict[str, Any]]:
        norm_role = "dn" if role == "dn" else "phuong"
        if norm_role not in self._catalogs or not self._catalogs[norm_role]:
            self._load_all_catalogs()
        return self._catalogs.get(norm_role, [])

    def get_valid_modules(self, role: str) -> List[str]:
        catalog = self.get_catalog(role)
        return [m["name"] for m in catalog if "name" in m]

    def get_all_valid_modules(self) -> List[str]:
        dn_mods = self.get_valid_modules("dn")
        p_mods = self.get_valid_modules("phuong")
        # Combine unique preserving order
        all_mods = []
        for m in p_mods + dn_mods:
            if m not in all_mods:
                all_mods.append(m)
        return all_mods

    def get_module_by_name(self, role: str, name: str) -> Optional[Dict[str, Any]]:
        catalog = self.get_catalog(role)
        norm_name = name.strip().lower()
        for m in catalog:
            if m.get("name", "").strip().lower() == norm_name:
                return m
        return None


domain_registry = DomainRegistry()

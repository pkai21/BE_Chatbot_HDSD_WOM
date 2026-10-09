import os
import pytest
from app.core.config import Settings, settings
from app.services.chat_history_service import ChatHistoryService
from app.services.rag_service import normalize_cdn_image_url


def test_development_profile_is_fully_self_contained():
    """TDD RED: In development, database defaults to None (local SQLite) and CDN points to localhost."""
    dev_settings = Settings(ENVIRONMENT="development", DATABASE_URL=None)
    
    # 1. Effective database url must be None for development when not explicitly overridden
    assert hasattr(dev_settings, "effective_database_url"), "Settings must have 'effective_database_url' property"
    assert dev_settings.effective_database_url is None

    # 2. Effective image cdn url must point to localhost static server
    assert hasattr(dev_settings, "effective_image_cdn_url"), "Settings must have 'effective_image_cdn_url' property"
    assert dev_settings.effective_image_cdn_url == f"http://localhost:{dev_settings.PORT}/static/images"

    # 3. ChatHistoryService initialized with dev profile should use SQLite directly
    service = ChatHistoryService()
    service.db_url = dev_settings.effective_database_url
    assert service._get_connection() is None, "In dev without cloud URL, should return None and use local SQLite"


def test_production_profile_uses_cloud_resources():
    """TDD RED: In production, database uses cloud URL and CDN points to Railway public domain."""
    prod_cloud_db = "postgresql://postgres:secret@db.cloud.supabase.co:5432/postgres"
    prod_settings = Settings(
        ENVIRONMENT="production",
        DATABASE_URL=prod_cloud_db,
        BASE_IMAGE_CDN_URL="https://chatbothdsd-production.up.railway.app/static/images"
    )

    # 1. Effective database url returns cloud database URL
    assert prod_settings.effective_database_url == prod_cloud_db

    # 2. Effective image cdn url returns Railway public domain
    assert prod_settings.effective_image_cdn_url == "https://chatbothdsd-production.up.railway.app/static/images"


def test_normalize_cdn_image_url_respects_environment_profile():
    """TDD RED: normalize_cdn_image_url uses effective_image_cdn_url."""
    # In development, normalized URL must point to localhost
    url = normalize_cdn_image_url("step1_login.png")
    assert "localhost" in url or settings.ENVIRONMENT == "production"

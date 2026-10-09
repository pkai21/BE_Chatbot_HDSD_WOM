import pytest
from pathlib import Path
from app.core.config import settings
from app.services.semantic_router import semantic_router


def test_settings_has_backend_dir_attribute():
    """TDD RED: settings must have BACKEND_DIR attribute as a valid Path."""
    assert hasattr(settings, "BACKEND_DIR"), "'Settings' object must have attribute 'BACKEND_DIR'"
    assert isinstance(settings.BACKEND_DIR, Path)
    assert settings.BACKEND_DIR.exists()


def test_semantic_router_warmup_no_attribute_error():
    """TDD RED: semantic_router.warmup() must not raise AttributeError."""
    semantic_router._is_warmed_up = False
    try:
        semantic_router.warmup()
    except AttributeError as e:
        pytest.fail(f"semantic_router.warmup() raised AttributeError: {e}")
    assert semantic_router._is_warmed_up is True

"""兼容转发层：后端 FastAPI 应用已独立迁移至 backend/app.py."""

from backend.app import app, create_app, local_app, public_app

__all__ = ["app", "create_app", "local_app", "public_app"]

import os
from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(os.path.join(BASE_DIR, ".env"))

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "django-insecure-quantellix-order-execution-portal-2026")

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "whitenoise.runserver_nostatic",
    "django.contrib.staticfiles",
    "portal.apps.PortalConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "quantellix_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.jinja2.Jinja2",
        "DIRS": [os.path.join(BASE_DIR, "portal", "templates")],
        "APP_DIRS": False,
        "OPTIONS": {
            "environment": "quantellix_backend.jinja2.environment",
            "autoescape": True,
        },
    },
]

WSGI_APPLICATION = "quantellix_backend.wsgi.application"
ASGI_APPLICATION = "quantellix_backend.asgi.application"

# Database: Handled via portal/database.py (Neon Cloud PostgreSQL connection pool)
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "portal", "static"),
]
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
WHITENOISE_USE_FINDERS = True

# Media files / Uploads
MEDIA_URL = "/uploads/"
MEDIA_ROOT = os.path.join(BASE_DIR, "portal", "uploads")

# 50MB Upload limit for CSV / Excel datasets
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800

# Agentic AI Feature Flags & Settings (Spec Section 0 & 12)
AGENTIC_ENABLED = os.getenv("AGENTIC_ENABLED", "False").lower() in ("true", "1", "yes")
AGENT_LLM_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "openai")
AGENT_LLM_MODEL_FAST = os.getenv("AGENT_LLM_MODEL_FAST", "gpt-4o-mini")
AGENT_LLM_MODEL_STRONG = os.getenv("AGENT_LLM_MODEL_STRONG", "gpt-4o-mini")
AGENT_TOKEN_BUDGET_PER_RUN = int(os.getenv("AGENT_TOKEN_BUDGET_PER_RUN", "40000"))
AGENT_CREDIT_COST = int(os.getenv("AGENT_CREDIT_COST", "0"))
AGENT_REANALYZE_RATE_LIMIT = int(os.getenv("AGENT_REANALYZE_RATE_LIMIT", "5"))
RUN_AGENTS_SYNC = os.getenv("RUN_AGENTS_SYNC", "True").lower() in ("true", "1", "yes")


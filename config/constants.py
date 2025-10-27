import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
from enum import Enum

# Load .env from project root (parent directory of ingest/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Environment variables
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")

# Storage configuration
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "textbook")
PDF_FILENAME = os.getenv("PDF_FILENAME")

# Validate required variables
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY not found in environment")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL not found in environment")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Models
HUGGING_FACE_HUB_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QA_EMBEDDING_MODEL = "text-embedding-3-small"

# Parameters
TARGET_CHARS = 1000
OVERLAP_CHARS = 150

MODEL_NAME = "mistral"
MODEL_QUANTIZATION = "4bit"
MODEL_TEMPERATURE = 0.4  # Lower = more focused, Higher = more creative
MODEL_MAX_TOKENS = 2048
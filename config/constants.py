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
if not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_ANON_KEY not found in environment")

def get_supabase():
    """Return a fresh Supabase client with service role key (bypasses RLS). Use for admin/CLI ops."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def get_supabase_for_user(jwt_token: str):
    """Return a Supabase client scoped to the authenticated user's JWT (respects RLS)."""
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(jwt_token)
    return client

supabase = get_supabase()

# Models
EMBEDDING_MODEL = "voyage-3-lite"

# Parameters
TARGET_CHARS = 1000
OVERLAP_CHARS = 150

MODEL_NAME = "mistral"
MODEL_QUANTIZATION = "4bit"
MODEL_TEMPERATURE = 0.4  # Lower = more focused, Higher = more creative
MODEL_MAX_TOKENS = 2048

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini"

# Anthropic / Claude
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
CLAUDE_MODEL_NAME = "claude-haiku-4-5"

# Voyage LLM API
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL_NAME = "voyage-3-lite"
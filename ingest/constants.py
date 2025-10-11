import os
from supabase import create_client

# Secrets
# WORKER_TOKEN = os.environ["WORKER_TOKEN"]

# DB Connection
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "textbook")
PDF_FILENAME = os.getenv("PDF_FILENAME")

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QA_EMBEDDING_MODEL = "text-embedding-3-small"
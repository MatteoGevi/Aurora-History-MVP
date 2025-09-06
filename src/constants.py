import os
from supabase import create_client

# Secrets
WORKER_TOKEN = os.environ["WORKER_TOKEN"]

# DB Connection
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
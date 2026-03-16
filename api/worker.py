# api/worker.py
import os, sys, tempfile, requests
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.constants import get_supabase, get_supabase_for_user

app = FastAPI()

# Service-role client for admin status updates (not subject to RLS)
_admin_sb = get_supabase()

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


def download_pdf(signed_url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        with requests.get(signed_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 15):
                if chunk:
                    tmp.write(chunk)
        return tmp.name


@app.post("/ingest")
def ingest(
    payload: dict,
    x_worker_token: str = Header(None),
    authorization: str = Header(None),
):
    if x_worker_token != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    doc_id = payload.get("docId")
    url = payload.get("signedUrl")
    user_id = payload.get("userId")
    if not doc_id or not url:
        raise HTTPException(status_code=400, detail="docId and signedUrl required")

    # Extract user JWT from Authorization: Bearer <jwt>
    user_jwt = None
    if authorization and authorization.lower().startswith("bearer "):
        user_jwt = authorization[7:]

    # Use user-scoped client so RLS policies apply to all writes
    sb = get_supabase_for_user(user_jwt) if user_jwt else _admin_sb

    _admin_sb.table("documents").update({"processing_status": "processing"}).eq("id", doc_id).execute()

    try:
        from ingest.ingest import ingest_document

        ingest_document(
            user_id=user_id,
            user_jwt=user_jwt,
            supabase_client=sb,
        )

        _admin_sb.table("documents").update(
            {"processing_status": "ready", "error_message": None}
        ).eq("id", doc_id).execute()
        return {"ok": True}
    except Exception as e:
        _admin_sb.table("documents").update(
            {"processing_status": "failed", "error_message": str(e)[:4000]}
        ).eq("id", doc_id).execute()
        raise
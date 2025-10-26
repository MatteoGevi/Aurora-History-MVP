# app/worker.py (FastAPI example)
import os, tempfile, requests
from fastapi import FastAPI, Header, HTTPException
from supabase import create_client
# import your existing pipeline function
# from aurora_history_mvp.ingest.pipeline import run_pipeline  # <- your code

app = FastAPI()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

def download_pdf(signed_url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        with requests.get(signed_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1<<15):
                if chunk: tmp.write(chunk)
        return tmp.name

@app.post("/ingest")
def ingest(payload: dict, x_worker_token: str = Header(None)):
    if x_worker_token != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    doc_id  = payload.get("docId")
    url     = payload.get("signedUrl")
    if not doc_id or not url:
        raise HTTPException(status_code=400, detail="docId and signedUrl required")

    # mark processing (just to be safe)
    sb.table("documents").update({"processing_status":"processing"}).eq("id", doc_id).execute()

    try:
        pdf_path = download_pdf(url)
        # >>> call YOUR existing chunking pipeline here <<<
        # It should write chunks/ToC into Postgres (or call helpers that do).
        # Example placeholder:
        # run_pipeline(pdf_path=pdf_path, doc_id=doc_id)
        #
        # If your pipeline currently saves files, adapt it to upsert DB rows instead.

        # done
        sb.table("documents").update({"processing_status":"ready", "error_message": None}).eq("id", doc_id).execute()
        return {"ok": True}
    except Exception as e:
        sb.table("documents").update({"processing_status":"failed", "error_message": str(e)[:4000]}).eq("id", doc_id).execute()
        raise
# ingest/ingest.py
from __future__ import annotations
from typing import List, Optional
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest.toc_chunk import (
    fetch_pdf_from_storage,
    build_toc,
    chunk_sections_with_hierarchy,
    flatten
)

from ingest.embedding_import import (
    generate_embeddings,
    validate_embeddings,
)

from config.constants import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    STORAGE_BUCKET,
    PDF_FILENAME,
    EMBEDDING_MODEL,
    TARGET_CHARS,
    OVERLAP_CHARS,
    supabase
)

def ingest_document(user_id: Optional[str] = None) -> str:
    """
    Complete end-to-end ingestion with document tracking.

    Args:
        user_id: UUID of user (None for MVP)

    Returns:
        document_id: UUID of created document record
    """
    if not PDF_FILENAME:
        raise ValueError("PDF_FILENAME must be set in .env file")

    print("=" * 80)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  PDF: {PDF_FILENAME}")
    print(f"  Bucket: {STORAGE_BUCKET}")
    print(f"  User ID: {user_id or 'None (MVP mode)'}")
    print(f"  Chunk size: {TARGET_CHARS}")
    print(f"  Chunk overlap: {OVERLAP_CHARS}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()

    # Step 1: Fetch PDF
    print("=" * 80)
    print("STEP 1/6: FETCHING PDF FROM SUPABASE STORAGE")
    print("=" * 80)
    pdf_bytes = fetch_pdf_from_storage(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
        STORAGE_BUCKET,
        PDF_FILENAME
    )

    # Step 2: Build ToC
    print("\n" + "=" * 80)
    print("STEP 2/6: BUILDING TABLE OF CONTENTS")
    print("=" * 80)
    doc_hash, toc, page_count = build_toc(pdf_bytes)
    toc_flat = flatten(toc)
    print(f"✅ Document hash: {doc_hash}")
    print(f"✅ Found {len(toc_flat)} sections across {page_count} pages")

    # Extract title from first H1 or use filename
    doc_title = toc[0]['title'] if toc and toc[0].get('level') == 1 else PDF_FILENAME

    # Step 3: Check if document already exists
    print("\n" + "=" * 80)
    print("STEP 3/6: CHECKING FOR EXISTING DOCUMENT")
    print("=" * 80)

    existing_doc = supabase.from_('documents') \
        .select('id, title, total_chunks') \
        .eq('doc_hash', doc_hash) \
        .execute()

    if existing_doc.data:
        print(f"⚠️  Document already exists!")
        print(f"   Title: {existing_doc.data[0]['title']}")
        print(f"   ID: {existing_doc.data[0]['id']}")
        print(f"   Chunks: {existing_doc.data[0]['total_chunks']}")

        response = input("\nRe-ingest? (y/n): ")
        if response.lower() != 'y':
            print("Ingestion cancelled.")
            return existing_doc.data[0]['id']

        # Delete existing document (cascades to toc_nodes and chunks)
        document_id = existing_doc.data[0]['id']
        print(f"\n🗑️  Deleting existing document...")
        supabase.from_('documents').delete().eq('id', document_id).execute()

    # Step 4: Create document record
    print("\n" + "=" * 80)
    print("STEP 4/6: CREATING DOCUMENT RECORD")
    print("=" * 80)

    document_record = {
        "user_id": user_id,
        "storage_path": PDF_FILENAME,
        "storage_bucket": STORAGE_BUCKET,
        "original_filename": PDF_FILENAME,
        "title": doc_title,
        "doc_hash": doc_hash,
        "total_pages": page_count,
        "total_sections": len(toc_flat),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": TARGET_CHARS,
        "chunk_overlap": OVERLAP_CHARS,
    }

    doc_response = supabase.table("documents").insert(document_record).execute()
    document_id = doc_response.data[0]['id']

    print(f"✅ Created document record")
    print(f"   Document ID: {document_id}")
    print(f"   Title: {doc_title}")

    try:
        # Step 5: Generate chunks
        print("\n" + "=" * 80)
        print("STEP 5/6: CHUNKING & EMBEDDING")
        print("=" * 80)
        chunks = chunk_sections_with_hierarchy(
            pdf_bytes,
            toc,
            chunk_size=TARGET_CHARS,
            chunk_overlap=OVERLAP_CHARS
        )
        print(f"✅ Generated {len(chunks)} chunks")

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(
            texts,
            model_name=EMBEDDING_MODEL,
            show_progress=True
        )
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")

        # Validate
        expected_dim = 384 if "small" in EMBEDDING_MODEL else 768
        passed, issues = validate_embeddings(embeddings, expected_dim=expected_dim)
        if not passed:
            print("⚠️  Validation warnings:", issues)

        # Step 6: Insert to database
        print("\n" + "=" * 80)
        print("STEP 6/6: INSERTING INTO DATABASE")
        print("=" * 80)

        # Insert ToC nodes
        print("📚 Inserting ToC nodes...")
        toc_records = [{
            "document_id": document_id,
            "node_id": n["id"],
            "title": n["title"],
            "level": n["level"],
            "page_start": n["page_start"],
            "page_end": n["page_end"]
        } for n in toc_flat]

        supabase.table("toc_nodes").insert(toc_records).execute()
        print(f"✅ Inserted {len(toc_records)} ToC nodes")

        # Get node_id -> db_id mapping
        result = supabase.table("toc_nodes") \
            .select("id, node_id") \
            .eq("document_id", document_id) \
            .execute()
        node_map = {r["node_id"]: r["id"] for r in result.data}

        # Insert chunks
        print(f"\n💾 Inserting {len(chunks)} chunks...")
        chunk_records = []
        for i, c in enumerate(chunks):
            chunk_records.append({
                "document_id": document_id,
                "chunk_id": f"{doc_hash}::{c['section_id']}::c{c['chunk_seq']:06d}",
                "toc_node_id": node_map.get(c["section_id"]),
                "chunk_seq": c["chunk_seq"],
                "section_title": c["section_title"],
                "level": c["level"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "text": c["text"],
                "embedding": embeddings[i].tolist()
            })

        BATCH_SIZE = 100
        total_batches = (len(chunk_records) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(chunk_records), BATCH_SIZE):
            batch = chunk_records[i:i+BATCH_SIZE]
            supabase.table("chunks").insert(batch).execute()
            batch_num = i // BATCH_SIZE + 1
            print(f"   ✓ Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        # Update document stats
        supabase.table("documents").update({
            "total_chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", document_id).execute()

    except Exception:
        print(f"\n❌ Ingestion failed — cleaning up partial document record...")
        supabase.from_('documents').delete().eq('id', document_id).execute()
        raise

    print("\n" + "=" * 80)
    print("🎉 INGESTION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Document ID: {document_id}")
    print(f"✅ Title: {doc_title}")
    print(f"✅ Total chunks: {len(chunks)}")
    print(f"\n💡 Use this document_id for retrieval!")

    return document_id


if __name__ == "__main__":
    # For MVP: no user_id
    user_id = None

    # For production: get from auth
    # user_id = get_authenticated_user_id()

    try:
        doc_id = ingest_document(user_id=user_id)
        print(f"\n✅ Success! Document ID: {doc_id}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

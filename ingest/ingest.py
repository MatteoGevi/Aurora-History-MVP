from __future__ import annotations
from typing import Tuple, List
import numpy as np
import json

# Import from your existing modules
from ingest.toc_chunk import (
    fetch_pdf_from_storage,
    build_toc,
    chunk_sections_with_hierarchy,
    flatten
)

from ingest.embedding_import import (
    generate_embeddings,
    validate_embeddings,
    ingest_to_supabase
)

# Import constants - everything we need is already there!
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


def ingest_document() -> str:
    """
    Complete end-to-end ingestion: Process PDF and insert into Supabase.
    Uses all configuration from constants.py (loaded from .env).
    
    Returns:
        doc_key: Unique document identifier
    """
    if not PDF_FILENAME:
        raise ValueError(
            "PDF_FILENAME must be set in .env file.\n"
            "Add: PDF_FILENAME=your-file.pdf"
        )
    
    print("=" * 80)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  PDF: {PDF_FILENAME}")
    print(f"  Bucket: {STORAGE_BUCKET}")
    print(f"  Chunk size: {TARGET_CHARS}")
    print(f"  Chunk overlap: {OVERLAP_CHARS}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()
    
    # Step 1: Fetch PDF
    print("=" * 80)
    print("STEP 1/5: FETCHING PDF FROM SUPABASE STORAGE")
    print("=" * 80)
    pdf_bytes = fetch_pdf_from_storage(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
        STORAGE_BUCKET,
        PDF_FILENAME
    )
    
    # Step 2: Build ToC
    print("\n" + "=" * 80)
    print("STEP 2/5: BUILDING TABLE OF CONTENTS")
    print("=" * 80)
    doc_key, toc, page_count = build_toc(pdf_bytes)
    toc_flat = flatten(toc)
    print(f"✅ Document key: {doc_key}")
    print(f"✅ Found {len(toc_flat)} sections across {page_count} pages")
    
    # Step 3: Generate chunks
    print("\n" + "=" * 80)
    print("STEP 3/5: CHUNKING DOCUMENT WITH HIERARCHY")
    print("=" * 80)
    chunks = chunk_sections_with_hierarchy(
        pdf_bytes,
        toc,
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS
    )
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Chunk statistics
    text_lengths = [len(c['text']) for c in chunks]
    print(f"\nChunk statistics:")
    print(f"  Min length: {min(text_lengths)} chars")
    print(f"  Max length: {max(text_lengths)} chars")
    print(f"  Mean length: {np.mean(text_lengths):.0f} chars")
    print(f"  Median length: {np.median(text_lengths):.0f} chars")
    
    # Step 4: Generate embeddings
    print("\n" + "=" * 80)
    print("STEP 4/5: GENERATING EMBEDDINGS")
    print("=" * 80)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(
        texts,
        model_name=EMBEDDING_MODEL,
        show_progress=True
    )
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    print(f"✅ Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Step 5: Validate embeddings
    print("\n" + "=" * 80)
    print("STEP 5/5: VALIDATING EMBEDDINGS")
    print("=" * 80)
    
    # Determine expected dimension based on model
    expected_dim = 384 if "small" in EMBEDDING_MODEL else (768 if "base" in EMBEDDING_MODEL else None)
    passed, issues = validate_embeddings(embeddings, expected_dim=expected_dim)
    
    if passed:
        print("✅ All validation checks passed!")
    else:
        print("⚠️  Validation warnings detected:")
        for issue in issues:
            print(f"   {issue}")
        
        # Check for critical issues (ones that should block ingestion)
        critical_issues = [i for i in issues if "❌" in i]
        if critical_issues:
            raise ValueError(f"Critical validation errors found: {critical_issues}")
        else:
            print("\n⚠️  Non-critical warnings detected, proceeding with ingestion...")
    
    # Insert into Supabase
    print("\n" + "=" * 80)
    print("INSERTING INTO SUPABASE")
    print("=" * 80)
    
    ingest_to_supabase(
        doc_key=doc_key,
        toc=toc,
        chunks=chunks,
        embeddings=embeddings,
        supabase_client=supabase
    )
    
    print("\n" + "=" * 80)
    print("🎉 INGESTION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Document '{PDF_FILENAME}' successfully ingested")
    print(f"✅ Document key: {doc_key}")
    print(f"✅ Total chunks inserted: {len(chunks)}")
    print(f"\n💡 You can now query this document using semantic search!")
    
    return doc_key


def analyze_document() -> dict:
    """
    Process and analyze document WITHOUT inserting into Supabase.
    Useful for testing chunking and embedding quality before ingestion.
    Saves analysis files for inspection.
    
    Returns:
        Dictionary with analysis results
    """
    if not PDF_FILENAME:
        raise ValueError("PDF_FILENAME must be set in .env file")
    
    print("=" * 80)
    print("DOCUMENT ANALYSIS (NO DATABASE INSERTION)")
    print("=" * 80)
    print(f"\nAnalyzing: {PDF_FILENAME}\n")
    
    # Step 1: Fetch PDF
    print("Step 1: Fetching PDF...")
    pdf_bytes = fetch_pdf_from_storage(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
        STORAGE_BUCKET,
        PDF_FILENAME
    )
    
    # Step 2: Build ToC
    print("\nStep 2: Building ToC...")
    doc_key, toc, page_count = build_toc(pdf_bytes)
    toc_flat = flatten(toc)
    print(f"✅ Found {len(toc_flat)} sections")
    
    # Save ToC
    toc_file = f"analysis_toc_{doc_key}.json"
    with open(toc_file, 'w', encoding='utf-8') as f:
        json.dump({"doc_key": doc_key, "toc": toc}, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved ToC to: {toc_file}")
    
    # Step 3: Generate chunks
    print("\nStep 3: Generating chunks...")
    chunks = chunk_sections_with_hierarchy(
        pdf_bytes,
        toc,
        chunk_size=TARGET_CHARS,
        chunk_overlap=OVERLAP_CHARS
    )
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Save chunks
    chunks_file = f"analysis_chunks_{doc_key}.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved chunks to: {chunks_file}")
    
    # Step 4: Generate embeddings
    print("\nStep 4: Generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(
        texts,
        model_name=EMBEDDING_MODEL,
        show_progress=True
    )
    
    # Step 5: Validate
    print("\nStep 5: Validating embeddings...")
    expected_dim = 384 if "small" in EMBEDDING_MODEL else (768 if "base" in EMBEDDING_MODEL else None)
    passed, issues = validate_embeddings(embeddings, expected_dim=expected_dim)
    
    if not passed:
        print("⚠️  Validation issues:")
        for issue in issues:
            print(f"   {issue}")
    
    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    # Text statistics
    text_lengths = [len(c['text']) for c in chunks]
    print(f"\n📊 Text Length Statistics:")
    print(f"   Min: {min(text_lengths)} chars")
    print(f"   Max: {max(text_lengths)} chars")
    print(f"   Mean: {np.mean(text_lengths):.0f} chars")
    print(f"   Median: {np.median(text_lengths):.0f} chars")
    
    # Similarity analysis
    print(f"\n🔗 Similarity Analysis:")
    sample_size = min(100, len(embeddings))
    sample_emb = embeddings[:sample_size]
    sim_matrix = sample_emb @ sample_emb.T
    np.fill_diagonal(sim_matrix, 0)
    
    print(f"   Sample size: {sample_size} chunks")
    print(f"   Min similarity: {sim_matrix.min():.4f}")
    print(f"   Max similarity: {sim_matrix.max():.4f}")
    print(f"   Mean similarity: {sim_matrix.mean():.4f}")
    
    # Check for duplicates
    near_duplicates = np.sum(sim_matrix > 0.95)
    exact_duplicates = np.sum(sim_matrix > 0.9999)
    print(f"   Near-duplicates (>0.95): {near_duplicates}")
    print(f"   Exact duplicates (>0.9999): {exact_duplicates}")
    
    if exact_duplicates > 0:
        print(f"\n   ⚠️  WARNING: Found {exact_duplicates} exact duplicate embeddings!")
    
    # Level distribution
    print(f"\n📈 Hierarchy Distribution:")
    from collections import Counter
    level_counts = Counter(c['level'] for c in chunks)
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = (count / len(chunks)) * 100
        print(f"   H{level}: {count:4d} chunks ({pct:5.1f}%)")
    
    # Save analysis summary
    analysis = {
        "doc_key": doc_key,
        "pdf_filename": PDF_FILENAME,
        "total_sections": len(toc_flat),
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": TARGET_CHARS,
        "chunk_overlap": OVERLAP_CHARS,
        "text_length_stats": {
            "min": int(min(text_lengths)),
            "max": int(max(text_lengths)),
            "mean": float(np.mean(text_lengths)),
            "median": float(np.median(text_lengths))
        },
        "similarity_stats": {
            "min": float(sim_matrix.min()),
            "max": float(sim_matrix.max()),
            "mean": float(sim_matrix.mean()),
            "near_duplicates": int(near_duplicates),
            "exact_duplicates": int(exact_duplicates)
        },
        "level_distribution": {f"H{k}": v for k, v in level_counts.items()},
        "validation_passed": passed,
        "validation_issues": issues
    }
    
    analysis_file = f"analysis_summary_{doc_key}.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved analysis summary to: {analysis_file}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Files saved:")
    print(f"   • {toc_file}")
    print(f"   • {chunks_file}")
    print(f"   • {analysis_file}")
    print(f"\n💡 Review the analysis, then run ingest_document() to insert into database")
    
    return analysis


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        print("🔍 Running in ANALYSIS-ONLY mode\n")
        try:
            result = analyze_document()
            print(f"\n✅ Analysis complete!")
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            raise
    else:
        print("🚀 Running FULL INGESTION\n")
        try:
            doc_key = ingest_document()
            print(f"\n✅ Ingestion complete! Doc key: {doc_key}")
        except Exception as e:
            print(f"\n❌ Error during ingestion: {e}")
            raise
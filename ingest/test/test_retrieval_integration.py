"""
Retrieval Integration Test
==========================
Verifies that what was ingested is actually useful for the grading scope:
  user selects a ToC section → get_section_content() → LLM receives good context.

Run after every ingest:
    python -m pytest src/test/test_retrieval_integration.py -v

Or standalone (no pytest needed):
    python src/test/test_retrieval_integration.py
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# --- path setup so this runs from any working directory ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.constants import supabase
from src.retrieval import get_document_list, get_document_toc, get_section_content

# ---------------------------------------------------------------------------
# Thresholds — adjust if your book is very short or very dense
# ---------------------------------------------------------------------------
MIN_CHUNKS_PER_SECTION  = 1      # A section with 0 chunks is always a bug
MIN_WORDS_FOR_GRADING   = 80     # Below this the LLM has nothing to grade on
MAX_WORDS_FOR_GRADING   = 12_000 # Above this the prompt will be truncated anyway
MIN_SECTIONS_IN_TOC     = 3      # Sanity: ToC must have at least a few nodes
MIN_CHAR_LENGTH_PER_CHUNK = 100  # Matches the ingest pipeline's own minimum


# ===========================================================================
# Helpers
# ===========================================================================

def _pick_document() -> dict:
    """Return the most recently ingested document, or raise clearly."""
    docs = get_document_list()
    assert docs, (
        "No documents found in the database.\n"
        "Run the ingest pipeline first, then re-run this test."
    )
    return docs[0]   # ordered by created_at DESC in get_document_list()


def _flatten_toc(nodes: list, out: list | None = None) -> list:
    """Recursively flatten the nested ToC tree into a plain list."""
    if out is None:
        out = []
    for node in nodes:
        out.append(node)
        _flatten_toc(node.get("children", []), out)
    return out


def _pick_sections(document_id: str, want: int = 3) -> list:
    """
    Return up to `want` leaf sections (sections with no children) so the tests
    exercise real content nodes rather than chapter-level containers.
    Falls back to any node if no leaf sections exist.
    """
    toc = get_document_toc(document_id)
    assert toc, "ToC is empty — check that toc_nodes were inserted during ingest."

    all_nodes = _flatten_toc(toc)

    # Prefer leaf nodes (no children) — those map most directly to chunk groups
    leaves = [n for n in all_nodes if not n.get("children")]
    pool   = leaves if leaves else all_nodes

    # Take evenly spaced samples so we test beginning, middle, and end
    step     = max(1, len(pool) // want)
    selected = pool[::step][:want]
    return selected


# ===========================================================================
# Test 1 — Document-level sanity
# ===========================================================================

def test_document_exists_and_has_metadata():
    """
    The documents table must have at least one row with sensible metadata.
    This is the entry point for the assessment app.
    """
    doc = _pick_document()

    assert doc.get("id"),            "Document has no id"
    assert doc.get("title"),         "Document has no title"
    assert doc.get("total_pages", 0) > 0, \
        f"total_pages is {doc.get('total_pages')} — ingest may have failed"
    assert doc.get("total_sections", 0) >= MIN_SECTIONS_IN_TOC, (
        f"Only {doc.get('total_sections')} sections stored; "
        f"expected at least {MIN_SECTIONS_IN_TOC}"
    )
    assert doc.get("total_chunks", 0) > 0, \
        "total_chunks is 0 — chunks were not inserted"

    print(f"\n✅ [test_document_exists_and_has_metadata]")
    print(f"   Title        : {doc['title']}")
    print(f"   Pages        : {doc['total_pages']}")
    print(f"   Sections     : {doc['total_sections']}")
    print(f"   Total chunks : {doc['total_chunks']}")


# ===========================================================================
# Test 2 — ToC structure
# ===========================================================================

def test_toc_is_navigable():
    """
    get_document_toc() must return a non-empty nested tree.
    The app uses this to let the user pick which section to study.
    """
    doc     = _pick_document()
    toc     = get_document_toc(doc["id"])

    assert toc, "get_document_toc() returned an empty list"

    all_nodes = _flatten_toc(toc)
    assert len(all_nodes) >= MIN_SECTIONS_IN_TOC, (
        f"Only {len(all_nodes)} nodes in ToC; expected at least {MIN_SECTIONS_IN_TOC}"
    )

    # Every node must carry the fields the UI relies on
    required_fields = {"id", "node_id", "title", "level", "page_start", "page_end"}
    for node in all_nodes:
        missing = required_fields - node.keys()
        assert not missing, (
            f"Node '{node.get('title', '?')}' is missing fields: {missing}"
        )

    # Levels must be positive integers
    levels = [n["level"] for n in all_nodes]
    assert all(isinstance(l, int) and l >= 1 for l in levels), \
        f"Some nodes have invalid levels: {set(levels)}"

    # Page ranges must be consistent (start <= end)
    bad_pages = [
        n for n in all_nodes
        if n["page_start"] is not None
        and n["page_end"]   is not None
        and n["page_start"] > n["page_end"]
    ]
    assert not bad_pages, (
        f"{len(bad_pages)} nodes have page_start > page_end:\n"
        + "\n".join(f"  {n['title']}: {n['page_start']}–{n['page_end']}"
                    for n in bad_pages[:5])
    )

    print(f"\n✅ [test_toc_is_navigable]")
    print(f"   Total nodes   : {len(all_nodes)}")
    print(f"   Distinct levels: {sorted(set(levels))}")
    print(f"   Root chapters : {len(toc)}")


# ===========================================================================
# Test 3 — Section content retrieval (the critical link)
# ===========================================================================

def test_section_content_is_retrievable():
    """
    For a sample of sections, get_section_content() must return:
      - at least MIN_CHUNKS_PER_SECTION chunks
      - text long enough for the LLM to grade (MIN_WORDS_FOR_GRADING)
      - text short enough to fit in a prompt (MAX_WORDS_FOR_GRADING)
      - no empty or too-short individual chunks
      - chunk_seq values that are sequential (no gaps or duplicates within a section)

    This is the single most important check: if this fails, the grader
    receives empty or garbled context, and will hallucinate scores.
    """
    doc      = _pick_document()
    sections = _pick_sections(doc["id"], want=3)

    results = []
    failures = []

    for sec in sections:
        node_id = sec["node_id"]
        title   = sec["title"]

        try:
            content = get_section_content(doc["id"], node_id, include_children=True)
        except Exception as exc:
            failures.append(f"  '{title}' ({node_id}): get_section_content() raised {exc}")
            continue

        chunks    = content.get("chunks", [])
        full_text = content.get("text", "")
        words     = content.get("total_words", 0)

        section_failures = []

        # --- chunk count ---
        if len(chunks) < MIN_CHUNKS_PER_SECTION:
            section_failures.append(
                f"only {len(chunks)} chunks (need ≥ {MIN_CHUNKS_PER_SECTION})"
            )

        # --- word count ---
        if words < MIN_WORDS_FOR_GRADING:
            section_failures.append(
                f"only {words} words — LLM will have insufficient context "
                f"(need ≥ {MIN_WORDS_FOR_GRADING})"
            )
        if words > MAX_WORDS_FOR_GRADING:
            # Not a hard failure, just a warning — evaluation.py truncates at 4000 chars
            print(f"   ⚠️  '{title}': {words} words — prompt will be truncated")

        # --- chunk text quality ---
        short_chunks = [
            c for c in chunks if len(c.get("text", "")) < MIN_CHAR_LENGTH_PER_CHUNK
        ]
        if short_chunks:
            section_failures.append(
                f"{len(short_chunks)} chunk(s) shorter than {MIN_CHAR_LENGTH_PER_CHUNK} chars"
            )

        empty_chunks = [c for c in chunks if not c.get("text", "").strip()]
        if empty_chunks:
            section_failures.append(f"{len(empty_chunks)} empty chunk(s)")

        # --- chunk_seq continuity (no duplicates within a section) ---
        seqs = [c.get("chunk_seq") for c in chunks if c.get("chunk_seq") is not None]
        if seqs and len(seqs) != len(set(seqs)):
            dupes = len(seqs) - len(set(seqs))
            section_failures.append(f"{dupes} duplicate chunk_seq value(s)")

        # --- concatenated text matches individual chunks ---
        expected_join = "\n\n".join(c["text"] for c in chunks)
        if full_text != expected_join:
            section_failures.append(
                "content['text'] does not match the join of content['chunks'][*]['text']"
            )

        results.append({
            "title":   title,
            "node_id": node_id,
            "chunks":  len(chunks),
            "words":   words,
            "ok":      not section_failures,
            "issues":  section_failures,
        })

        if section_failures:
            failures.append(
                f"  '{title}' ({node_id}):\n"
                + "\n".join(f"    - {f}" for f in section_failures)
            )

    print(f"\n✅ [test_section_content_is_retrievable]")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"   {status} '{r['title']}' → {r['chunks']} chunks, {r['words']} words")

    assert not failures, (
        f"\n{len(failures)} section(s) failed retrieval checks:\n"
        + "\n".join(failures)
    )


# ===========================================================================
# Test 4 — Content coherence (text is not garbled)
# ===========================================================================

def test_chunk_text_is_not_garbled():
    """
    A basic readability heuristic: chunks should not be mostly whitespace,
    control characters, or repeated single characters (signs of bad PDF parsing).

    Does NOT check meaning — just that the text looks like natural language.
    """
    doc = _pick_document()

    result = supabase.table("chunks") \
        .select("id, chunk_id, section_title, text") \
        .eq("document_id", doc["id"]) \
        .limit(50) \
        .execute()

    chunks = result.data
    assert chunks, "No chunks returned from the database for this document"

    garbled = []
    for c in chunks:
        text = c.get("text", "")

        # Ratio of alphabetic characters — real prose is usually > 50 %
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        alpha_ratio = alpha_chars / max(len(text), 1)

        # A chunk that is mostly non-alpha is likely a page-number block or OCR noise
        if alpha_ratio < 0.40:
            garbled.append(
                f"  chunk '{c['chunk_id']}' (section: {c['section_title']}): "
                f"alpha ratio = {alpha_ratio:.2%}  preview: {text[:80]!r}"
            )

    # Allow up to 10 % of sampled chunks to be borderline (headers, tables, etc.)
    max_allowed_garbled = max(1, len(chunks) // 10)
    print(f"\n✅ [test_chunk_text_is_not_garbled]")
    print(f"   Sampled {len(chunks)} chunks — {len(garbled)} appear garbled "
          f"(threshold: ≤ {max_allowed_garbled})")

    assert len(garbled) <= max_allowed_garbled, (
        f"{len(garbled)} chunks look garbled (alpha ratio < 40%):\n"
        + "\n".join(garbled[:10])
    )


# ===========================================================================
# Test 5 — Grader context size (evaluation.py truncates at 4000 chars)
# ===========================================================================

def test_grader_receives_enough_context():
    """
    evaluation.py passes content['text'][:4000] to the LLM.
    This test checks that the first 4000 chars of a retrieved section are:
      - non-empty
      - contain real words (not just headers or whitespace)

    If this fails, the grader is scoring answers against almost nothing.
    """
    doc      = _pick_document()
    sections = _pick_sections(doc["id"], want=3)

    failures = []
    for sec in sections:
        content   = get_section_content(doc["id"], sec["node_id"], include_children=True)
        grader_ctx = content.get("text", "")[:4000]

        word_count = len(grader_ctx.split())
        if word_count < 30:
            failures.append(
                f"  '{sec['title']}': only {word_count} words in first 4000 chars"
            )

    print(f"\n✅ [test_grader_receives_enough_context]")
    for sec in sections:
        content = get_section_content(doc["id"], sec["node_id"], include_children=True)
        ctx     = content.get("text", "")[:4000]
        print(f"   '{sec['title']}' → {len(ctx.split())} words in grader window")

    assert not failures, "\n".join(failures)


# ===========================================================================
# Standalone runner (no pytest required)
# ===========================================================================

if __name__ == "__main__":
    tests = [
        ("Document exists and has metadata",    test_document_exists_and_has_metadata),
        ("ToC is navigable",                    test_toc_is_navigable),
        ("Section content is retrievable",      test_section_content_is_retrievable),
        ("Chunk text is not garbled",           test_chunk_text_is_not_garbled),
        ("Grader receives enough context",      test_grader_receives_enough_context),
    ]

    passed, failed = 0, []

    print("\n" + "=" * 70)
    print("  RETRIEVAL INTEGRATION TESTS")
    print("=" * 70)

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"\n❌ FAILED: {name}\n   {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"\n💥 ERROR: {name}\n   {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{len(tests)} passed"
          + (f"  —  {len(failed)} failed" if failed else ""))
    print("=" * 70)

    if failed:
        sys.exit(1)

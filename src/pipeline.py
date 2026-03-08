from __future__ import annotations

from typing import Dict

from src.retrieval import get_section_content
from src.guardrailed_grader import grade_with_guardrails, Grade, MAX_SCORE

# The fixed recall prompt shown to the grader as the "question".
# The user never sees this — it's the grader's framing for what a good answer looks like.
_RECALL_PROMPT = (
    "Describe the key content, main concepts, and important details of this section "
    "as thoroughly as you can from memory."
)


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate at a paragraph or sentence boundary, not mid-sentence."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut == -1:
        cut = text.rfind(". ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut] + "\n\n[...content truncated...]"


def _make_ollama_adapter():
    """
    Returns a callable (system: str, user: str) -> str
    that guardrailed_grader expects as `call_llm`.
    Uses Ollama's chat API for proper system/user role separation.
    """
    from src.models import generate_chat

    def call_llm(system: str, user: str) -> str:
        return generate_chat(system, user, max_tokens=2500, temperature=0.2)

    return call_llm


def _make_openai_adapter():
    """
    Returns a callable (system: str, user: str) -> str backed by the OpenAI API.
    Drop-in replacement for _make_ollama_adapter() for the grader.
    """
    from src.models import generate_chat_openai

    def call_llm(system: str, user: str) -> str:
        return generate_chat_openai(system, user, max_tokens=2500, temperature=0.2)

    return call_llm


def run_section_recall(
    document_id: str,
    node_id: str,
    student_recall: str,
    max_retries: int = 1,
) -> Dict:
    """
    Core Aurora flow: user selects a ToC section, writes a free recall of its
    content, and the LLM scores it against the actual section text.

    Args:
        document_id:    UUID of the document in Supabase
        node_id:        ToC node the user chose to be tested on
        student_recall: Free-form text the user wrote from memory
        max_retries:    How many times to retry if LLM returns invalid JSON

    Returns:
        {
            "total_score":       int   (0–10),
            "max_score":         int   (10),
            "percentage":        float (0–100),
            "overall_feedback":  str,
            "criteria_scores":   List[{"id", "score", "feedback"}],
            "performance_level": str,   # "Mastery" / "Proficient" / "Needs Review"
            "interpretation":    str,
            "section_title":     str,
            "page_range":        str,
        }
    """
    content = get_section_content(document_id, node_id, include_children=True)
    context_text = _truncate_at_boundary(content["text"], max_chars=5000)

    call_llm = _make_openai_adapter()

    grade: Grade = grade_with_guardrails(
        question=_RECALL_PROMPT,
        student_answer=student_recall,
        context=context_text,
        call_llm=call_llm,
        max_retries=max_retries,
        allow_repair=True,
        repair_llm=call_llm,
    )

    return {
        "total_score":       grade.total_score,
        "max_score":         MAX_SCORE,
        "percentage":        grade.percentage,
        "overall_feedback":  grade.overall_feedback,
        "criteria_scores": [
            {"id": cs.criterion_id, "score": cs.score, "feedback": cs.feedback}
            for cs in grade.criteria_scores
        ],
        "performance_level": grade.performance_level,
        "interpretation":    grade.interpretation,
        "section_title":     content["section"]["title"],
        "page_range":        content["page_range"],
    }


def run_quick_check(
    context_text: str,
    student_recall: str,
) -> Grade:
    """
    Lightweight version — pass context directly (no DB call).
    Useful for rapid testing without a full Supabase setup.

    Returns a Grade object; call .to_dict() to get a flat dict for display.
    """
    call_llm = _make_openai_adapter()
    return grade_with_guardrails(
        question=_RECALL_PROMPT,
        student_answer=student_recall,
        context=context_text,
        call_llm=call_llm,
    )


# ──────────────────────────────────────────────
# Interactive CLI test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.retrieval import get_document_list, get_document_toc

    print("Aurora — Section Recall Assessment")
    print("=" * 60)

    # ── Step 1: DB connectivity + document list ─────────────────
    try:
        docs = get_document_list()
    except Exception as exc:
        print(f"\nCannot reach database: {exc}")
        print("Check your SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)

    if not docs:
        print("\nNo documents found. Run ingestion first.")
        sys.exit(1)

    # ── Step 2: Document selection ──────────────────────────────
    print("\nAvailable documents:")
    for i, doc in enumerate(docs):
        print(f"  [{i}] {doc['title']}  ({doc['total_sections']} sections, {doc['total_pages']} pages)")

    raw = input("\nSelect document [0]: ").strip()
    doc_idx = int(raw) if raw else 0
    document_id = docs[doc_idx]["id"]

    # ── Step 3: ToC display ─────────────────────────────────────
    print(f"\nTable of Contents — {docs[doc_idx]['title']}")
    print("-" * 60)

    def _print_toc(nodes, indent=0):
        for node in nodes:
            prefix = "  " * indent
            print(f"  {prefix}{node['node_id']}  |  {node['title']}  (p.{node['page_start']}–{node['page_end']})")
            if node.get("children"):
                _print_toc(node["children"], indent + 1)

    toc = get_document_toc(document_id)
    _print_toc(toc)

    # ── Step 4: Section selection ───────────────────────────────
    node_id = input("\nPaste node_id to be tested on: ").strip()

    # ── Step 5: Free recall input ───────────────────────────────
    print("\nWrite everything you remember about this section.")
    print("Press Enter twice when done.\n")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    student_recall = "\n".join(lines).rstrip()

    if not student_recall:
        print("Nothing written.")
        sys.exit(1)

    # ── Step 6: Grade ───────────────────────────────────────────
    print("\nGrading...")
    try:
        ev = run_section_recall(document_id, node_id, student_recall)
    except ValueError as exc:
        # e.g. node_id not found
        print(f"\nError: {exc}")
        sys.exit(1)
    except ConnectionError:
        print("\nCannot reach Ollama. Start it with:  ollama serve")
        print("Then make sure your model is pulled:  ollama pull <model-name>")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"RESULT  —  {ev['section_title']}  (p.{ev['page_range']})")
    print(f"{'=' * 60}")
    print(f"Score:  {ev['total_score']}/{ev['max_score']}  ({ev['percentage']}%)  —  {ev['performance_level']}")
    print(f"\n{ev['interpretation']}")
    print(f"\nFeedback:\n{ev['overall_feedback']}")
    print("\nPer-criterion breakdown:")
    for cs in ev["criteria_scores"]:
        print(f"  {cs['id']}: {cs['score']}/2  —  {cs['feedback']}")

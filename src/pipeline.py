from __future__ import annotations

from typing import Dict

from src.retrieval import get_section_content
from src.guardrailed_grader import grade_with_guardrails, Grade, MAX_SCORE


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


def run_evaluation(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str,
    max_retries: int = 1,
) -> Dict:
    """
    Full evaluation pipeline for Aurora.

    Args:
        document_id:    UUID of the document in Supabase
        node_id:        ToC node the student chose to study
        question:       The question being answered (generated or free-form)
        student_answer: Raw text the student typed
        max_retries:    How many times to retry if LLM returns invalid JSON

    Returns:
        {
            "total_score":      int   (0–10),
            "max_score":        int   (10),
            "percentage":       float (0–100),
            "overall_feedback": str,
            "criteria_scores":  List[{"id", "score", "feedback"}],
            "citations":        List[str],
            "performance_level": str,
            "interpretation":   str,
            "section_title":    str,
            "page_range":       str,
        }
    """
    content = get_section_content(document_id, node_id, include_children=True)
    context_text = _truncate_at_boundary(content["text"], max_chars=5000)

    call_llm = _make_ollama_adapter()

    grade: Grade = grade_with_guardrails(
        question=question,
        student_answer=student_answer,
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
        "citations":         grade.citations,
        "performance_level": grade.performance_level,
        "interpretation":    grade.interpretation,
        "section_title":     content["section"]["title"],
        "page_range":        content["page_range"],
    }


def run_quick_check(
    context_text: str,
    question: str,
    student_answer: str,
) -> Grade:
    """
    Lightweight version — pass context directly (no DB call).
    Useful for rapid testing without a full Supabase setup.

    Returns a Grade object; call .to_dict() to get a flat dict for display.
    """
    call_llm = _make_ollama_adapter()
    return grade_with_guardrails(
        question=question,
        student_answer=student_answer,
        context=context_text,
        call_llm=call_llm,
    )


# ──────────────────────────────────────────────
# Interactive CLI test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("Aurora — Assessment CLI")
    print("=" * 60)

    # ── Try full DB-connected flow ──────────────────────────────
    try:
        from src.retrieval import get_document_list, get_document_toc
        from src.assessment import generate_questions

        docs = get_document_list()
        if not docs:
            raise RuntimeError("No documents found in database. Run ingestion first.")

        print("\nAvailable documents:")
        for i, doc in enumerate(docs):
            print(f"  [{i}] {doc['title']}  ({doc['total_sections']} sections, {doc['total_pages']} pages)")

        raw = input("\nSelect document [0]: ").strip()
        doc_idx = int(raw) if raw else 0
        document_id = docs[doc_idx]["id"]

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

        node_id = input("\nPaste node_id to study: ").strip()

        print("\nGenerating questions...")
        result = generate_questions(document_id, node_id, num_questions=3)

        if not result["questions"]:
            print("No questions generated. Check that Ollama is running and the model is loaded.")
            sys.exit(1)

        section_info = result["section_info"]
        print(f"\nSection: {section_info['title']}  (p.{section_info['page_range']})")
        print("-" * 60)
        for i, q in enumerate(result["questions"]):
            print(f"  Q{i + 1} [{q['difficulty']}]: {q['question']}")

        raw = input("\nSelect question to answer [0]: ").strip()
        q_idx = int(raw) if raw else 0
        question = result["questions"][q_idx]["question"]

        print(f"\nQuestion: {question}")
        student_answer = input("Your answer: ").strip()

        if not student_answer:
            print("No answer provided.")
            sys.exit(1)

        print("\nGrading...")
        ev = run_evaluation(document_id, node_id, question, student_answer)

        print(f"\n{'=' * 60}")
        print(f"RESULT  —  {ev['section_title']}  (p.{ev['page_range']})")
        print(f"{'=' * 60}")
        print(f"Score:  {ev['total_score']}/{ev['max_score']}  ({ev['percentage']}%)  —  {ev['performance_level']}")
        print(f"\n{ev['interpretation']}")
        print(f"\nFeedback:\n{ev['overall_feedback']}")
        print("\nPer-criterion breakdown:")
        for cs in ev["criteria_scores"]:
            print(f"  {cs['id']}: {cs['score']}/2  —  {cs['feedback']}")
        if ev["citations"]:
            print(f"\nCitations: {', '.join(ev['citations'])}")

    # ── Fallback: quick check with hardcoded sample ─────────────
    except Exception as exc:
        print(f"\nDB mode unavailable ({exc}). Running quick check with sample data...\n")

        sample_context = (
            "Newton's Second Law of Motion states that the force acting on an object "
            "is equal to the mass of that object multiplied by its acceleration: F = ma. "
            "This means that for a constant force, a heavier object will accelerate less "
            "than a lighter one. The unit of force is the Newton (N), defined as 1 kg·m/s²."
        )

        grade = run_quick_check(
            context_text=sample_context,
            question="Explain Newton's Second Law and its implications.",
            student_answer=(
                "Newton's second law says that force equals mass times acceleration. "
                "So if you push something heavier, it moves slower."
            ),
        )

        print(f"Score:    {grade.total_score}/{MAX_SCORE}  ({grade.percentage}%)")
        print(f"Level:    {grade.performance_level}")
        print(f"Feedback: {grade.overall_feedback}")

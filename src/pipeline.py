from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from src.retrieval import get_section_content
from src.guardrailed_grader import grade_with_guardrails, Grade

RUBRIC_PATH = Path(__file__).parent / "assessor_rubric.json"
with open(RUBRIC_PATH) as f:
    RUBRIC = json.load(f)

def _make_ollama_adapter():
    """
    Returns a callable (system: str, user: str) -> str
    that guardrailed_grader expects as `call_llm`.
    """
    from src.models import generate

    def call_llm(system: str, user: str) -> str:
        # Ollama doesn't separate system/user in the simple generate() API.
        # Prepend system context inline — works well for instruction-following models.
        full_prompt = f"{system}\n\n{user}" if system.strip() else user
        return generate(full_prompt, max_tokens=1000, temperature=0.2)

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
            "score":        int (0–5),
            "max_score":    int (5),
            "percentage":   float (0–100),
            "rationale":    str,            # overall narrative feedback
            "criteria":     List[str],      # rubric criteria touched
            "citations":    List[str],      # chunks/pages the grader referenced
            "level":        str,            # "Mastery" / "Proficient" / "Needs Review"
            "section_title": str,
            "page_range":   str,
        }
    """

    # 1. Retrieve reference content for this section
    content = get_section_content(document_id, node_id, include_children=True)
    context_text = content["text"][:5000]  # guard against huge sections

    # 2. Build LLM adapter
    call_llm = _make_ollama_adapter()

    # 3. Run graded evaluation with guardrails
    grade: Grade = grade_with_guardrails(
        question=question,
        student_answer=student_answer,
        context=context_text,
        call_llm=call_llm,
        max_retries=max_retries,
        allow_repair=True,
        repair_llm=call_llm,  # reuse same model for repair
    )

    # 4. Map score (0–5) to performance level using rubric interpretation
    # The rubric uses 0–10 internally; guardrailed_grader uses 0–5.
    # We normalize to percentage and apply the rubric thresholds.
    percentage = round((grade.score / 5) * 100)
    normalized_10 = grade.score * 2  # 0–5 → 0–10 for rubric lookup

    if normalized_10 >= 9:
        level = "Mastery"
        interpretation = RUBRIC["interpretation"]["9-10"]
    elif normalized_10 >= 6:
        level = "Proficient"
        interpretation = RUBRIC["interpretation"]["6-8"]
    else:
        level = "Needs Review"
        interpretation = RUBRIC["interpretation"]["0-5"]

    # 5. Return structured result
    return {
        "score": grade.score,
        "max_score": 5,
        "percentage": percentage,
        "rationale": grade.rationale,
        "criteria": grade.criteria,
        "citations": grade.citations,
        "level": level,
        "interpretation": interpretation,
        "section_title": content["section"]["title"],
        "page_range": content["page_range"],
    }


def run_quick_check(
    context_text: str,
    question: str,
    student_answer: str,
) -> Grade:
    """
    Lightweight version — pass context directly (no DB call).
    Useful for rapid testing without a full Supabase setup.

    Example:
        grade = run_quick_check(
            context_text="Newton's second law states F = ma...",
            question="Explain Newton's second law",
            student_answer="Force equals mass times acceleration",
        )
        print(grade.score, grade.rationale)
    """
    call_llm = _make_ollama_adapter()
    return grade_with_guardrails(
        question=question,
        student_answer=student_answer,
        context=context_text,
        call_llm=call_llm,
    )


# ──────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Aurora — Quick Check (no DB)")
    print("=" * 50)

    sample_context = """
    Newton's Second Law of Motion states that the force acting on an object
    is equal to the mass of that object multiplied by its acceleration: F = ma.
    This means that for a constant force, a heavier object will accelerate less
    than a lighter one. The unit of force is the Newton (N), defined as
    1 kg·m/s².
    """

    result = run_quick_check(
        context_text=sample_context,
        question="Explain Newton's Second Law and its implications.",
        student_answer=(
            "Newton's second law says that force equals mass times acceleration. "
            "So if you push something heavier, it moves slower."
        ),
    )

    print(f"Score:     {result.score}/5")
    print(f"Rationale: {result.rationale}")
    print(f"Criteria:  {result.criteria}")
    print(f"Citations: {result.citations}")
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.retrieval import get_section_content
from src.guardrailed_grader import grade_with_guardrails, Grade, MAX_SCORE

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
            "total_score":      int   (0–10),
            "max_score":        int   (10),
            "percentage":       float (0–100),
            "overall_feedback": str,
            "criteria_scores":  List[{"id", "score", "feedback"}],
            "citations":        List[str],
            "performance_level": str,   # "Mastery" / "Proficient" / "Needs Review"
            "interpretation":   str,
            "section_title":    str,
            "page_range":       str,
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

    # 4. Return structured result using Grade's own computed properties.
    #    Grade.total_score  = sum of 5 criteria × 2 pts each  = 0–10
    #    Grade.percentage   = (total_score / MAX_SCORE) * 100
    #    Grade.performance_level / .interpretation derived from rubric thresholds.
    return {
        "total_score":      grade.total_score,
        "max_score":        MAX_SCORE,
        "percentage":       grade.percentage,
        "overall_feedback": grade.overall_feedback,
        "criteria_scores": [
            {
                "id":       cs.criterion_id,
                "score":    cs.score,
                "feedback": cs.feedback,
            }
            for cs in grade.criteria_scores
        ],
        "citations":        grade.citations,
        "performance_level": grade.performance_level,
        "interpretation":   grade.interpretation,
        "section_title":    content["section"]["title"],
        "page_range":       content["page_range"],
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

    Example:
        grade = run_quick_check(
            context_text="Newton's second law states F = ma...",
            question="Explain Newton's second law",
            student_answer="Force equals mass times acceleration",
        )
        print(grade.total_score, grade.overall_feedback)
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

    grade = run_quick_check(
        context_text=sample_context,
        question="Explain Newton's Second Law and its implications.",
        student_answer=(
            "Newton's second law says that force equals mass times acceleration. "
            "So if you push something heavier, it moves slower."
        ),
    )

    print(f"Score:     {grade.total_score}/{MAX_SCORE}")
    print(f"Feedback:  {grade.overall_feedback}")
    print(f"Level:     {grade.performance_level}")
    print(f"Citations: {grade.citations}")

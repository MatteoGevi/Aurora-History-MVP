# src/guardrailed_grader.py
# pip install pydantic==2.*

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────
# Load rubric once at import time
# ──────────────────────────────────────────────
_RUBRIC_PATH = Path(__file__).parent / "assessor_rubric.json"
with open(_RUBRIC_PATH) as _f:
    RUBRIC_DEF = json.load(_f)

# Pre-build a lookup: criterion_id -> criterion dict
CRITERIA_BY_ID = {c["id"]: c for c in RUBRIC_DEF["criteria"]}
MAX_SCORE = RUBRIC_DEF["total_score"]  # 10


# ──────────────────────────────────────────────
# Pydantic schema
# ──────────────────────────────────────────────
class CriterionScore(BaseModel):
    criterion_id: str = Field(..., description="e.g. C1, C2, ...")
    score: int = Field(..., ge=0, le=2, description="0, 1, or 2 as defined in rubric levels")
    feedback: str = Field(..., min_length=5, max_length=500)

    @field_validator("criterion_id")
    @classmethod
    def must_be_known_criterion(cls, v: str) -> str:
        if v not in CRITERIA_BY_ID:
            raise ValueError(f"Unknown criterion_id '{v}'. Valid: {list(CRITERIA_BY_ID)}")
        return v


class Grade(BaseModel):
    criteria_scores: List[CriterionScore] = Field(
        ...,
        min_length=len(RUBRIC_DEF["criteria"]),
        max_length=len(RUBRIC_DEF["criteria"]),
        description="One entry per rubric criterion",
    )
    overall_feedback: str = Field(..., min_length=10, max_length=2000)
    citations: List[str] = Field(
        default_factory=list,
        description="Section titles or page ranges referenced",
    )

    # ── Derived properties ──────────────────────
    @property
    def total_score(self) -> int:
        return sum(c.score for c in self.criteria_scores)

    @property
    def percentage(self) -> float:
        return round((self.total_score / MAX_SCORE) * 100, 1)

    @property
    def performance_level(self) -> str:
        s = self.total_score
        if s >= 9:
            return "Mastery"
        elif s >= 6:
            return "Proficient"
        return "Needs Review"

    @property
    def interpretation(self) -> str:
        s = self.total_score
        if s >= 9:
            return RUBRIC_DEF["interpretation"]["9-10"]
        elif s >= 6:
            return RUBRIC_DEF["interpretation"]["6-8"]
        return RUBRIC_DEF["interpretation"]["0-5"]

    def to_dict(self) -> dict:
        """Flat dict ready for Streamlit or JSON serialisation."""
        return {
            "total_score": self.total_score,
            "max_score": MAX_SCORE,
            "percentage": self.percentage,
            "performance_level": self.performance_level,
            "interpretation": self.interpretation,
            "overall_feedback": self.overall_feedback,
            "citations": self.citations,
            "criteria_scores": [
                {
                    "id": cs.criterion_id,
                    "title": CRITERIA_BY_ID[cs.criterion_id]["title"],
                    "score": cs.score,
                    "max": 2,
                    "feedback": cs.feedback,
                }
                for cs in self.criteria_scores
            ],
        }


# ──────────────────────────────────────────────
# JSON schema for the prompt (derived from rubric)
# ──────────────────────────────────────────────
def _build_schema_str() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "criteria_scores": {
                    "type": "array",
                    "minItems": len(RUBRIC_DEF["criteria"]),
                    "maxItems": len(RUBRIC_DEF["criteria"]),
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion_id": {
                                "type": "string",
                                "enum": list(CRITERIA_BY_ID.keys()),
                            },
                            "score": {"type": "integer", "minimum": 0, "maximum": 2},
                            "feedback": {"type": "string"},
                        },
                        "required": ["criterion_id", "score", "feedback"],
                    },
                },
                "overall_feedback": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["criteria_scores", "overall_feedback"],
            "additionalProperties": False,
        },
        indent=2,
    )


SCHEMA_JSON_STR = _build_schema_str()


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────
def _build_rubric_block() -> str:
    lines = [f"RUBRIC (each criterion scored 0, 1, or 2 — total max {MAX_SCORE}):\n"]
    for c in RUBRIC_DEF["criteria"]:
        lines.append(f"{c['id']}: {c['title']}")
        lines.append(f"  Description: {c['description']}")
        for lvl in c["levels"]:
            lines.append(f"  {lvl['score']} pts — {lvl['description']}")
        lines.append("")
    return "\n".join(lines)


RUBRIC_BLOCK = _build_rubric_block()

SYSTEM_PROMPT = (
    "You are a strict academic assessor for Aurora, a personalised learning platform.\n"
    "Grade ONLY using the supplied CONTEXT. Penalise answers that use knowledge outside the CONTEXT.\n"
    "Output ONLY valid JSON matching the provided schema. No prose, no markdown, no comments."
)


def build_user_prompt(question: str, student_answer: str, context: str) -> str:
    return f"""TASK:
Evaluate the student's answer using ONLY the CONTEXT and the rubric below.

QUESTION:
{question.strip()}

STUDENT ANSWER:
{student_answer.strip()}

CONTEXT (authoritative source — do not use outside knowledge):
{context.strip()}

{RUBRIC_BLOCK}

JSON SCHEMA (enforce strictly):
{SCHEMA_JSON_STR}

Rules:
- Score every criterion (C1–C5) individually.
- In "citations", list section titles or page ranges from CONTEXT that you used.
- "overall_feedback" must mention at least one strength and one area to improve.
- Return ONLY JSON. No prose outside the JSON object.
"""


# ──────────────────────────────────────────────
# Repair prompt
# ──────────────────────────────────────────────
def default_repair_prompt(invalid_text: str) -> str:
    return f"""The following text should be valid JSON matching this schema:

SCHEMA:
{SCHEMA_JSON_STR}

TEXT:
{invalid_text}

Return ONLY corrected JSON. No explanation, no markdown.
"""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
LLMFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> raw_string


def _extract_json(text: str) -> str:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text


# ──────────────────────────────────────────────
# Core grading loop
# ──────────────────────────────────────────────
def grade_with_guardrails(
    question: str,
    student_answer: str,
    context: str,
    call_llm: LLMFn,
    max_retries: int = 1,
    allow_repair: bool = True,
    repair_llm: Optional[LLMFn] = None,
) -> Grade:
    """
    1. Call LLM with structured rubric prompt.
    2. Validate response against Pydantic Grade schema.
    3. On failure: retry up to max_retries with a reminder.
    4. On continued failure: optionally run a repair pass (format-only fix).
    5. Raise ValueError with a clean message if all attempts fail.
    """
    attempt = 0
    last_err: Optional[Exception] = None
    sys_prompt = SYSTEM_PROMPT
    candidate = ""

    while attempt <= max_retries:
        user_prompt = build_user_prompt(question, student_answer, context)
        raw = call_llm(sys_prompt, user_prompt)
        candidate = _extract_json(raw)
        try:
            data = json.loads(candidate)
            return Grade(**data)
        except Exception as e:
            last_err = e
            sys_prompt = SYSTEM_PROMPT + "\nREMINDER: Output ONLY JSON matching the schema. No prose."
            attempt += 1

    # Repair pass — fixes JSON format only, does not re-grade
    if allow_repair and repair_llm is not None:
        try:
            repaired = repair_llm("", default_repair_prompt(candidate))
            fixed = _extract_json(repaired)
            data = json.loads(fixed)
            return Grade(**data)
        except Exception as e2:
            last_err = e2

    snippet = candidate[:400].replace("\n", " ")
    raise ValueError(
        f"Grader failed after {max_retries + 1} attempts + repair. "
        f"Last error: {last_err}. Output snippet: {snippet}"
    )
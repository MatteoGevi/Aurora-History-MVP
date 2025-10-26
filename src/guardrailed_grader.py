# guardrailed_grader.py
# pip install pydantic==2.*  (or adapt to v1 if needed)

from __future__ import annotations
from typing import List, Callable, Any, Optional
import json, re
from pydantic import BaseModel, Field, ValidationError, conint

# --------- SCHEMA ---------
class Grade(BaseModel):
    score: conint(ge=0, le=5) = Field(..., description="Integer score 0–5")
    rationale: str = Field(..., min_length=5, max_length=2000)
    criteria: List[str] = Field(..., min_items=1, max_items=8)
    citations: List[str] = Field(
        default_factory=list,
        description="Chunk/section identifiers used (e.g., node ids, level_path, or page ranges)",
    )

SCHEMA_JSON_STR = json.dumps({
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "rationale": {"type": "string"},
        "criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
        "citations": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["score", "rationale", "criteria"],
    "additionalProperties": False
}, indent=2)

# --------- PROMPTS ---------
SYSTEM_PROMPT = """You are a strict history assessor.
Grade ONLY using the supplied CONTEXT. If the answer relies on knowledge outside the CONTEXT, deduct points.
Output ONLY valid JSON that matches the provided JSON schema, with no extra text, no markdown, no comments.
"""

RUBRIC = """SCORE RUBRIC (0–5):
5: Fully correct, precise, and complete, explicitly grounded in context.
4: Mostly correct; minor omission/ambiguity; grounded.
3: Partially correct; key gaps or mild speculation.
2: Limited correctness; major gaps; weak grounding.
1: Minimal relevant content; mostly incorrect.
0: Off-topic or contradicts context; no grounding.
"""

def build_user_prompt(question: str, student_answer: str, context: str, citation_hint: Optional[str] = None) -> str:
    return f"""TASK:
Grade the student's answer to the question using ONLY the CONTEXT below. Apply the rubric strictly.

QUESTION:
{question.strip()}

STUDENT_ANSWER:
{student_answer.strip()}

CONTEXT (verbatim, authoritative; do not use outside knowledge):
{context.strip()}

{RUBRIC}

JSON SCHEMA (enforce strictly):
{SCHEMA_JSON_STR}

Rules:
- Cite which context parts you used in "citations" (e.g., section titles or page ranges you see).
- Return ONLY JSON conforming to the schema. No prose.
"""

# --------- LLM ADAPTERS ---------
# Provide your own LLM call (OpenAI, vLLM, etc.). It must return a raw string.
LLMFn = Callable[[str, str], str]  # (system, user) -> model_text

def default_repair_prompt(invalid_text: str) -> str:
    return f"""The following text should be valid JSON matching this schema:

SCHEMA:
{SCHEMA_JSON_STR}

TEXT:
{invalid_text}

Return ONLY corrected JSON that conforms to the schema. Do not explain, no markdown, no comments.
"""

# --------- CORE LOOP ---------
def _extract_json_maybe(text: str) -> str:
    # Try raw first
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # Try to grab a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text  # let validator fail -> repair

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
    1) Ask LLM for strict JSON (system+user prompts).
    2) Validate with Pydantic.
    3) If invalid -> retry once with a reminder.
    4) If still invalid and allow_repair -> run repair model to coerce to schema.
    """
    attempt = 0
    last_err: Optional[Exception] = None
    sys_prompt = SYSTEM_PROMPT

    while attempt <= max_retries:
        user_prompt = build_user_prompt(question, student_answer, context)
        raw = call_llm(sys_prompt, user_prompt)
        candidate = _extract_json_maybe(raw)
        try:
            data = json.loads(candidate)
            return Grade(**data)
        except Exception as e:
            last_err = e
            # one retry with explicit reminder
            sys_prompt = SYSTEM_PROMPT + "\nREMINDER: Output ONLY JSON matching the schema. No prose."
            attempt += 1

    # Optional repair step (does not re-grade; just fixes format, if close)
    if allow_repair and repair_llm is not None:
        repaired = repair_llm("", default_repair_prompt(candidate))
        try:
            fixed = _extract_json_maybe(repaired)
            data = json.loads(fixed)
            return Grade(**data)
        except Exception as e2:
            last_err = e2

    # If we reach here, bubble a clean error with the last invalid output snippet
    snippet = candidate[:500].replace("\n", " ")
    raise ValueError(f"Grader produced invalid JSON after retries. Last parse error: {last_err}. Snippet: {snippet}")
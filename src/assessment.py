# src/assessment.py
import json
from typing import List, Dict
from src.models import generate_chat_claude
from src.retrieval import get_section_content


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a paragraph or sentence boundary, not mid-word."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut == -1:
        cut = text.rfind(". ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut] + "\n\n[...content truncated...]"


def generate_questions(
    document_id: str,
    node_id: str,
    num_questions: int = 5,
    difficulty: str = "mixed",
    max_retries: int = 2,
) -> Dict:
    """Generate assessment questions with retry on JSON parse failure."""

    content = get_section_content(document_id, node_id, include_children=True)
    context_text = _truncate_at_boundary(content['text'], max_chars=4000)

    prompt = f"""You are an expert educator creating assessment questions.

Section: {content['section']['title']}
Content: {context_text}

Generate {num_questions} {difficulty} questions. Return ONLY a JSON array:

[
  {{
    "question": "...",
    "difficulty": "easy/medium/hard",
    "page_reference": "X-Y",
    "key_concepts": ["concept1", "concept2"]
  }}
]"""

    questions = []
    last_err = None

    for attempt in range(max_retries + 1):
        response = generate_chat_claude("You are an expert educator creating assessment questions.", prompt, max_tokens=2000)
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            questions = json.loads(response[start:end]) if start >= 0 else json.loads(response)
            break  # success
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                print(f"⚠️ Parse error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying...")

    if not questions:
        print(f"⚠️ Failed to generate questions after {max_retries + 1} attempts: {last_err}")

    return {
        "questions": questions,
        "section_info": {
            "title": content['section']['title'],
            "page_range": content['page_range']
        }
    }
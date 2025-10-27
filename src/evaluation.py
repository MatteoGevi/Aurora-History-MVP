# src/evaluation.py
import json
from typing import Dict
from src.models import generate  # Just import the function
from src.retrieval import get_section_content

def evaluate_answer(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str
) -> Dict:
    """Evaluate student answer"""
    
    content = get_section_content(document_id, node_id, include_children=True)
    
    prompt = f"""You are grading a student's answer.

Question: {question}
Student Answer: {student_answer}

Reference Material: {content['text'][:4000]}

Evaluate based ONLY on the reference. Return ONLY JSON:

{{
  "score": 85,
  "correct_points": ["...", "..."],
  "missing_points": ["..."],
  "misconceptions": ["..."],
  "suggestions": "..."
}}
"""
    
    response = generate(prompt, max_tokens=1500, temperature=0.3)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        evaluation = json.loads(response[start:end]) if start >= 0 else json.loads(response)
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        evaluation = {"score": 0, "error": "Parse failed"}
    
    return evaluation
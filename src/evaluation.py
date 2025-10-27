# src/evaluation.py (SIMPLIFIED)
import json
from typing import Dict
from src.models import model
from src.retrieval import get_section_content

def evaluate_answer(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str
) -> Dict:
    """Evaluate student answer"""
    
    # Get reference content
    content = get_section_content(document_id, node_id, include_children=True)
    
    # Build prompt
    prompt = f"""You are an expert AI Engineering educator evaluating a student's answer.

**Question:** {question}

**Student's Answer:**
{student_answer}

**Reference Material from Textbook:**
{content['text'][:4000]}

---

Evaluate the student's answer based ONLY on the reference material provided.

Return ONLY JSON with this exact format (no other text):
{{
  "score": 85,
  "correct_points": ["Point 1 they got right", "Point 2 they got right"],
  "missing_points": ["Key concept they missed"],
  "misconceptions": ["Any incorrect statements"],
  "suggestions": "How they can improve their answer"
}}
"""
    
    # Generate evaluation
    response = model.generate(prompt, max_tokens=1500, temperature=0.3)
    
    # Parse
    try:
        # Try to extract JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            evaluation = json.loads(response[start:end])
        else:
            evaluation = json.loads(response)
    except Exception as e:
        print(f"⚠️ Failed to parse evaluation: {e}")
        print(f"Raw response: {response[:500]}")
        evaluation = {
            "score": 0,
            "error": "Failed to parse evaluation",
            "raw_response": response[:500]
        }
    
    evaluation['model_used'] = "mistralai/Mistral-7B-Instruct-v0.3"
    return evaluation
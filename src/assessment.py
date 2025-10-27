# src/assessment.py
import json
from typing import List, Dict
from src.models import generate  # Just import the function
from src.retrieval import get_section_content

def generate_questions(
    document_id: str,
    node_id: str,
    num_questions: int = 5,
    difficulty: str = "mixed"
) -> Dict:
    """Generate assessment questions"""
    
    content = get_section_content(document_id, node_id, include_children=True)
    
    prompt = f"""You are an expert educator creating assessment questions.

Section: {content['section']['title']}
Content: {content['text'][:4000]}

Generate {num_questions} {difficulty} questions. Return ONLY a JSON array:

[
  {{
    "question": "...",
    "difficulty": "easy/medium/hard",
    "page_reference": "X-Y",
    "key_concepts": ["concept1", "concept2"]
  }}
]
"""
    
    response = generate(prompt, max_tokens=2000, temperature=0.5)
    
    # Parse
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        questions = json.loads(response[start:end]) if start >= 0 else json.loads(response)
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        questions = []
    
    return {
        "questions": questions,
        "section_info": {
            "title": content['section']['title'],
            "page_range": content['page_range']
        }
    }
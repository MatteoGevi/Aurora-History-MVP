# src/assessment.py (SIMPLIFIED)
import json
from typing import List, Dict
from src.models import model
from src.retrieval import get_section_content

def generate_questions(
    document_id: str,
    node_id: str,
    num_questions: int = 5,
    difficulty: str = "mixed"
) -> Dict:
    """Generate assessment questions"""
    
    # Get section content
    content = get_section_content(document_id, node_id, include_children=True)
    
    # Build prompt
    prompt = f"""You are an expert educator creating assessment questions for an AI Engineering course.

**Section:** {content['section']['title']}
**Pages:** {content['page_range']}

**Content:**
{content['text'][:4000]}

---

Generate {num_questions} {difficulty} assessment questions that test understanding of this section.

Requirements:
- Mix conceptual understanding with technical details
- Reference specific concepts from the text
- Include page numbers where answers can be found
- Test both theory and practical application

Return ONLY a JSON array with this exact format (no other text):
[
  {{
    "question": "What is...",
    "difficulty": "easy/medium/hard",
    "page_reference": "45-47",
    "key_concepts": ["concept1", "concept2"]
  }}
]
"""
    
    # Generate
    response = model.generate(prompt, max_tokens=2000, temperature=0.5)
    
    # Parse
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            questions = json.loads(response[start:end])
        else:
            questions = json.loads(response)
    except Exception as e:
        print(f"⚠️ Failed to parse questions: {e}")
        print(f"Raw response: {response[:500]}")
        questions = []
    
    return {
        "questions": questions,
        "section_info": {
            "title": content['section']['title'],
            "page_range": content['page_range'],
            "total_words": content['total_words']
        },
        "model_used": "mistralai/Mistral-7B-Instruct-v0.3"
    }
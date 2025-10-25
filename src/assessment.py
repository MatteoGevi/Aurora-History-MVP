# retrieval/assessment.py
from typing import List, Dict
from retrieval import get_section_content
import anthropic  # or openai, or your LLM provider
from config.constants import ANTHROPIC_API_KEY

# Initialize your LLM client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_questions(
    document_id: str,
    node_id: str,
    num_questions: int = 5,
    difficulty: str = "mixed",  # easy, medium, hard, mixed
    question_types: List[str] = None  # ["multiple_choice", "short_answer", "true_false"]
) -> Dict:
    """
    Generate assessment questions for a selected section.
    
    This is WHERE YOUR LLM IS CALLED for question generation.
    
    Args:
        document_id: Document UUID
        node_id: Section node_id (e.g., "h2-3-1__newtons-laws")
        num_questions: How many questions to generate
        difficulty: Question difficulty level
        question_types: Types of questions to generate
        
    Returns:
        {
            "section_title": "...",
            "page_range": "45-67",
            "questions": [
                {
                    "id": 1,
                    "type": "short_answer",
                    "question": "What is Newton's second law?",
                    "difficulty": "medium",
                    "page_reference": 46,
                    "key_concepts": ["force", "acceleration", "mass"]
                },
                ...
            ]
        }
    """
    # 1. Retrieve section content
    content = get_section_content(document_id, node_id, include_children=True)
    
    if not content['text']:
        return {"error": "No content found for this section"}
    
    # 2. Build LLM prompt
    prompt = f"""You are an educational assessment expert. Generate {num_questions} questions to test student understanding of this textbook section.

**Section:** {content['section']['title']}
**Pages:** {content['page_range']}
**Difficulty:** {difficulty}

**Content to base questions on:**
{content['text']}

---

**Instructions:**
- Generate {num_questions} questions that test deep understanding, not just memorization
- Mix question types: multiple choice, short answer, and conceptual questions
- Reference specific page numbers where answers can be found
- Include key concepts being tested
- Difficulty should be {difficulty}

**Output format (JSON):**
{{
  "questions": [
    {{
      "id": 1,
      "type": "short_answer",
      "question": "Explain Newton's second law and provide an example.",
      "difficulty": "medium",
      "page_reference": 46,
      "key_concepts": ["force", "acceleration", "mass"],
      "sample_answer": "Newton's second law states that F=ma..."
    }},
    ...
  ]
}}
"""
    
    # 3. Call LLM
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # 4. Parse response
    import json
    try:
        response_text = message.content[0].text
        # Extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        questions_data = json.loads(response_text[start:end])
        
        return {
            "section_title": content['section']['title'],
            "page_range": content['page_range'],
            "total_words": content['total_words'],
            "questions": questions_data['questions']
        }
    except Exception as e:
        return {"error": f"Failed to parse LLM response: {str(e)}"}


def generate_adaptive_question(
    document_id: str,
    node_id: str,
    previous_performance: Dict = None
) -> Dict:
    """
    Generate a single adaptive question based on student's previous performance.
    
    Called when: Adaptive learning - generate next question based on what student got wrong
    
    Args:
        document_id: Document UUID
        node_id: Section node_id
        previous_performance: {
            "correct_count": 3,
            "incorrect_count": 2,
            "weak_concepts": ["acceleration", "force"]
        }
        
    Returns:
        Single question dict
    """
    content = get_section_content(document_id, node_id)
    
    if previous_performance and previous_performance.get('weak_concepts'):
        focus_areas = ", ".join(previous_performance['weak_concepts'])
        prompt_addition = f"\n\n**IMPORTANT:** Focus specifically on these concepts the student struggled with: {focus_areas}"
    else:
        prompt_addition = ""
    
    prompt = f"""Generate ONE assessment question for this section.

**Section:** {content['section']['title']}
**Content:**
{content['text'][:2000]}...  # Limit to avoid token limits

{prompt_addition}

Output a single question in JSON format.
"""
    
    # Call LLM similar to above
    # ...
    
    return question_data
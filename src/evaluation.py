# retrieval/evaluation.py
from typing import Dict
from retrieval.retrieval import get_section_content
import anthropic
from config.constants import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def evaluate_answer(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str,
    sample_answer: str = None
) -> Dict:
    """
    Evaluate a student's answer using the section content as reference.
    
    This is WHERE YOUR LLM IS CALLED for answer grading.
    
    Args:
        document_id: Document UUID
        node_id: Section node_id where question came from
        question: The question that was asked
        student_answer: Student's response
        sample_answer: Optional sample answer for reference
        
    Returns:
        {
            "score": 85,  # 0-100
            "feedback": "Your answer correctly identifies...",
            "what_was_correct": ["..."],
            "what_was_missing": ["..."],
            "misconceptions": ["..."],
            "suggestions": "Consider reviewing..."
        }
    """
    # 1. Get reference content
    content = get_section_content(document_id, node_id, include_children=True)
    
    # 2. Build evaluation prompt
    prompt = f"""You are an expert teacher evaluating a student's answer to a textbook question.

**Question:** {question}

**Student's Answer:**
{student_answer}

**Reference Material (from textbook):**
{content['text']}

{f"**Sample Answer:** {sample_answer}" if sample_answer else ""}

---

**Evaluate the student's answer:**

1. **Score (0-100):** How accurate and complete is the answer?

2. **What was correct:** List specific points the student got right

3. **What was missing:** Key concepts or details they didn't mention

4. **Misconceptions:** Any factual errors or misunderstandings

5. **Suggestions:** Specific advice to improve their understanding

**Be encouraging but honest. Reference specific page numbers from the textbook where relevant.**

Output in JSON format:
{{
  "score": 85,
  "feedback": "Overall assessment...",
  "what_was_correct": ["point 1", "point 2"],
  "what_was_missing": ["concept A", "detail B"],
  "misconceptions": ["error if any"],
  "suggestions": "Review pages X-Y focusing on...",
  "page_references": [46, 47]
}}
"""
    
    # 3. Call LLM
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.3,  # Lower temperature for consistent grading
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
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        evaluation = json.loads(response_text[start:end])
        
        return evaluation
    except Exception as e:
        return {
            "error": f"Failed to evaluate: {str(e)}",
            "raw_response": message.content[0].text
        }


def provide_hint(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str
) -> str:
    """
    Provide a hint without giving away the answer.
    
    Called when: Student clicks "Get a hint" button
    """
    content = get_section_content(document_id, node_id)
    
    prompt = f"""A student is struggling with this question. Provide a helpful hint WITHOUT giving away the answer.

**Question:** {question}

**Student's current attempt:**
{student_answer}

**Reference (don't quote directly):**
{content['text'][:1000]}...

**Provide a hint that:**
- Points them in the right direction
- References a page number to review
- Asks a guiding question
- Does NOT give the answer directly
"""
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text


def explain_concept(
    document_id: str,
    node_id: str,
    concept: str
) -> str:
    """
    Explain a specific concept from the section.
    
    Called when: Student clicks "Explain this concept" after getting question wrong
    """
    content = get_section_content(document_id, node_id)
    
    prompt = f"""Explain this concept to a student in simple terms:

**Concept:** {concept}

**From textbook section:**
{content['text']}

Provide a clear, concise explanation with an example.
"""
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text
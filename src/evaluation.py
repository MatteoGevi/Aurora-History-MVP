# src/evaluation.py - WITH RUBRIC
import json
from typing import Dict
from pathlib import Path
from src.models import generate
from src.retrieval import get_section_content

# Load rubric (do this once at module level)
RUBRIC_PATH = Path(__file__).parent / "assessor_rubric.json"
with open(RUBRIC_PATH) as f:
    RUBRIC = json.load(f)

def evaluate_answer(
    document_id: str,
    node_id: str,
    question: str,
    student_answer: str
) -> Dict:
    """Evaluate student answer using structured rubric"""
    
    content = get_section_content(document_id, node_id, include_children=True)
    
    # Build rubric description for prompt
    rubric_text = f"""
GRADING RUBRIC (Total: {RUBRIC['total_score']} points):

"""
    for criterion in RUBRIC['criteria']:
        rubric_text += f"""
{criterion['id']}: {criterion['title']} (Weight: {criterion['weight']})
Description: {criterion['description']}
Scoring levels:
"""
        for level in criterion['levels']:
            rubric_text += f"  - {level['score']} points: {level['description']}\n"
    
    # Build evaluation prompt
    prompt = f"""You are an expert educator evaluating a student's answer using a structured rubric.

QUESTION: {question}

STUDENT'S ANSWER:
{student_answer}

REFERENCE MATERIAL (from textbook section: {content['section']['title']}):
{content['text'][:4000]}

{rubric_text}

INSTRUCTIONS:
1. Evaluate the student's answer against each criterion in the rubric
2. Assign a score (0, 1, or 2) for each criterion based on the scoring levels
3. Provide specific feedback for each criterion
4. Calculate total score (sum of all criterion scores)
5. Give overall suggestions for improvement

Return your evaluation as JSON:

{{
  "total_score": 8,
  "max_score": {RUBRIC['total_score']},
  "percentage": 80,
  "criteria_scores": [
    {{
      "criterion_id": "C1",
      "criterion_title": "Factual correctness",
      "score": 2,
      "feedback": "All major facts are correct..."
    }},
    {{
      "criterion_id": "C2",
      "criterion_title": "Coverage of section scope",
      "score": 1,
      "feedback": "Partial coverage, missing discussion of..."
    }},
    ...
  ],
  "overall_feedback": "Your answer demonstrates good understanding of...",
  "strengths": ["Strong grasp of core concepts", "Clear terminology use"],
  "areas_for_improvement": ["Expand coverage of subsection X", "Add more explanation of Y"],
  "performance_level": "Proficient"
}}
"""
    
    print("⏳ Evaluating with rubric...\n")
    
    response = generate(prompt, max_tokens=2000, temperature=0.3)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        evaluation = json.loads(response[start:end]) if start >= 0 else json.loads(response)
        
        # Add interpretation from rubric
        score = evaluation['total_score']
        if score >= 9:
            interpretation = RUBRIC['interpretation']['9-10']
        elif score >= 6:
            interpretation = RUBRIC['interpretation']['6-8']
        else:
            interpretation = RUBRIC['interpretation']['0-5']
        
        evaluation['interpretation'] = interpretation
        
    except Exception as e:
        print(f"⚠️ Parse error: {e}")
        evaluation = {
            "total_score": 0,
            "max_score": RUBRIC['total_score'],
            "error": "Failed to parse evaluation"
        }
    
    return evaluation
# api/routes.py
from flask import Flask, request, jsonify
from retrieval.assessment import generate_questions, generate_adaptive_question
from retrieval.evaluation import evaluate_answer, provide_hint, explain_concept
from retrieval.retrieval import get_document_list, get_document_toc

app = Flask(__name__)

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Get all documents"""
    docs = get_document_list()
    return jsonify(docs)


@app.route('/api/documents/<document_id>/toc', methods=['GET'])
def get_toc(document_id):
    """Get table of contents for browsing"""
    toc = get_document_toc(document_id)
    return jsonify(toc)


@app.route('/api/assessment/generate', methods=['POST'])
def generate_assessment():
    """
    Generate questions for a selected section.
    
    Request body:
    {
        "document_id": "uuid",
        "node_id": "h2-3-1__section",
        "num_questions": 5,
        "difficulty": "medium"
    }
    """
    data = request.json
    result = generate_questions(
        document_id=data['document_id'],
        node_id=data['node_id'],
        num_questions=data.get('num_questions', 5),
        difficulty=data.get('difficulty', 'mixed')
    )
    return jsonify(result)


@app.route('/api/assessment/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate a student's answer.
    
    Request body:
    {
        "document_id": "uuid",
        "node_id": "h2-3-1__section",
        "question": "What is...",
        "student_answer": "The answer is...",
        "sample_answer": "..." (optional)
    }
    """
    data = request.json
    result = evaluate_answer(
        document_id=data['document_id'],
        node_id=data['node_id'],
        question=data['question'],
        student_answer=data['student_answer'],
        sample_answer=data.get('sample_answer')
    )
    return jsonify(result)


@app.route('/api/assessment/hint', methods=['POST'])
def get_hint():
    """Get a hint for struggling student"""
    data = request.json
    hint = provide_hint(
        document_id=data['document_id'],
        node_id=data['node_id'],
        question=data['question'],
        student_answer=data['student_answer']
    )
    return jsonify({"hint": hint})


@app.route('/api/assessment/explain', methods=['POST'])
def explain():
    """Explain a concept"""
    data = request.json
    explanation = explain_concept(
        document_id=data['document_id'],
        node_id=data['node_id'],
        concept=data['concept']
    )
    return jsonify({"explanation": explanation})


if __name__ == '__main__':
    app.run(debug=True)
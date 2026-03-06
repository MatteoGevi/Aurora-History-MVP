# api/routes.py (KEEP IT SIMPLE)
from flask import Flask, request, jsonify
from src.assessment import generate_questions
from src.pipeline import run_evaluation
from src.retrieval import get_document_list, get_document_toc

app = Flask(__name__)

@app.route('/api/documents', methods=['GET'])
def list_documents():
    docs = get_document_list()
    return jsonify(docs)

@app.route('/api/documents/<document_id>/toc', methods=['GET'])
def get_toc(document_id):
    toc = get_document_toc(document_id)
    return jsonify(toc)

@app.route('/api/assessment/generate', methods=['POST'])
def generate_assessment():
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
    data = request.json
    result = run_evaluation(
        document_id=data['document_id'],
        node_id=data['node_id'],
        question=data['question'],
        student_answer=data['student_answer']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
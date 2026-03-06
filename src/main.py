# Example session — pseudocode showing the full Aurora flow

# 1. User opens app, selects document
docs = get_document_list()
# → Shows "My Physics Textbook"

# 2. User browses ToC
toc = get_document_toc(docs[0]['id'])
# → Shows tree: Chapter 3 > Section 3.2 > ...

# 3. User selects "Chapter 3: Newton's Laws"
questions = generate_questions(
    document_id=docs[0]['id'],
    node_id="h1-3__newtons-laws",
    num_questions=5
)
# → 🤖 LLM CALLED HERE to generate questions

# 4. Show questions to user
# UI displays: questions['questions']

# 5. User answers Question 1
evaluation = run_evaluation(
    document_id=docs[0]['id'],
    node_id="h1-3__newtons-laws",
    question=questions['questions'][0]['question'],
    student_answer="Force equals mass times acceleration"
)
# → 🤖 LLM CALLED HERE to grade answer

# 6. Show feedback
# UI displays: evaluation['overall_feedback'], evaluation['total_score']

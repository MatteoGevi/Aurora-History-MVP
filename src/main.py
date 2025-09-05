# Replace with your LLM adapter (OpenAI, vLLM, Mistral server, etc.)
def my_llm(system: str, user: str) -> str:
    # Pseudocode with OpenAI:
    # resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":system}, {"role":"user","content":user}], temperature=0)
    # return resp.choices[0].message.content
    raise NotImplementedError

def my_repair_llm(system: str, user: str) -> str:
    # Often same model; temperature=0; short max_tokens
    raise NotImplementedError

from guardrailed_grader import grade_with_guardrails

question = "What were the main causes of the French and Indian War?"
student_answer = "It started because the British and French fought over trade and territory in North America."
context = """Chapter 3 > 3.1 Imperial Rivalries (pp. 45–47)
- Conflict escalated over the Ohio River Valley...
- British colonists and French forces clashed due to overlapping land claims and alliances with Native nations...
"""

grade = grade_with_guardrails(
    question=question,
    student_answer=student_answer,
    context=context,
    call_llm=my_llm,
    max_retries=1,
    allow_repair=True,
    repair_llm=my_repair_llm,
)

print(grade.model_dump())
# -> {'score': 4, 'rationale': '...', 'criteria': ['Context-grounded','Covers territory & trade','Misses alliances'], 'citations': ['Chapter 3 > 3.1 (pp.45–47)']}
#!/usr/bin/env python3
"""
Aurora — CLI Evaluation Tester
================================
Two modes:

  1. QUICK (no DB, no Supabase):
     Paste context + question + answer directly in the terminal.
     Use this first to verify the grader + rubric work correctly.

     python cli_eval_test.py --mode quick

  2. FULL (reads from your Supabase DB):
     Picks a real document and section, generates a question,
     lets you type an answer, and runs the full pipeline.

     python cli_eval_test.py --mode full

Prerequisites:
  - Ollama running locally with your model (check config/constants.py)
  - For --mode full: Supabase configured in config/constants.py
"""

import argparse
import json
import sys
import textwrap

# ── Colour helpers (no deps) ────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
GREY   = "\033[90m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _print_result(result: dict) -> None:
    """Pretty-print a grade result dict (from grade.to_dict() or run_evaluation())."""
    score     = result["total_score"]
    max_score = result["max_score"]
    pct       = result["percentage"]
    level     = result["performance_level"]

    # Colour the level
    level_colour = {
        "Mastery":      GREEN,
        "Proficient":   YELLOW,
        "Needs Review": RED,
    }.get(level, RESET)

    print()
    print(_c(_hr("═"), BOLD))
    print(_c(f"  RESULT: {score}/{max_score}  ({pct}%)  —  {level}", level_colour + BOLD))
    print(_c(_hr("═"), BOLD))

    # Interpretation
    print(_c(f"\n  {result['interpretation']}\n", GREY))

    # Per-criterion breakdown
    print(_c("  CRITERIA BREAKDOWN", CYAN + BOLD))
    print(_c("  " + _hr(), GREY))
    for cs in result["criteria_scores"]:
        bar   = "█" * cs["score"] + "░" * (2 - cs["score"])
        label = _c(f"  {cs['id']} {cs['title']}", BOLD)
        score_str = _c(f"[{bar}] {cs['score']}/2", GREEN if cs["score"] == 2 else YELLOW if cs["score"] == 1 else RED)
        print(f"{label}  {score_str}")
        wrapped = textwrap.fill(cs["feedback"], width=70, initial_indent="     → ", subsequent_indent="       ")
        print(_c(wrapped, GREY))

    # Overall feedback
    print()
    print(_c("  OVERALL FEEDBACK", CYAN + BOLD))
    print(_c("  " + _hr(), GREY))
    wrapped_fb = textwrap.fill(result["overall_feedback"], width=68, initial_indent="  ", subsequent_indent="  ")
    print(wrapped_fb)

    # Citations
    if result.get("citations"):
        print()
        print(_c("  CITATIONS USED", CYAN + BOLD))
        for cit in result["citations"]:
            print(f"    • {cit}")

    # Section metadata (only present in full mode)
    if result.get("section_title"):
        print()
        print(_c(f"  Section : {result['section_title']}", GREY))
        print(_c(f"  Pages   : {result['page_range']}", GREY))

    print(_c("\n" + _hr("═"), BOLD))
    print()


# ── QUICK MODE ───────────────────────────────────────────────────────────────
QUICK_DEFAULTS = {
    "context": """\
Newton's Second Law of Motion states that the net force acting on an object
is equal to the product of its mass and its acceleration: F = ma.
This means that for a given force, a more massive object will accelerate less
than a lighter one. The SI unit of force is the Newton (N), defined as
1 kg·m/s². The law implies directionality: force and acceleration share
the same direction vector. Importantly, the law applies to net force —
the vector sum of all forces acting on the body.""",

    "question": "Explain Newton's Second Law and describe at least two implications of the equation F = ma.",

    "answer": "Newton's second law says force equals mass times acceleration. Heavier things are harder to push.",
}


def run_quick_mode(use_defaults: bool) -> None:
    print(_c("\n  AURORA — QUICK EVAL (no DB)\n", BOLD + CYAN))

    if use_defaults:
        context  = QUICK_DEFAULTS["context"]
        question = QUICK_DEFAULTS["question"]
        answer   = QUICK_DEFAULTS["answer"]
        print(_c("  Using built-in Newton's Laws example.\n", GREY))
    else:
        print(_c("  Paste your CONTEXT (the reference text). Enter a blank line when done:", BOLD))
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        context = "\n".join(lines)

        print(_c("\n  QUESTION:", BOLD))
        question = input("> ").strip()

        print(_c("\n  STUDENT ANSWER (press Enter twice to finish):", BOLD))
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        answer = "\n".join(lines)

    print(_c("\n  ⏳ Evaluating…", YELLOW))

    try:
        from src.pipeline import run_quick_check
        grade  = run_quick_check(context_text=context, question=question, student_answer=answer)
        result = grade.to_dict()
        _print_result(result)
    except Exception as e:
        print(_c(f"\n  ✗ Error during evaluation: {e}", RED))
        sys.exit(1)


# ── FULL MODE ────────────────────────────────────────────────────────────────
def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def _print_toc_tree(nodes: list, indent: int = 0) -> list:
    """Print tree and return flat list of (index, node_id, title) for selection."""
    flat = []
    for node in nodes:
        idx = len(flat)
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        print(f"  {_c(str(idx).rjust(3), CYAN)}  {prefix}{node['title']}")
        flat.append((idx, node["node_id"], node["title"]))
        if node.get("children"):
            flat.extend(_print_toc_tree(node["children"], indent + 1))
            # re-index after recursion — rebuild flat inline instead
    return flat


def _flat_toc(nodes: list, depth: int = 0) -> list:
    result = []
    for node in nodes:
        result.append((depth, node["node_id"], node["title"]))
        if node.get("children"):
            result.extend(_flat_toc(node["children"], depth + 1))
    return result


def run_full_mode() -> None:
    print(_c("\n  AURORA — FULL EVAL (Supabase + Ollama)\n", BOLD + CYAN))

    try:
        from src.retrieval import get_document_list, get_document_toc
        from src.assessment import generate_questions
        from src.pipeline import run_evaluation
    except ImportError as e:
        print(_c(f"  ✗ Import error: {e}", RED))
        print(_c("  Make sure you run this from the project root: python cli_eval_test.py --mode full", GREY))
        sys.exit(1)

    # 1. Pick document
    print(_c("  Fetching documents…", GREY))
    docs = get_document_list()
    if not docs:
        print(_c("  ✗ No documents found in Supabase. Upload one first.", RED))
        sys.exit(1)

    print(_c("\n  DOCUMENTS", BOLD))
    print(_c("  " + _hr(), GREY))
    for i, doc in enumerate(docs):
        sections = doc['total_sections']
        pages = doc['total_pages']
        print(f"  {_c(str(i).rjust(2), CYAN)}  {doc['title']}  "
              f"{_c(f'({sections} sections, {pages} pages)', GREY)}")

    doc_idx = int(input(_c("\n  Select document number: ", BOLD)))
    doc = docs[doc_idx]
    document_id = doc["id"]
    print(_c(f"\n  ✓ Selected: {doc['title']}", GREEN))

    # 2. Browse ToC
    print(_c("\n  Loading table of contents…", GREY))
    toc = get_document_toc(document_id)
    flat = _flat_toc(toc)

    print(_c("\n  TABLE OF CONTENTS", BOLD))
    print(_c("  " + _hr(), GREY))
    for i, (depth, node_id, title) in enumerate(flat):
        indent_str = "    " * depth
        print(f"  {_c(str(i).rjust(3), CYAN)}  {indent_str}{title}")

    section_idx = int(input(_c("\n  Select section number: ", BOLD)))
    _, node_id, section_title = flat[section_idx]
    print(_c(f"\n  ✓ Selected: {section_title}", GREEN))

    # 3. Generate or write question
    print(_c("\n  How do you want to set the question?", BOLD))
    print("    1  Generate with LLM (recommended)")
    print("    2  Write your own")
    q_choice = input(_c("\n  Choice [1/2]: ", BOLD)).strip()

    if q_choice != "2":
        print(_c("\n  ⏳ Generating question…", YELLOW))
        try:
            q_result  = generate_questions(document_id, node_id, num_questions=1, difficulty="medium")
            questions = q_result.get("questions", [])
            if not questions:
                raise ValueError("LLM returned no questions.")
            question = questions[0]["question"]
            print(_c(f"\n  Generated question:", BOLD))
            print(f"  {question}")
        except Exception as e:
            print(_c(f"\n  ⚠ Could not generate question ({e}). Enter one manually:", YELLOW))
            question = input("  > ").strip()
    else:
        print(_c("\n  Enter your question:", BOLD))
        question = input("  > ").strip()

    # 4. Student answer
    print(_c("\n  YOUR ANSWER (press Enter twice to submit):", BOLD))
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    student_answer = "\n".join(lines)

    if not student_answer.strip():
        print(_c("  ✗ Empty answer — aborting.", RED))
        sys.exit(1)

    # 5. Evaluate
    print(_c("\n  ⏳ Evaluating…", YELLOW))
    try:
        result = run_evaluation(
            document_id=document_id,
            node_id=node_id,
            question=question,
            student_answer=student_answer,
        )
        _print_result(result)
    except Exception as e:
        print(_c(f"\n  ✗ Evaluation error: {e}", RED))
        sys.exit(1)

    # 6. Optionally save result to JSON
    save = input(_c("  Save result to JSON? [y/N]: ", BOLD)).strip().lower()
    if save == "y":
        out_path = f"eval_result_{node_id[:20]}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(_c(f"  ✓ Saved to {out_path}", GREEN))


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aurora CLI — test the evaluation pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help=(
            "quick : no DB, uses built-in or pasted context (default)\n"
            "full  : reads from Supabase, generates question, full pipeline"
        ),
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="In quick mode, skip prompts and use the built-in Newton example",
    )
    args = parser.parse_args()

    if args.mode == "quick":
        run_quick_mode(use_defaults=args.defaults)
    else:
        run_full_mode()


if __name__ == "__main__":
    main()
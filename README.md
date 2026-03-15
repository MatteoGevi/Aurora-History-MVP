# Aurora — AI Learning Platform (MVP)

A personalised learning platform that lets students study PDF textbooks and test their retention using **free recall assessment** graded by Claude.

## How it works

1. **Ingest** — upload a PDF to Supabase Storage; the ingest pipeline extracts the Table of Contents, chunks the text by section, and stores everything in Supabase (text + embeddings).
2. **Study** — open the Streamlit app, select a document, browse the ToC, and read sections in the PDF viewer.
3. **Assess** — click any ToC section, write everything you remember about it from memory, and submit. Claude grades your recall against the actual section text using a 5-criterion analytic rubric (score 0–10).

---

## Project structure

```
Aurora-History-MVP/
├── app/
│   ├── app.py              # Streamlit app (PDF viewer + ToC + assessment UI)
│   └── components.py       # PDF rendering helpers
├── src/
│   ├── pipeline.py         # Entry point: run_section_recall() / run_evaluation()
│   ├── guardrailed_grader.py  # ASAG engine — rubric prompting, JSON validation, retry loop
│   ├── assessor_rubric.json   # 5-criterion rubric (C1–C5, max 10 pts)
│   ├── models.py           # LLM adapters (Claude, OpenAI, Ollama)
│   ├── retrieval.py        # Supabase queries (document list, ToC, section content)
│   └── assessment.py       # Question generation (unused in main flow)
├── ingest/
│   ├── toc_chunk.py        # PDF → ToC extraction + section chunking → Supabase
│   ├── ingest.py           # Orchestrates full ingest pipeline
│   └── embedding_import.py # Generates and stores embeddings
├── config/
│   └── constants.py        # All env vars and model settings
└── .env                    # Secrets (not committed)
```

---

## Setup

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure environment

Create a `.env` file at the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
STORAGE_BUCKET=textbook

CLAUDE_API_KEY=sk-ant-...
```

### 3. Ingest a PDF

Place your PDF in `ingest/data/` then run:

```bash
python -m ingest.ingest
```

This extracts the ToC, chunks the text by section, and uploads everything to Supabase.

---

## Running the app

```bash
cd app
streamlit run app.py
```

---

## Testing the assessment pipeline from the CLI

```bash
python -m src.pipeline
```

Interactive CLI that walks you through: pick a document → pick a section → write your recall → get graded.

---

## Assessment rubric

| Criterion | What it measures |
|-----------|-----------------|
| C1 — Factual correctness | Are the facts right? |
| C2 — Coverage | Did you cover all major parts of the section? |
| C3 — Terminology | Did you use the right terms from the text? |
| C4 — Logical coherence | Is it organised and explained, not just listed? |
| C5 — Depth of understanding | Do you explain *why/how*, not just *what*? |

Each criterion scored 0–2. Total max: **10 points**.

| Score | Level |
|-------|-------|
| 9–10 | Mastery |
| 6–8 | Proficient |
| 0–5 | Needs Review |

---

## Model configuration

Set `CLAUDE_MODEL_NAME` in `config/constants.py`:

| Model | Use case |
|-------|----------|
| `claude-haiku-4-5` | Fast and cheap — good for testing |
| `claude-opus-4-6` | Best quality — recommended for production |

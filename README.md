# Aurora-SLM
Mistral Local Model for the MVP. Good luck!! We need to start with a minimal, self-hostable stack to serve subject-specific Mistral SLMs behind a single OpenAI-compatible gateway. Includes:

- FastAPI gateway with routing by model or metadata.subject (aliases: @stable, @canary).

- vLLM backends per subject (Math & Literature examples), ready for LoRA/QLoRA adapters.

- Optional Qdrant for RAG and a tiny search service stub.

- docker-compose for local dev + Makefile tasks.

- Golden-set eval script.

## Repository Structure

edu-llm/
├─ gateway/                         # Python FastAPI – single OpenAI‑compatible entrypoint
│  ├─ app.py                        # routes by model name or subject
│  ├─ registry.yml                  # {model_name → backend url, version, health}
│  ├─ routes.yml                    # {subject → model_name@alias}
│  ├─ schemas/                      # JSON schemas for tool/JSON outputs
│  ├─ tests/
│  └─ Dockerfile
├─ backends/                        # One service per subject/type (vLLM/TGI/Ollama)
│  ├─ math/
│  │  ├─ vllm.yaml                  # launch args (model path, quant, max len, etc.)
│  │  ├─ adapters/                  # LoRA/QLoRA weights (tiny files)
│  │  ├─ Dockerfile
│  │  └─ README.md
│  ├─ stem/
│  └─ lit/
├─ models/                          # Pointers or scripts to fetch base checkpoints
│  ├─ mistral-7b/README.md
│  └─ fetch_model.sh                # pulls weights from HF/S3 (not committed)
├─ rag/                             # Optional shared RAG service
│  ├─ ingest/                       # dbt manifest, PDFs, etc.
│  ├─ vectorstore/                  # Qdrant config & migrations
│  ├─ service/                      # FastAPI “/search” microservice
│  └─ Dockerfile
├─ deployments/
│  ├─ docker-compose.yml            # local/dev: gateway + qdrant + 2–3 backends
│  ├─ k8s/                          # prod manifests/Helm values
│  └─ skypilot/                     # optional cloud one‑liners
├─ ci/
│  ├─ eval.yaml                     # run golden-set evals on PR
│  └─ build.yaml                    # build & push images per folder
├─ evals/
│  ├─ math_golden.csv               # ~50–200 Q/A + citations
│  └─ eval.py                       # runs /answer and scores exactness
├─ config/
│  ├─ .env.example
│  └─ logging.yaml
├─ Makefile                         # make dev, up, test, eval, deploy
└─ README.md

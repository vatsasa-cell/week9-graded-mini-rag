# Week 9 Graded Mini Project — Mini-RAG

This is a standalone RAG system over the five Sherlock Holmes story summaries supplied with the course. It uses OpenAI `text-embedding-3-small`, Qdrant vector search, and `gpt-4o-mini` answer generation. It intentionally does not use LangChain or LlamaIndex.

## Included files

- `mp2_rag.py` — end-to-end ingestion, retrieval, answer generation, and validation CLI.
- `corpus/` — five course-provided story summaries.
- `data/predefined_questions.jsonl` — two course-provided validation questions.
- `data/learner_questions.jsonl` — three authored questions (easy, medium, harder).
- `mp2_reflection.md` — reflection based on the actual ingestion and validation results.
- `mp2_validation.txt` — actual validation output from the completed run.
- `requirements.txt`, `.env.example`, `.gitignore` — setup files.

## Requirements

- Python 3.10+
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`
- A Qdrant Cloud URL and API key, or a reachable Qdrant instance

## Setup

### Linux / macOS / Vocareum terminal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the example values with your own credentials. Do not share or commit `.env`. The script loads `.env` using `python-dotenv`, so `source .env` is not required.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your credentials.

## Run the pipeline

From this project directory:

```bash
python mp2_rag.py ingest
python mp2_rag.py ask
python mp2_rag.py validate
```

1. `ingest` reads all `.txt` stories, chunks them, embeds each chunk, recreates the project-specific Qdrant collection, and upserts vectors and metadata. Re-running `ingest` replaces this project's collection.
2. `ask` starts an interactive question loop. Answers are generated from retrieved excerpts and cite story title and section.
3. `validate` checks source matches for the two predefined questions and the three learner questions. A `PASS` means the expected source file appeared in the retrieved citations; it does **not** guarantee every expected fact is present or the answer is fully correct. Review the printed answer and fact phrase counts.

## Submission checklist

- [x] Run `ingest` successfully.
- [x] Run `validate` and inspect all five answers.
- [x] Save the actual terminal output in `mp2_validation.txt` (do not submit a fabricated log).
- [x] Personalize `mp2_reflection.md` with observations from your real run.
- [ ] Keep `.env` out of the submission zip and Git history.
- [ ] If pushing to GitHub, use a separate personal repository as required by the brief.

## Cost and troubleshooting

The brief estimates roughly $0.05–$0.20 in OpenAI spend for the project, depending on runs. Set a suitable usage limit. If Qdrant appears asleep, wait about a minute and retry. If answers cite the wrong story, inspect chunk boundaries, retrieved chunks, and the question's expected source. Never paste API keys into chat or commit them.

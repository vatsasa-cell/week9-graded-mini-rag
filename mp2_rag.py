"""Week 9 Mini Project 2 — standalone Mini-RAG over Sherlock Holmes stories.

Commands:
    python mp2_rag.py ingest
    python mp2_rag.py ask
    python mp2_rag.py validate

Environment variables (or .env):
    OPENAI_API_KEY
    QDRANT_URL
    QDRANT_API_KEY (optional for local Qdrant)

Uses OpenAI directly and Qdrant directly; no LangChain or LlamaIndex.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

load_dotenv()

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mp2_sherlock_gorthi")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBED_BATCH_SIZE = 64

def get_clients() -> tuple[OpenAI, QdrantClient]:
    """Create clients only when a command actually needs them."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or the environment.")
    if not os.getenv("QDRANT_URL"):
        raise RuntimeError("QDRANT_URL is missing. Set it in .env or the environment.")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
    )
    return client, qdrant

def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    """Read every .txt story, preserving filename and title."""
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        documents.append({"source": path.name, "title": title, "text": text})
    if not documents:
        raise FileNotFoundError(f"No .txt stories found in {corpus_dir}")
    return documents

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk by paragraph where possible, with a fixed-window fallback and overlap."""
    text = str(doc["text"]).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0
    section = str(doc["title"])

    def emit(value: str, section_name: str) -> None:
        value = value.strip()
        if value:
            chunks.append({
                "source": doc["source"], "title": doc["title"],
                "section": section_name or doc["title"], "text": value,
            })

    for paragraph in paragraphs:
        # In these summaries, short standalone lines commonly denote section headings.
        if len(paragraph) <= 90 and "\n" not in paragraph and not paragraph.endswith((".", "!", "?", ",", ";", ":")):
            if current:
                emit("\n\n".join(current), section)
                current, current_len = [], 0
            section = paragraph
            continue
        if len(paragraph) > TARGET_CHUNK_SIZE:
            if current:
                emit("\n\n".join(current), section)
                current, current_len = [], 0
            start = 0
            while start < len(paragraph):
                piece = paragraph[start:start + TARGET_CHUNK_SIZE]
                emit(piece, section)
                if start + TARGET_CHUNK_SIZE >= len(paragraph):
                    break
                start += TARGET_CHUNK_SIZE - CHUNK_OVERLAP
            continue
        proposed_len = current_len + len(paragraph) + (2 if current else 0)
        if current and proposed_len > TARGET_CHUNK_SIZE:
            emit("\n\n".join(current), section)
            # Keep a short tail as overlap so adjacent chunks retain context.
            tail = current[-1][-CHUNK_OVERLAP:] if current else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_len = sum(map(len, current)) + 2 * (len(current) - 1)
        else:
            current.append(paragraph)
            current_len = proposed_len
    if current:
        emit("\n\n".join(current), section)
    return chunks

def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Create embeddings in batches while preserving input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
    return vectors

def setup_collection(qdrant: QdrantClient) -> None:
    """Recreate this project's collection; ingestion is intentionally repeatable."""
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )

def ingest_chunks(client: OpenAI, qdrant: QdrantClient, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("No chunks to ingest.")
    vectors = embed_texts(client, [c["text"] for c in chunks])
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={**chunk, "chunk_index": i, "metadata_version": "mp2-v1"},
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    for start in range(0, len(points), 100):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[start:start + 100], wait=True)

def retrieve(client: OpenAI, qdrant: QdrantClient, query: str, k: int = 3) -> list[dict[str, Any]]:
    """Embed a query and retrieve the top-k Qdrant chunks."""
    query = query.strip()
    if not query:
        raise ValueError("Question cannot be empty.")
    vector = embed_texts(client, [query])[0]
    # query_points is the current Qdrant client API.
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    results = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload["score"] = float(point.score or 0.0)
        results.append(payload)
    return results

SYSTEM_PROMPT = """You answer questions using only the provided excerpts from a small
collection of Sherlock Holmes story summaries. Do not use outside knowledge or invent
details. If the excerpts do not contain enough evidence, say that the supplied stories
do not establish the answer. Cite claims with the story title and section in square
brackets, for example [The Red-Headed League — The Vacancy in the Red-Headed League]."""

def answer(client: OpenAI, qdrant: QdrantClient, question: str, k: int = 3) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = retrieve(client, qdrant, question, k=k)
    context = "\n\n".join(
        f"[Source: {c.get('title', c.get('source', 'Unknown'))} — {c.get('section', 'Unknown section')}; "
        f"file={c.get('source', 'unknown')}]\n{c.get('text', '')}"
        for c in chunks
    )
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved excerpts:\n{context}\n\nAnswer with citations."},
        ],
    )
    answer_text = completion.choices[0].message.content or "The model returned an empty answer."
    # Unique citations preserve ranked order; source names are used by the supplied validator.
    citations = []
    seen = set()
    for c in chunks:
        key = (c.get("source"), c.get("title"), c.get("section"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": c.get("source", "unknown"),
                "title": c.get("title", c.get("source", "Unknown")),
                "section": c.get("section", "Unknown section"),
                "score": c.get("score", 0.0),
            })
    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "retrieved_chunks": chunks,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }

def validate_against(client: OpenAI, qdrant: QdrantClient, jsonl_path: Path) -> tuple[int, int]:
    questions = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\nValidating {len(questions)} questions from {jsonl_path.name}...\n")
    hits = 0
    for q in questions:
        result = answer(client, qdrant, q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources
        ans_lower = result["answer"].lower()
        facts = q.get("expected_facts", [])
        facts_hit = sum(1 for fact in facts if fact.lower() in ans_lower)
        verdict = "PASS" if source_hit else "CHECK"
        print(f"[{verdict}] {q['id']}")
        print(f"  Q: {q['question']}")
        print(f"  Cited files: {', '.join(sorted(cited_sources)) or '(none)'}")
        print(f"  Expected file: {q['expected_source']}")
        print(f"  Expected fact phrases present: {facts_hit}/{len(facts)}")
        print(f"  Latency: {result['latency_ms']} ms")
        print(f"  Answer: {result['answer']}\n")
        hits += int(source_hit)
    print(f"Source-match: {hits}/{len(questions)}")
    return hits, len(questions)

def cmd_ingest() -> None:
    client, qdrant = get_clients()
    try:
        docs = load_corpus()
        chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
        print(f"Loaded {len(docs)} documents and created {len(chunks)} chunks.")
        setup_collection(qdrant)
        ingest_chunks(client, qdrant, chunks)
        print(f"Successfully ingested {len(chunks)} chunks into Qdrant collection '{COLLECTION_NAME}'.")
    finally:
        qdrant.close()

def cmd_ask() -> None:
    client, qdrant = get_clients()
    try:
        print("Mini-RAG over Sherlock Holmes story summaries. Enter a question; blank line exits.")
        while True:
            try:
                question = input("\n? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                break
            result = answer(client, qdrant, question)
            print(f"\n{result['answer']}\n")
            print("Sources:")
            for citation in result["citations"]:
                print(f" - {citation['title']} — {citation['section']} ({citation['source']})")
            print(f"Latency: {result['latency_ms']} ms")
    finally:
        qdrant.close()

def cmd_validate() -> None:
    client, qdrant = get_clients()
    try:
        total_hits, total = validate_against(client, qdrant, DATA_DIR / "predefined_questions.jsonl")
        learner_path = DATA_DIR / "learner_questions.jsonl"
        if learner_path.exists():
            questions = [json.loads(line) for line in learner_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if questions and not questions[0]["question"].startswith("Replace this"):
                learner_hits, learner_total = validate_against(client, qdrant, learner_path)
                total_hits += learner_hits
                total += learner_total
        print(f"\nOVERALL source-match: {total_hits}/{total}")
    finally:
        qdrant.close()

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__)
        return
    command = sys.argv[1].lower()
    commands = {"ingest": cmd_ingest, "ask": cmd_ask, "validate": cmd_validate}
    if command not in commands:
        print(f"Unknown command: {command}. Choose ingest, ask, or validate.", file=sys.stderr)
        raise SystemExit(2)
    try:
        commands[command]()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()

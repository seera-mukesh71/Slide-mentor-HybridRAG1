"""rag_pipeline.py - connects ingestion, chapter-first retrieval, grounded generation, memory and evaluation.

    ask():  rewrite follow-up -> identify chapter(s) -> HyDE -> scoped BM25 + vector -> RRF -> rerank
            -> top chunks -> grounded answer -> self-critique (+1 regeneration) -> validated citations
"""
from __future__ import annotations

import dataclasses
import logging
import random
import time

from pydantic import BaseModel, Field

import generation as gen
import ingestion
import retrieval
import storage
from storage import Scope, Store

log = logging.getLogger(__name__)

HISTORY_MESSAGES = 5  # recent chat messages used to rewrite follow-up questions

# (label shown in the UI, retrieval mode)
EVAL_SYSTEMS = [
    ("BM25", "bm25"),
    ("Vector search", "vector"),
    ("Hybrid (BM25 + vector, RRF)", "hybrid"),
    ("Hybrid + HyDE + rerank", "full"),
]


class EvalQuestion(BaseModel):
    question: str = Field(description="One exam-style question answered by the passage")


class RAGPipeline:
    def __init__(self, store: Store, ensure_indexes: bool = True):
        self.store = store
        if ensure_indexes:
            store.ensure_indexes(gen.EMBED_DIM)

    # ---- documents ------------------------------------------------------- #
    def ingest(self, data: bytes, filename: str, progress=None) -> dict:
        return ingestion.ingest_pdf(self.store, data, filename, progress)

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)

    # ---- question answering --------------------------------------------- #
    def ask(self, question: str, session_id: str, document_ids: list[str]) -> dict:
        """Answer one question. Returns the assistant message ``{"content", "meta"}`` (also saved to MongoDB)."""
        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")
        if not document_ids:
            raise ValueError("Select at least one textbook to search.")
        t0 = time.time()

        history = self.store.get_recent_messages(session_id, HISTORY_MESSAGES)  # before this question is stored
        standalone = retrieval.rewrite_question(question, history, HISTORY_MESSAGES)

        ctx = retrieval.QueryContext(standalone, self.store.list_chapters(document_ids))
        ret = retrieval.retrieve(self.store, ctx, document_ids, mode="full")
        sources = retrieval.build_sources(ret.hits)
        scope_desc = retrieval.describe_scope(ret)
        answer = gen.generate_grounded_answer(standalone, sources, scope_desc)

        sel = ret.selection
        meta = {
            "status": answer.status,
            "explanation": answer.explanation,
            "citations": [dataclasses.asdict(c) for c in answer.citations],
            "closest": [dataclasses.asdict(c) for c in answer.closest],
            "trace": {
                "original_question": question,
                "standalone_question": standalone,
                "chapter_method": sel.method,
                "chapter_note": sel.note,
                "chapters": [
                    {"chapter_id": c["chapter_id"], "number": c.get("chapter_number"), "name": c["chapter_name"], "document": c["document_name"]}
                    for c in sel.chapters
                ],
                "scope_chapter_ids": list(ret.scope.chapter_ids or []),
                "scope_description": scope_desc,
                "fallback": ret.fallback,
                "fallback_reason": ret.fallback_reason,
                "hyde_passage": ret.hyde_passage,  # shown for transparency only - never used as evidence
                "hyde_error": ctx.hyde_error,
                "sources": [{"sid": s.sid, "chunk_id": s.chunk["chunk_id"], "score": s.score, "pages": s.pages()} for s in sources],
                "critique": answer.critique,
                "regenerated": answer.regenerated,
                "problems": answer.problems,
                "seconds": round(time.time() - t0, 1),
            },
        }
        self.store.add_message(session_id, "user", question)
        self.store.add_message(session_id, "assistant", answer.text, meta)
        return {"content": answer.text, "meta": meta}

    # ---- evaluation ------------------------------------------------------ #
    def build_eval_set(self, document_id: str, n: int = 15, seed: int = 7, progress=None) -> list[dict]:
        """Synthetic test set: Gemini writes one question per sampled passage; that passage is the known relevant chunk."""
        chunks = [c for c in self.store.get_chunks(Scope((document_id,))) if 50 <= c.get("word_count", 0) <= 300]
        if not chunks:
            raise ValueError("This book has no passages suitable for generating test questions.")
        rng = random.Random(seed)
        by_chapter: dict[str, list[dict]] = {}
        for c in chunks:
            by_chapter.setdefault(c["chapter_id"], []).append(c)
        for lst in by_chapter.values():
            rng.shuffle(lst)
        picked: list[dict] = []
        while len(picked) < min(n, len(chunks)):  # round-robin so every chapter is covered
            for lst in by_chapter.values():
                if lst and len(picked) < n:
                    picked.append(lst.pop())
        items = []
        for i, c in enumerate(picked):
            prompt = (
                "Write ONE exam-style question that a student could answer using only the passage below. Rules: paraphrase "
                "instead of copying phrases (never reuse more than 3 consecutive words), do not mention 'the passage' or "
                "'the text', and make the question understandable on its own.\n\n"
                f"CHAPTER: {c['chapter_name']}\nPASSAGE:\n{c['text']}"
            )
            q = gen.generate_json(prompt, EvalQuestion, temperature=0.7).question.strip()
            mentions = i % 3 == 2 and c.get("chapter_number") is not None  # every third question names its chapter, like real students do
            if mentions:
                q = f"In chapter {c['chapter_number']}, {q[0].lower() + q[1:] if q[1:2].islower() else q}"
            items.append({"id": f"q{i + 1}", "question": q, "relevant_chunk_ids": [c["chunk_id"]], "chapter_id": c["chapter_id"], "mentions_chapter": mentions})
            if progress:
                progress(i + 1, len(picked))
        self.store.save_eval_set(document_id, items)
        return items

    def import_eval_set(self, document_id: str, items: list[dict]) -> list[dict]:
        """Validate and store a user-supplied dataset: [{"question": str, "relevant_chunk_ids": [str]}]."""
        clean = []
        for i, it in enumerate(items, start=1):
            if not isinstance(it, dict) or not str(it.get("question", "")).strip() or not it.get("relevant_chunk_ids"):
                raise ValueError(f"Item {i} needs a non-empty 'question' and 'relevant_chunk_ids'.")
            clean.append({"id": it.get("id", f"q{i}"), "question": str(it["question"]).strip(), "relevant_chunk_ids": [str(x) for x in it["relevant_chunk_ids"]]})
        known = {c["chunk_id"] for c in self.store.get_chunks_by_ids({x for it in clean for x in it["relevant_chunk_ids"]})}
        unknown = sorted({x for it in clean for x in it["relevant_chunk_ids"]} - known)
        if unknown:
            raise ValueError(f"Unknown chunk ids (they must belong to an uploaded book): {unknown[:5]}")
        self.store.save_eval_set(document_id, clean)
        return clean

    def evaluate(self, items: list[dict], document_ids: list[str], use_chapter_filter: bool = True, progress=None) -> dict:
        """Run the four retrieval systems on the same questions with the same chapter-filter setting."""
        chapters = self.store.list_chapters(document_ids)
        runs: dict[str, list] = {label: [] for label, _ in EVAL_SYSTEMS}
        stats = {"identified": 0, "correct": 0, "fallback": 0}
        for n, item in enumerate(items, start=1):
            ctx = retrieval.QueryContext(item["question"], chapters, use_chapter_filter)
            gold = {c["chapter_id"] for c in self.store.get_chunks_by_ids(item["relevant_chunk_ids"])}
            for label, mode in EVAL_SYSTEMS:
                ret = retrieval.retrieve(self.store, ctx, document_ids, mode)
                runs[label].append(([h.chunk["chunk_id"] for h in ret.hits[:10]], item["relevant_chunk_ids"]))
                if mode == "full":
                    if ret.selection.chapter_ids:
                        stats["identified"] += 1
                        stats["correct"] += bool(gold & set(ret.selection.chapter_ids))
                    stats["fallback"] += ret.fallback
            if progress:
                progress(n, len(items))
        total = len(items) or 1
        return {
            "metrics": {label: storage.evaluate_rankings(r) for label, r in runs.items()},
            "chapter_filter": (
                {
                    "identified_rate": stats["identified"] / total,
                    "correct_when_identified": (stats["correct"] / stats["identified"]) if stats["identified"] else None,
                    "fallback_rate": stats["fallback"] / total,
                }
                if use_chapter_filter else None
            ),
            "questions": len(items),
        }

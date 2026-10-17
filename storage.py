"""storage.py - MongoDB persistence, conversation memory and retrieval-evaluation metrics.

Collections
-----------
documents  one record per uploaded textbook (metadata + chapter outline)
chunks     text chunks, each tagged with its chapter, + embeddings (Atlas Vector Search index: ``chunks_vector``)
messages   conversation history keyed by ``session_id``
eval_sets  evaluation questions with known relevant chunk ids
pdfs.*     GridFS copy of each original PDF (needed by the page viewer)
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import gridfs
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

load_dotenv()
log = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "chunks_vector"
INSERT_BATCH = 500


class StorageError(RuntimeError):
    """Raised for connection / configuration problems with MongoDB."""


# --------------------------------------------------------------------------- #
# Retrieval scope
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scope:
    """Which chunks are eligible for retrieval.

    ``chapter_ids=None`` means "every chunk of ``document_ids``" (the fallback scope).
    The very same Scope object is used for BM25 (``Store.get_chunks``) and for
    vector search (``Store.vector_search``), which keeps both retrievers consistent.
    """

    document_ids: tuple[str, ...]
    chapter_ids: tuple[str, ...] | None = None

    def mongo_filter(self) -> dict:
        clauses: list[dict] = [{"document_id": {"$in": list(self.document_ids)}}]
        if self.chapter_ids:
            clauses.append({"chapter_id": {"$in": list(self.chapter_ids)}})
        return {"$and": clauses}


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, db: Any = None, uri: str | None = None, db_name: str | None = None):
        if db is None:
            uri = uri or os.getenv("MONGODB_URI")
            if not uri:
                raise StorageError(
                    "MONGODB_URI is not set. Copy .env.example to .env and add your MongoDB Atlas connection string."
                )
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=8000)
                client.admin.command("ping")
            except PyMongoError as e:
                raise StorageError(
                    f"Could not connect to MongoDB ({e.__class__.__name__}). Check MONGODB_URI, the database user's "
                    "password and the Atlas Network Access (IP allow-list)."
                ) from e
            db = client[db_name or os.getenv("MONGODB_DB", "student_rag")]
        self.db = db
        self.documents = db["documents"]
        self.chunks = db["chunks"]
        self.messages = db["messages"]
        self.eval_sets = db["eval_sets"]
        self.fs = gridfs.GridFS(db, collection="pdfs")

    # ---- setup ----------------------------------------------------------- #
    def ensure_indexes(self, dim: int) -> None:
        """Create regular indexes and the Atlas vector index (idempotent)."""
        try:
            self.documents.create_index("document_id", unique=True)
            self.chunks.create_index([("document_id", ASCENDING), ("chapter_id", ASCENDING)])
            self.chunks.create_index("chunk_id", unique=True)
            self.messages.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
            self.eval_sets.create_index("document_id", unique=True)
            if not list(self.chunks.list_search_indexes(VECTOR_INDEX_NAME)):
                self.chunks.create_search_index(
                    SearchIndexModel(
                        name=VECTOR_INDEX_NAME,
                        type="vectorSearch",
                        definition={
                            "fields": [
                                {"type": "vector", "path": "embedding", "numDimensions": dim, "similarity": "cosine"},
                                # filter fields => chapter/document filtering happens INSIDE $vectorSearch
                                {"type": "filter", "path": "document_id"},
                                {"type": "filter", "path": "chapter_id"},
                            ]
                        },
                    )
                )
                log.info("Created Atlas vector index %s (it takes ~1 minute to become queryable)", VECTOR_INDEX_NAME)
        except PyMongoError as e:
            raise StorageError(
                "Could not prepare MongoDB indexes. Vector Search needs a MongoDB Atlas cluster "
                f"(the free M0 tier works) and a database user with readWrite access. Details: {e}"
            ) from e

    def vector_index_status(self) -> str:
        """'ready' | 'building' | 'missing'."""
        try:
            idx = list(self.chunks.list_search_indexes(VECTOR_INDEX_NAME))
        except PyMongoError:
            return "missing"
        if not idx:
            return "missing"
        return "ready" if idx[0].get("queryable") else "building"

    # ---- documents ------------------------------------------------------- #
    def get_document(self, document_id: str) -> dict | None:
        return self.documents.find_one({"document_id": document_id}, {"_id": 0})

    def list_documents(self) -> list[dict]:
        proj = {"_id": 0, "chapters": 0}
        return list(self.documents.find({}, proj).sort("uploaded_at", ASCENDING))

    def save_document(self, record: dict) -> None:
        self.documents.replace_one({"document_id": record["document_id"]}, record, upsert=True)

    def list_chapters(self, document_ids: Sequence[str]) -> list[dict]:
        """Flat list of chapter records (actual names stored at ingestion time)."""
        by_doc = {
            d["document_id"]: d
            for d in self.documents.find({"document_id": {"$in": list(document_ids)}}, {"_id": 0})
        }
        out: list[dict] = []
        for doc_id in document_ids:  # keep the caller's document order
            d = by_doc.get(doc_id)
            if not d:
                continue
            for ch in d.get("chapters", []):
                out.append(
                    {
                        "document_id": doc_id,
                        "document_name": d["name"],
                        "chapter_id": ch["chapter_id"],
                        "chapter_number": ch.get("number"),
                        "chapter_name": ch["name"],
                        "chapter_title": ch.get("title") or ch["name"],
                    }
                )
        return out

    def delete_document(self, document_id: str) -> None:
        self.chunks.delete_many({"document_id": document_id})
        self.eval_sets.delete_many({"document_id": document_id})
        for f in list(self.fs.find({"filename": document_id})):
            self.fs.delete(f._id)
        self.documents.delete_many({"document_id": document_id})

    # ---- pdf bytes ------------------------------------------------------- #
    def save_pdf(self, document_id: str, data: bytes) -> None:
        for f in list(self.fs.find({"filename": document_id})):
            self.fs.delete(f._id)
        self.fs.put(data, filename=document_id, content_type="application/pdf")

    def load_pdf(self, document_id: str) -> bytes | None:
        f = self.fs.find_one({"filename": document_id})
        return f.read() if f else None

    # ---- chunks ---------------------------------------------------------- #
    def insert_chunks(self, chunks: Sequence[dict]) -> None:
        for i in range(0, len(chunks), INSERT_BATCH):
            self.chunks.insert_many([dict(c) for c in chunks[i : i + INSERT_BATCH]])

    def get_chunks(self, scope: Scope) -> list[dict]:
        """All chunks inside ``scope`` (no embeddings) - the BM25 corpus."""
        return list(self.chunks.find(scope.mongo_filter(), {"_id": 0, "embedding": 0}).sort("order", ASCENDING))

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> list[dict]:
        return list(self.chunks.find({"chunk_id": {"$in": list(chunk_ids)}}, {"_id": 0, "embedding": 0}))

    def vector_search(
        self, query_vector: Sequence[float], scope: Scope, limit: int = 30, num_candidates: int | None = None
    ) -> list[dict]:
        """Atlas ``$vectorSearch`` with the chapter/document filter applied *inside* the search stage."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": list(query_vector),
                    "numCandidates": max(num_candidates or limit * 10, limit),
                    "limit": limit,
                    "filter": scope.mongo_filter(),
                }
            },
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"_id": 0, "embedding": 0}},
        ]
        try:
            return list(self.chunks.aggregate(pipeline))
        except PyMongoError as e:
            raise StorageError(
                "Vector search failed. If you just uploaded a book, the Atlas vector index may still be building "
                f"(wait about a minute). Details: {e}"
            ) from e

    # ---- conversation memory --------------------------------------------- #
    def add_message(self, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
        self.messages.insert_one(
            {"session_id": session_id, "role": role, "content": content, "meta": meta or {}, "ts": time.time()}
        )

    def get_messages(self, session_id: str) -> list[dict]:
        return list(self.messages.find({"session_id": session_id}, {"_id": 0}).sort([("ts", 1)]))

    def get_recent_messages(self, session_id: str, n: int = 5) -> list[dict]:
        msgs = list(self.messages.find({"session_id": session_id}, {"_id": 0}).sort([("ts", -1)]).limit(n))
        return list(reversed(msgs))

    def clear_session(self, session_id: str) -> None:
        self.messages.delete_many({"session_id": session_id})

    # ---- evaluation datasets --------------------------------------------- #
    def save_eval_set(self, document_id: str, items: list[dict]) -> None:
        self.eval_sets.replace_one({"document_id": document_id}, {"document_id": document_id, "items": items}, upsert=True)

    def get_eval_set(self, document_id: str) -> list[dict]:
        rec = self.eval_sets.find_one({"document_id": document_id}, {"_id": 0})
        return rec["items"] if rec else []


# --------------------------------------------------------------------------- #
# Retrieval evaluation metrics (binary relevance)
# --------------------------------------------------------------------------- #
def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, cid in enumerate(ranked, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(runs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> dict[str, float]:
    """``runs`` = [(ranked_chunk_ids, relevant_chunk_ids), ...] -> averaged metrics."""
    n = len(runs)
    if n == 0:
        return {}
    out = {
        "Recall@3": sum(recall_at_k(r, rel, 3) for r, rel in runs) / n,
        "Recall@5": sum(recall_at_k(r, rel, 5) for r, rel in runs) / n,
        "Recall@10": sum(recall_at_k(r, rel, 10) for r, rel in runs) / n,
        "MRR": sum(reciprocal_rank(r, rel) for r, rel in runs) / n,
        "nDCG@5": sum(ndcg_at_k(r, rel, 5) for r, rel in runs) / n,
        "nDCG@10": sum(ndcg_at_k(r, rel, 10) for r, rel in runs) / n,
        "Empty rate": sum(1 for r, _ in runs if len(r) == 0) / n,
    }
    out["Questions"] = float(n)
    return out

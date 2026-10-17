"""retrieval.py - chapter-first filtering, HyDE, BM25, vector search, RRF, cross-encoder reranking.

Order of operations for one question (see ``retrieve``):
    1. identify chapter(s) from the chapter names stored at ingestion time
         number match -> normalised title match -> fuzzy match -> Gemini selection (only when needed)
    2. build ONE ``Scope`` (document ids + chapter ids) and use it for BOTH retrievers:
         BM25  over ``store.get_chunks(scope)``            (corpus is filtered *before* indexing)
         vector ``store.vector_search(vec, scope)``        (Atlas ``$vectorSearch`` pre-filter)
    3. RRF fusion -> cross-encoder rerank
    4. if the chapter-scoped result is insufficient, repeat once on all eligible chunks
    5. ``build_sources`` hands the best chunks (and only those) to the LLM

HyDE text is only ever used to compute an embedding for vector search. It is never returned as a Source.
"""
from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

import generation as gen
from generation import Source
from storage import Scope, Store

log = logging.getLogger(__name__)

TOP_K_EACH = 30  # candidates taken from each retriever
RRF_K = 60
RERANK_POOL = 30  # fused candidates sent to the cross-encoder
RESULT_K = 20  # ranked chunks returned by retrieve()
CONTEXT_K = 8  # chunks passed to the LLM
MIN_CANDIDATES = 3  # fewer than this inside a chapter scope => fall back to all eligible chunks
RERANK_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Cross-encoder logit below which a chapter-scoped result is treated as insufficient (=> broaden scope).
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

Mode = Literal["bm25", "vector", "hybrid", "full"]


# --------------------------------------------------------------------------- #
# Chapter identification
# --------------------------------------------------------------------------- #
@dataclass
class ChapterSelection:
    chapter_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    method: str = "none"  # number | title | fuzzy | gemini | none
    confident: bool = False
    chapters: list[dict] = field(default_factory=list)
    note: str = ""


_WORD_NUMS = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split(), start=1)}
_ORDINALS = {w: i for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(), start=1)}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_NUM = r"(?:\d{1,3}|[IVXLC]{1,6}|(?i:" + "|".join(_WORD_NUMS) + r"))"
_CHAPTER_REF = re.compile(
    r"(?i:\b(?:chapters?|chs?\.?|units?|lessons?|modules?))\s*#?\s*(" + _NUM + r"(?:\s*(?:,|\band\b|&|\bto\b|-|–)\s*" + _NUM + r")*)\b"
)
_ORDINAL_REF = re.compile(r"(?i)\b(?:(\d{1,2})(?:st|nd|rd|th)|(" + "|".join(_ORDINALS) + r"))\s+chapter\b")
_GENERIC_TITLES = {
    "introduction", "intro", "overview", "summary", "conclusion", "conclusions", "preface", "contents", "index",
    "references", "bibliography", "appendix", "exercises", "review", "problems", "glossary", "notes", "foreword", "background",
}
_STOP = {"the", "of", "and", "a", "an", "in", "to", "for", "on", "with", "is", "are", "what", "how", "why", "do", "does", "by", "as", "at", "or"}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def _content_tokens(norm: str) -> list[str]:
    return [t for t in norm.split() if t not in _STOP]


def _roman_to_int(s: str) -> int | None:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return total if 0 < total < 200 else None


def _to_number(tok: str) -> int | None:
    tok = tok.strip()
    if tok.isdigit():
        return int(tok)
    if tok.lower() in _WORD_NUMS:
        return _WORD_NUMS[tok.lower()]
    if tok and all(ch in _ROMAN for ch in tok):
        return _roman_to_int(tok)
    return None


def extract_chapter_numbers(question: str) -> list[int]:
    """'chapter 3', 'Ch. IV', 'chapters 2 and 5', 'chapters 2-4', 'the third chapter' -> [3] / [4] / [2, 5] / [2, 3, 4] / [3]."""
    found: list[int] = []
    for m in _CHAPTER_REF.finditer(question):
        parts = re.split(r"\s*(,|\band\b|&|\bto\b|-|–)\s*", m.group(1))
        nums = [_to_number(p) for p in parts[0::2]]
        seps = parts[1::2]
        i = 0
        while i < len(nums):
            cur = nums[i]
            if cur is None:
                i += 1
                continue
            if i < len(seps) and seps[i] in ("to", "-", "–") and i + 1 < len(nums) and nums[i + 1] and 0 < nums[i + 1] - cur <= 20:
                found.extend(range(cur, nums[i + 1] + 1))
                i += 2
            else:
                found.append(cur)
                i += 1
    for m in _ORDINAL_REF.finditer(question):
        found.append(int(m.group(1)) if m.group(1) else _ORDINALS[m.group(2).lower()])
    return list(dict.fromkeys(found))


def strip_chapter_references(question: str) -> str:
    """'In chapter 3, what is X?' -> 'In , what is X?' - the chapter reference is routing information, not search terms."""
    q = _ORDINAL_REF.sub(" ", _CHAPTER_REF.sub(" ", question))
    return re.sub(r"\s+", " ", q).strip()


def _selection(matches: list[dict], method: str, confident: bool = True, note: str = "") -> ChapterSelection:
    if not matches:
        return ChapterSelection(method="none", note=note)
    return ChapterSelection(
        chapter_ids=list(dict.fromkeys(c["chapter_id"] for c in matches)),
        document_ids=list(dict.fromkeys(c["document_id"] for c in matches)),
        method=method, confident=confident, chapters=matches, note=note,
    )


def _docs_mentioned(nq: str, chapters: list[dict]) -> set[str]:
    hit: set[str] = set()
    for did, name in {c["document_id"]: _normalize(c["document_name"]) for c in chapters}.items():
        if len(name) >= 6 and (f" {name} " in f" {nq} " or (len(name) <= len(nq) and fuzz.partial_ratio(name, nq) >= 92)):
            hit.add(did)
    return hit


def _is_distinctive(norm_title: str) -> bool:
    return len(_content_tokens(norm_title)) >= 2 or len(norm_title) >= 8


def _drop_subsumed(matches: list[dict]) -> list[dict]:
    """If 'energy' and 'work power and energy' both match, keep only the more specific chapter."""
    titles = {c["chapter_id"]: _normalize(c["chapter_title"]) for c in matches}
    keep = []
    for c in matches:
        t = titles[c["chapter_id"]]
        if not any(t != o and f" {t} " in f" {o} " for o in titles.values()):
            keep.append(c)
    return keep


def _match_by_number(question: str, nq: str, chapters: list[dict]) -> ChapterSelection | None:
    numbers = extract_chapter_numbers(question)
    if not numbers:
        return None
    matches = [c for c in chapters if c.get("chapter_number") in numbers]
    if not matches:
        return None
    mentioned = _docs_mentioned(nq, chapters)
    if mentioned:
        narrowed = [c for c in matches if c["document_id"] in mentioned]
        matches = narrowed or matches
    docs = {c["document_id"] for c in matches}
    note = f"Chapter number(s) {sorted(set(numbers))} matched in {len(docs)} book(s)."
    return _selection(matches, "number", True, note)


def _match_by_title(nq: str, chapters: list[dict]) -> ChapterSelection | None:
    explicit = bool(re.search(r"\b(chapter|chapters|ch|unit|lesson|module)\b", nq))
    hits = []
    for ch in chapters:
        t = _normalize(ch["chapter_title"])
        if len(t) < 4 or t in _GENERIC_TITLES:
            continue
        if not _is_distinctive(t) and not explicit:
            continue  # short titles like "Energy" are too easy to hit by accident
        if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", nq):
            hits.append(ch)
    hits = _drop_subsumed(hits)
    if not hits or len(hits) > 4:
        return None
    return _selection(hits, "title", True, "Chapter title found in the question.")


def _match_fuzzy(nq: str, chapters: list[dict]) -> ChapterSelection | None:
    qtoks = nq.split()
    scored = []
    for ch in chapters:
        t = _normalize(ch["chapter_title"])
        toks = _content_tokens(t)
        if not toks or t in _GENERIC_TITLES or not _is_distinctive(t):
            continue
        matched = 0
        for tok in toks:
            best = max((fuzz.ratio(tok, qt) for qt in qtoks), default=0)
            if best >= (85 if len(tok) >= 5 else 100):
                matched += 1
        coverage = matched / len(toks)
        if coverage >= 0.999 or (len(toks) >= 3 and coverage >= 0.75):
            scored.append((coverage, ch))
    hits = _drop_subsumed([c for _, c in sorted(scored, key=lambda x: -x[0])])
    if not hits or len(hits) > 4:
        return None
    return _selection(hits, "fuzzy", True, "Chapter title fuzzy-matched to the question.")


class ChapterChoice(BaseModel):
    chapter_ids: list[str] = Field(description="0-3 chapter ids from the list (like C4) that contain the answer")
    confidence: Literal["high", "low"] = Field(description="high only if you are sure; low if ambiguous, general or spanning many chapters")
    reason: str = Field(description="One short sentence")


def _outline(chapters: list[dict]) -> tuple[str, dict[str, dict]]:
    multi_doc = len({c["document_id"] for c in chapters}) > 1
    alias: dict[str, dict] = {}
    lines = []
    for i, ch in enumerate(chapters, start=1):
        alias[f"C{i}"] = ch
        parts = [f"C{i}"]
        if multi_doc:
            parts.append(ch["document_name"])
        parts.append(f"ch {ch['chapter_number']}" if ch.get("chapter_number") is not None else "ch -")
        parts.append(ch["chapter_title"])
        lines.append(" | ".join(parts))
    return "\n".join(lines), alias


def _select_with_gemini(question: str, chapters: list[dict]) -> ChapterSelection | None:
    chapters = chapters[:400]
    outline, alias = _outline(chapters)
    prompt = (
        "A student asks a question about their textbook(s). Choose which chapter(s) should be searched.\n\n"
        f"QUESTION: {question}\n\nCHAPTERS (id | [book] | number | title):\n{outline}\n\n"
        "Return up to 3 chapter ids that together contain the answer. If the question is general (e.g. 'summarise the book'), "
        "ambiguous, spans many chapters, or you are not sure, set confidence to \"low\" and return no ids."
    )
    try:
        choice = gen.generate_json(prompt, ChapterChoice, temperature=0.0)
    except gen.LLMError as e:
        log.warning("Gemini chapter selection failed: %s", e)
        return None
    ids = [alias[i] for i in dict.fromkeys(x.strip().upper() for x in choice.chapter_ids) if i in alias]  # drop invented ids
    if choice.confidence != "high" or not ids or len(ids) > 3:
        return ChapterSelection(method="none", note=f"Gemini was not confident: {choice.reason}")
    return _selection(ids, "gemini", True, choice.reason)


def identify_chapters(question: str, chapters: list[dict], use_gemini: bool = True) -> ChapterSelection:
    """Decide which chapter(s) to search. Returns method='none' (=> search everything) when unsure."""
    if len({c["chapter_id"] for c in chapters}) < 2:
        return ChapterSelection(method="none", note="No chapter structure to filter on.")
    nq = _normalize(question)
    for step in (lambda: _match_by_number(question, nq, chapters), lambda: _match_by_title(nq, chapters), lambda: _match_fuzzy(nq, chapters)):
        sel = step()
        if sel:
            return sel
    if use_gemini:
        sel = _select_with_gemini(question, chapters)
        if sel:
            return sel
    return ChapterSelection(method="none", note="No chapter could be identified confidently.")


# --------------------------------------------------------------------------- #
# Query rewriting and HyDE
# --------------------------------------------------------------------------- #
def rewrite_question(question: str, history: list[dict], max_messages: int = 5) -> str:
    """Turn an ambiguous follow-up into a standalone question using the last few chat messages."""
    recent = history[-max_messages:]
    if not recent:
        return question
    convo = "\n".join(f"{m['role'].capitalize()}: {str(m['content'])[:600]}" for m in recent)
    prompt = (
        "Rewrite the student's latest question so it can be understood without the conversation: resolve pronouns and "
        "ellipsis using the recent messages. If it is already self-contained, return it unchanged. "
        "Output only the question.\n\n"
        f"RECENT CONVERSATION:\n{convo}\n\nLATEST QUESTION: {question}"
    )
    try:
        out = gen.generate_text(prompt, temperature=0.0).strip().strip('"')
    except gen.LLMError as e:
        log.warning("Question rewrite failed, using the original question: %s", e)
        return question
    return out if 3 <= len(out) <= 600 else question


def generate_hyde(question: str) -> str:
    """A hypothetical answer passage, used ONLY to build a query embedding - never as evidence."""
    prompt = (
        "Write a short passage (about 120 words) in the style of a university textbook that would directly answer the "
        f"question below. Write it as if it were part of the book; do not mention that it is hypothetical.\n\nQUESTION: {question}"
    )
    return gen.generate_text(prompt, temperature=0.4)


class QueryContext:
    """Per-question state: chapter selection and query/HyDE embeddings, each computed at most once."""

    def __init__(self, question: str, chapters: list[dict], use_chapter_filter: bool = True):
        self.question = question
        self.chapters = chapters
        self.use_chapter_filter = use_chapter_filter
        self._selection: ChapterSelection | None = None
        self._qvec: list[float] | None = None
        self._hyde: str | None = None
        self._hvec: list[float] | None = None
        self.hyde_error: str = ""

    @property
    def selection(self) -> ChapterSelection:
        if self._selection is None:
            self._selection = (
                identify_chapters(self.question, self.chapters) if self.use_chapter_filter
                else ChapterSelection(method="none", note="Chapter filtering disabled.")
            )
        return self._selection

    def query_vec(self) -> list[float]:
        if self._qvec is None:
            self._qvec = gen.embed_texts([self.question], task_type="RETRIEVAL_QUERY")[0]
        return self._qvec

    @property
    def hyde_text(self) -> str:
        if self._hyde is None:
            try:
                self._hyde = generate_hyde(self.question)
            except gen.LLMError as e:
                self.hyde_error = str(e)
                self._hyde = ""
        return self._hyde

    def hyde_vec(self) -> list[float]:
        if self._hvec is None:
            if self.hyde_text:  # the hypothetical passage is embedded like a document it would match
                self._hvec = gen.embed_texts([self.hyde_text], task_type="RETRIEVAL_DOCUMENT")[0]
            else:  # HyDE unavailable -> degrade to the plain question embedding
                self._hvec = self.query_vec()
        return self._hvec


# --------------------------------------------------------------------------- #
# BM25, RRF, reranking
# --------------------------------------------------------------------------- #
def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", _normalize(text)) if t not in _STOP]


@dataclass
class Hit:
    chunk: dict
    score: float
    stage: str  # bm25 | vector | rrf | rerank


def bm25_search(query: str, chunks: list[dict], k: int = 30) -> list[tuple[str, float]]:
    """BM25 over exactly the chunks it is given (the caller has already applied the scope)."""
    q = tokenize(query)
    corpus = [tokenize(f"{c['chapter_name']} {c['text']}") for c in chunks]
    if not q or not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(q)
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:k]
    return [(chunks[i]["chunk_id"], float(scores[i])) for i in order if scores[i] > 0]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers is not installed. Run: pip install -r requirements.txt") from e
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = RESULT_K) -> list[Hit]:
    if not chunks:
        return []
    pairs = [(query, f"{c['chapter_name']}\n{c['text']}") for c in chunks]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))[:top_k]
    return [Hit(c, float(s), "rerank") for c, s in ranked]


# --------------------------------------------------------------------------- #
# Scoped search + chapter-first retrieval with fallback
# --------------------------------------------------------------------------- #
def search_scope(store: Store, ctx: QueryContext, scope: Scope, mode: Mode) -> list[Hit]:
    """Run one retrieval mode inside ``scope``. BM25 and vector search receive the identical scope."""
    by_id: dict[str, dict] = {}
    bm25_ranked: list[tuple[str, float]] = []
    vec_docs: list[dict] = []

    if mode in ("bm25", "hybrid", "full"):
        chunks = store.get_chunks(scope)  # only in-scope chunks ever reach BM25
        by_id.update({c["chunk_id"]: c for c in chunks})
        bm25_ranked = bm25_search(strip_chapter_references(ctx.question), chunks, TOP_K_EACH)
    if mode in ("vector", "hybrid", "full"):
        vec = ctx.hyde_vec() if mode == "full" else ctx.query_vec()
        vec_docs = store.vector_search(vec, scope, limit=TOP_K_EACH)
        for d in vec_docs:
            by_id.setdefault(d["chunk_id"], d)

    if mode == "bm25":
        return [Hit(by_id[cid], s, "bm25") for cid, s in bm25_ranked[:RESULT_K]]
    if mode == "vector":
        return [Hit(d, float(d.get("score", 0.0)), "vector") for d in vec_docs[:RESULT_K]]

    fused = rrf_fuse([[cid for cid, _ in bm25_ranked], [d["chunk_id"] for d in vec_docs]])
    if mode == "hybrid":
        return [Hit(by_id[cid], s, "rrf") for cid, s in fused[:RESULT_K]]
    pool = [by_id[cid] for cid, _ in fused[:RERANK_POOL]]
    return rerank(ctx.question, pool, RESULT_K)  # rerank against the real question, not the HyDE text


def _sufficient(hits: list[Hit]) -> tuple[bool, str]:
    if len(hits) < MIN_CANDIDATES:
        return False, f"only {len(hits)} candidate(s) inside the chapter scope"
    if hits[0].stage == "rerank" and not math.isnan(MIN_RERANK_SCORE) and hits[0].score < MIN_RERANK_SCORE:
        return False, f"best reranker score {hits[0].score:.1f} is below {MIN_RERANK_SCORE}"
    return True, ""


@dataclass
class Retrieval:
    hits: list[Hit]
    selection: ChapterSelection
    scope: Scope  # the scope that produced `hits`
    fallback: bool = False
    fallback_reason: str = ""
    hyde_passage: str = ""  # display/debug only - never evidence


def retrieve(store: Store, ctx: QueryContext, document_ids: list[str], mode: Mode = "full") -> Retrieval:
    all_scope = Scope(tuple(document_ids))
    sel = ctx.selection
    if not sel.chapter_ids:
        hits = search_scope(store, ctx, all_scope, mode)
        return Retrieval(hits, sel, all_scope, hyde_passage=ctx._hyde or "")

    chapter_scope = Scope(tuple(sel.document_ids), tuple(sel.chapter_ids))  # filter by document AND chapter
    hits = search_scope(store, ctx, chapter_scope, mode)
    ok, reason = _sufficient(hits)
    if ok:
        return Retrieval(hits, sel, chapter_scope, hyde_passage=ctx._hyde or "")
    log.info("Chapter-scoped retrieval insufficient (%s); falling back to all eligible chunks", reason)
    hits = search_scope(store, ctx, all_scope, mode)
    return Retrieval(hits, sel, all_scope, True, reason, ctx._hyde or "")


def describe_scope(ret: Retrieval) -> str:
    sel = ret.selection
    if ret.fallback:
        names = ", ".join(gen.chapter_label(c.get("chapter_number"), c["chapter_name"]) for c in sel.chapters)
        return f"all selected books (the chapter-scoped search of {names} was not sufficient)"
    if ret.scope.chapter_ids:
        return "; ".join(f"{gen.chapter_label(c.get('chapter_number'), c['chapter_name'])} ({c['document_name']})" for c in sel.chapters)
    return "all selected books"


# ----------------------------------------------------------------------------- #
# The LLM's evidence: the top chunks, nothing else
# ----------------------------------------------------------------------------- #
def build_sources(hits: list[Hit], limit: int = CONTEXT_K) -> list[Source]:
    return [Source(f"S{i}", h.chunk, h.score) for i, h in enumerate(hits[:limit], start=1)]

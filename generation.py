"""generation.py - Gemini access, grounded answers, self-critique, citation validation, abstention.

Everything that talks to Gemini lives here (generation, JSON output, embeddings, retries), so the other
modules only call ``generation.generate_text / generate_json / embed_texts``.

Answer flow (``generate_grounded_answer``):
    draft  -> validate citations in Python -> critique with the SAME model -> regenerate at most once
           -> validate again -> answered | partial | abstained
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
log = logging.getLogger(__name__)

GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

_RETRYABLE = {429, 500, 502, 503, 504}


class ConfigError(RuntimeError):
    """Missing / invalid configuration (e.g. no API key)."""


class LLMError(RuntimeError):
    """A Gemini call failed; the message is safe to show to the student."""


# --------------------------------------------------------------------------- #
# Gemini client, retries, text / JSON / embeddings
# --------------------------------------------------------------------------- #
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and put it in your .env file."
            )
        _client = genai.Client(api_key=key)
    return _client


def _explain(code: int | None, what: str, err: Exception) -> str:
    if code in (400,) and "API key" in str(err):
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if code in (401, 403):
        return f"Gemini refused the {what} request (HTTP {code}). Check that GEMINI_API_KEY is valid and the API is enabled."
    if code == 404:
        return f"Gemini model not found during {what}. Check GEMINI_MODEL / EMBED_MODEL in your .env file."
    if code == 429:
        return f"Gemini rate limit or quota reached during {what}. Wait a minute and try again (free-tier quotas are small)."
    if code in _RETRYABLE:
        return f"Gemini is temporarily unavailable ({what}, HTTP {code}). Please try again shortly."
    return f"Gemini request failed during {what} (HTTP {code}): {getattr(err, 'message', err)}"


def _with_retries(fn, what: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code in _RETRYABLE and attempt < attempts:
                wait = 5 * 2 ** (attempt - 1) if code == 429 else 1.5 * 2 ** (attempt - 1)
                log.warning("Gemini %s failed with HTTP %s; retry %d/%d in %.0fs", what, code, attempt, attempts - 1, wait)
                time.sleep(wait)
                continue
            raise LLMError(_explain(code, what, e)) from e
        except httpx.TransportError as e:
            if attempt < attempts:
                time.sleep(1.5 * 2 ** (attempt - 1))
                continue
            raise LLMError(f"Could not reach the Gemini API during {what} ({e.__class__.__name__}). Check your connection.") from e


def generate_text(prompt: str, *, system: str | None = None, temperature: float = 0.2, thinking_budget: int = 0) -> str:
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    resp = _with_retries(lambda: get_client().models.generate_content(model=GEN_MODEL, contents=prompt, config=cfg), "text generation")
    text = (resp.text or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response (it may have been blocked by a safety filter). Try rephrasing.")
    return text


def generate_json(prompt: str, schema: type[BaseModel], *, system: str | None = None, temperature: float = 0.1, thinking_budget: int = 0):
    """Structured output validated against a pydantic ``schema``; one re-ask if the JSON is malformed."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    last: Exception | None = None
    for _ in range(2):
        resp = _with_retries(lambda: get_client().models.generate_content(model=GEN_MODEL, contents=prompt, config=cfg), "structured generation")
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(resp.text or "")
        except (ValidationError, ValueError, TypeError) as e:
            last = e
    raise LLMError(f"Gemini returned malformed structured output ({last.__class__.__name__}). Please try again.")


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 100, progress=None) -> list[list[float]]:
    """Embed with Gemini (L2-normalised, ``EMBED_DIM`` dims). ``progress(done, total)`` is optional."""
    out: list[list[float]] = []
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _with_retries(lambda: get_client().models.embed_content(model=EMBED_MODEL, contents=batch, config=cfg), "embedding")
        vecs = [list(e.values) for e in (resp.embeddings or [])]
        if len(vecs) != len(batch):
            raise LLMError("Gemini returned an unexpected number of embeddings.")
        out.extend(_normalize(v) for v in vecs)
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return out


# --------------------------------------------------------------------------- #
# Page maps: exact page provenance inside a block of text
#   page_map = [[page, char_offset], ...]  (sorted; first offset is 0)
# --------------------------------------------------------------------------- #
def _page_segments(text: str, page_map: list) -> list[tuple[int, str]]:
    if not page_map:
        return []
    bounds = [off for _, off in page_map] + [len(text)]
    return [(page, text[bounds[i] : bounds[i + 1]].strip()) for i, (page, _) in enumerate(page_map)]


def render_with_page_markers(text: str, page_map: list) -> str:
    segs = _page_segments(text, page_map)
    if not segs:
        return text
    return "\n".join(f"⟨page {p}⟩ {t}" for p, t in segs)


def text_for_page(text: str, page_map: list, page: int) -> str:
    return "\n\n".join(t for p, t in _page_segments(text, page_map) if p == page)


def chapter_label(number: int | None, name: str) -> str:
    if number is None or re.match(r"^\s*(chapter|unit|lesson|module|\d)", name or "", re.I):
        return name
    return f"Chapter {number}: {name}"


# --------------------------------------------------------------------------- #
# Data classes shared with retrieval / UI
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One retrieved chunk - the only evidence the model sees."""

    sid: str  # "S1", "S2", ... - the only ids the model may cite
    chunk: dict
    score: float | None = None

    def pages(self) -> list[int]:
        return sorted({p for p, _ in self.chunk.get("page_map", [])} or {self.chunk["page_start"]})

    def page_text(self, page: int) -> str:
        return text_for_page(self.chunk["text"], self.chunk.get("page_map", []), page) or self.chunk["text"]


@dataclass
class Citation:
    number: int  # display number [1], [2] ... in order of first use
    sid: str
    chunk_id: str
    document_id: str
    document_name: str
    chapter_id: str
    chapter_number: int | None
    chapter_name: str
    page: int  # validated: always a page that exists in the cited source
    page_start: int
    page_end: int
    page_exact: bool  # True when the model named a valid page (or the source is a single page)
    excerpt: str  # actual textbook text from the cited page
    corrected: bool = False  # the model named a page that was not in the source; fixed in Python


@dataclass
class GroundedAnswer:
    status: Literal["answered", "partial", "abstained"]
    text: str
    citations: list[Citation] = field(default_factory=list)
    explanation: str = ""
    closest: list[Citation] = field(default_factory=list)  # nearest passages when abstaining
    critique: dict | None = None
    regenerated: bool = False
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Structured-output schemas
# --------------------------------------------------------------------------- #
class CitedPage(BaseModel):
    source_id: str = Field(description="A source id such as S1 that appears in the sources")
    page: int = Field(description="The page number (from the ⟨page N⟩ markers) where the supporting text appears")


class AnswerDraft(BaseModel):
    answer: str = Field(description="Markdown answer; every factual sentence ends with [S#] markers; empty if coverage is none")
    citations: list[CitedPage] = Field(description="For each source cited in the answer, the page where the support appears")
    coverage: Literal["full", "partial", "none"] = Field(description="Do the sources fully, partly, or not at all answer the question?")
    missing: str = Field(description="What the sources do not cover (empty string if coverage is full)")


class Critique(BaseModel):
    verdict: Literal["pass", "revise"]
    unsupported_claims: list[str] = Field(description="Claims in the answer that no cited source supports")
    incorrect_citations: list[str] = Field(description="Citation markers or pages that do not match the text they are attached to")
    missing_from_answer: list[str] = Field(description="Important points that ARE in the sources and answer the question but were left out")
    feedback: str = Field(description="Concise instructions for fixing the answer; empty if verdict is pass")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
ANSWER_SYSTEM = """You are an exam-preparation tutor. You answer a student's question using ONLY the textbook passages
supplied in <source> blocks. Rules:
1. Use only information found in the sources. Never add outside knowledge and never guess.
2. Every factual sentence must end with one or more markers such as [S1] or [S1][S3]. Use only ids that appear in the sources.
3. Page markers like ⟨page 45⟩ show where each page begins. In `citations`, give for every source you cite the single page
   where the supporting text appears. Do NOT write page numbers inside the answer text.
4. If the sources answer only part of the question, answer that part and state in `missing` what is not covered
   (coverage = "partial"). If they do not contain the answer, set coverage = "none" and leave `answer` empty.
5. Write for a student revising for an exam: give the direct answer first, then definitions, key points or steps as short
   bullets when that helps. Be concise and precise."""

CRITIQUE_SYSTEM = """You are a strict reviewer checking a tutoring answer against the textbook sources it was written from.
Check only against the sources shown, not against your own knowledge."""


def render_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        c = s.chunk
        pages = f"{c['page_start']}" if c["page_start"] == c["page_end"] else f"{c['page_start']}-{c['page_end']}"
        head = (
            f'<source id="{s.sid}" textbook="{c["document_name"]}" '
            f'chapter="{chapter_label(c.get("chapter_number"), c["chapter_name"])}" pages="{pages}">'
        )
        body = render_with_page_markers(c["text"], c.get("page_map", []))
        blocks.append(f"{head}\n{body}\n</source>")
    return "\n\n".join(blocks)


def generate_draft(question: str, sources: list[Source], feedback: str | None = None, previous: AnswerDraft | None = None) -> AnswerDraft:
    prompt = f"QUESTION:\n{question}\n\nSOURCES:\n{render_sources(sources)}\n"
    if feedback and previous:
        prompt += (
            f"\nYOUR PREVIOUS ANSWER (rejected by the reviewer):\n{previous.answer}\n\n"
            f"REVIEWER FEEDBACK - fix every point, using only the sources:\n{feedback}\n"
        )
    prompt += "\nWrite the answer as JSON."
    return generate_json(prompt, AnswerDraft, system=ANSWER_SYSTEM, temperature=0.2, thinking_budget=1024)


# --------------------------------------------------------------------------- #
# Citation validation (pure Python, against real source metadata)
# --------------------------------------------------------------------------- #
_MARKER = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_PAGE_MENTION = re.compile(r"\s*\((?:see\s+)?(?:pp?\.|pages?)\s*(\d+)(?:\s*[-–—]\s*(\d+))?\)", re.I)


@dataclass
class Validated:
    text_sids: str  # answer with cleaned [S#] markers (what the critic reviews)
    text: str  # answer renumbered to [1], [2], ...
    citations: list[Citation]
    problems: list[str]
    unknown_ids: bool
    no_inline_markers: bool


def _citation_for(number: int, s: Source, page: int, exact: bool, corrected: bool) -> Citation:
    c = s.chunk
    return Citation(
        number=number, sid=s.sid, chunk_id=c["chunk_id"], document_id=c["document_id"], document_name=c["document_name"],
        chapter_id=c["chapter_id"], chapter_number=c.get("chapter_number"), chapter_name=c["chapter_name"],
        page=page, page_start=c["page_start"], page_end=c["page_end"],
        page_exact=exact, excerpt=s.page_text(page), corrected=corrected,
    )


def validate_citations(draft: AnswerDraft, sources: list[Source]) -> Validated:
    by_sid = {s.sid: s for s in sources}
    problems: list[str] = []
    unknown = False

    def _clean(m: re.Match) -> str:
        nonlocal unknown
        keep = []
        for sid in re.findall(r"S\d+", m.group(1)):
            if sid in by_sid:
                keep.append(sid)
            else:
                unknown = True
                problems.append(f"Unknown source id {sid} was removed from the answer.")
        return "".join(f"[{sid}]" for sid in dict.fromkeys(keep))

    text = _MARKER.sub(_clean, draft.answer or "")
    used = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", text)))

    claimed: dict[str, list[int]] = {}
    for c in draft.citations:
        sid = (c.source_id or "").strip().upper()
        if sid in by_sid:
            claimed.setdefault(sid, []).append(int(c.page))
        elif sid:
            unknown = True
            problems.append(f"The citation list named unknown source id {sid}.")

    no_markers = not used
    if no_markers and claimed:
        used = list(claimed)
        problems.append("The answer text had no inline [S#] markers.")

    # strip page numbers the model wrote in prose unless they exist in the cited sources
    allowed_pages = {p for sid in used for p in by_sid[sid].pages()}

    def _scrub(m: re.Match) -> str:
        nums = [int(n) for n in m.groups() if n]
        if all(n in allowed_pages for n in nums):
            return m.group(0)
        problems.append(f"Removed a page mention ({m.group(0).strip()}) that does not match the cited sources.")
        return ""

    text = _PAGE_MENTION.sub(_scrub, text)

    citations: list[Citation] = []
    for number, sid in enumerate(used, start=1):
        s = by_sid[sid]
        valid_pages = set(s.pages())
        pages = claimed.get(sid, [])
        good = [p for p in pages if p in valid_pages]
        bad = [p for p in pages if p not in valid_pages]
        for p in bad:
            problems.append(f"{sid}: page {p} is not in that source (valid: {s.pages()}); corrected.")
        if good:
            page, exact = good[0], True
        else:
            page = s.chunk["page_start"]
            exact = s.chunk["page_start"] == s.chunk["page_end"]
        citations.append(_citation_for(number, s, page, exact, corrected=bool(bad)))

    numbering = {c.sid: c.number for c in citations}
    display = re.sub(r"\[(S\d+)\]", lambda m: f"[{numbering[m.group(1)]}]" if m.group(1) in numbering else "", text)
    return Validated(text, display.strip(), citations, problems, unknown, no_markers)


# --------------------------------------------------------------------------- #
# Self-critique (same model) and the single regeneration
# --------------------------------------------------------------------------- #
def critique_answer(question: str, sources: list[Source], val: Validated) -> Critique:
    cited = "; ".join(f"{c.sid} -> page {c.page}" for c in val.citations) or "(none)"
    prompt = (
        f"QUESTION:\n{question}\n\nSOURCES:\n{render_sources(sources)}\n\n"
        f"ANSWER UNDER REVIEW (markers like [S1] point to sources):\n{val.text_sids}\n\n"
        f"PAGES THE ANSWER CITES: {cited}\n\n"
        "Review the answer:\n"
        "- unsupported_claims: statements that no cited source supports.\n"
        "- incorrect_citations: a marker or page whose source does not actually say what the sentence claims.\n"
        "- missing_from_answer: important points present in the sources that directly answer the question but were omitted.\n"
        "Set verdict to \"revise\" only if there is at least one unsupported claim, an incorrect citation, or a major omission; "
        "otherwise \"pass\". Put short, concrete fix instructions in `feedback`."
    )
    return generate_json(prompt, Critique, system=CRITIQUE_SYSTEM, temperature=0.0, thinking_budget=1024)


def _needs_revision(critique: Critique | None, val: Validated) -> bool:
    if val.unknown_ids or val.no_inline_markers:
        return True
    return bool(critique and critique.verdict == "revise")


def _feedback_text(critique: Critique | None, val: Validated) -> str:
    lines: list[str] = []
    if critique:
        lines += [f"- Unsupported claim (remove it or support it from a source): {x}" for x in critique.unsupported_claims]
        lines += [f"- Incorrect citation: {x}" for x in critique.incorrect_citations]
        lines += [f"- Present in the sources but missing from the answer: {x}" for x in critique.missing_from_answer]
        if critique.feedback:
            lines.append(critique.feedback)
    lines += [f"- Validation problem: {p}" for p in val.problems]
    return "\n".join(lines) or "Re-check every claim and citation against the sources."


# --------------------------------------------------------------------------- #
# Abstention + orchestration
# --------------------------------------------------------------------------- #
def _abstain(sources: list[Source], reason: str, scope_description: str, critique=None, regenerated=False, problems=None) -> GroundedAnswer:
    closest = [_citation_for(i, s, s.chunk["page_start"], s.chunk["page_start"] == s.chunk["page_end"], False) for i, s in enumerate(sources[:3], 1)]
    where = f" I searched: {scope_description}." if scope_description else ""
    text = (
        "**I can't answer this from your uploaded textbook(s).** "
        f"{reason}{where} Try rephrasing the question, naming the chapter, or selecting a different book."
    )
    return GroundedAnswer("abstained", text, [], explanation=reason, closest=closest, critique=_critique_dict(critique), regenerated=regenerated, problems=problems or [])


def _critique_dict(c: Critique | None) -> dict | None:
    return c.model_dump() if c else None


def generate_grounded_answer(question: str, sources: list[Source], scope_description: str = "") -> GroundedAnswer:
    if not sources:
        return _abstain([], "No passages relevant to this question were found in the selected book(s).", scope_description)

    draft = generate_draft(question, sources)
    val = validate_citations(draft, sources)
    critique: Critique | None = None
    regenerated = False

    if draft.coverage != "none":
        try:
            critique = critique_answer(question, sources, val)
        except LLMError as e:  # the critique is a safety net; do not lose the answer if it fails
            log.warning("Self-critique unavailable: %s", e)
        if _needs_revision(critique, val):
            feedback = _feedback_text(critique, val)
            draft = generate_draft(question, sources, feedback=feedback, previous=draft)  # regenerate once only
            val = validate_citations(draft, sources)
            regenerated = True

    if draft.coverage == "none" or not (draft.answer or "").strip():
        reason = draft.missing.strip() or "The retrieved passages do not contain the information needed."
        return _abstain(sources, reason, scope_description, critique, regenerated, val.problems)
    if not val.citations:
        return _abstain(
            sources, "I could not produce an answer that I could verify against specific textbook pages.",
            scope_description, critique, regenerated, val.problems,
        )

    status = "answered" if draft.coverage == "full" else "partial"
    explanation = "" if status == "answered" else (draft.missing.strip() or "The textbook only covers part of this question.")
    return GroundedAnswer(status, val.text, val.citations, explanation, [], _critique_dict(critique), regenerated, val.problems)

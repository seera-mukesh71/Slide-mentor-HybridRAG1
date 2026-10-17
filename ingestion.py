"""ingestion.py - PDF -> chapters (from the table of contents) -> page-tagged chunks -> embeddings -> MongoDB.

A chapter is only a label. Every chunk carries its chapter_id / chapter_name, and retrieval filters on them before BM25
and vector search. Chunks are cut inside one chapter (never across two) and keep a ``page_map``
([[page, char_offset], ...]) so a citation can name the exact page.

Where the chapters come from, in order: the PDF's bookmarks, the printed contents page (printed page numbers are matched
to PDF pages), or - when the book has no table of contents - the whole book as one chapter.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import generation as gen
from storage import Store

log = logging.getLogger(__name__)

MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "100"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "1500"))
CHUNK_WORDS = 200  # a chunk is closed once it reaches this size
CHUNK_MAX_WORDS = 300  # one paragraph longer than this is split at sentence ends


class IngestionError(ValueError):
    """The uploaded file cannot be ingested; the message is safe to show to the student."""


CHAPTERISH = re.compile(r"^\s*((?i:chapter|unit|lesson|module)\s+\S+|\d{1,2}[.:]?\s+\S)")
CHAPTER_LABEL_ONLY = re.compile(r"^\s*(?i:chapter|unit|lesson|module|part)\s+\d{1,3}\s*[:.]?\s*$")
NUMBERED = re.compile(r"^\s*(?:(?i:chapter|unit|lesson|module)\s+)?(\d{1,3})\b[\s.:)\-–—]*(.*)$")
NUM_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+\S")
BACK_MATTER = re.compile(
    r"^\s*(preface|foreword|contents|table of contents|list of (figures|tables)|index|subject index|glossary|bibliography|"
    r"references|acknowledg\w*|about the authors?|copyright|title page)\s*$", re.I)
PAGE_NUM_ONLY = re.compile(r"^(page\s+)?(\d{1,4}|[ivxlcdm]{1,7})$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", s).lower())).strip()


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in (x.strip() for x in lines):
        if not ln:
            continue
        if out.endswith("-") and ln[:1].islower() and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + ln  # "thermo-\ndynamics" -> "thermodynamics"
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------- #
# 1. Validation
# --------------------------------------------------------------------------- #
def validate_pdf(data: bytes, filename: str) -> pymupdf.Document:
    if not filename.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are supported.")
    if not data:
        raise IngestionError("The uploaded file is empty.")
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise IngestionError(f"This PDF is larger than the {MAX_PDF_MB} MB limit.")
    if b"%PDF-" not in data[:1024]:
        raise IngestionError("This file does not look like a PDF (missing the %PDF header).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # PyMuPDF raises several exception types for corrupt files
        raise IngestionError(f"The PDF could not be opened; it may be corrupted ({e.__class__.__name__}).") from e
    if doc.needs_pass:
        raise IngestionError("This PDF is password-protected. Remove the password and upload it again.")
    if doc.page_count < 1:
        raise IngestionError("The PDF has no pages.")
    if doc.page_count > MAX_PDF_PAGES:
        raise IngestionError(f"This PDF has {doc.page_count} pages; the limit is {MAX_PDF_PAGES}.")
    return doc


# --------------------------------------------------------------------------- #
# 2. Text blocks in reading order, tagged with their page, without running headers / footers
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    page: int
    text: str
    y0: float
    y1: float
    page_h: float
    n_lines: int


def _in_margin(b: Block) -> bool:
    return b.y1 < 0.06 * b.page_h or b.y0 > 0.94 * b.page_h


def extract_blocks(doc: pymupdf.Document) -> list[Block]:
    raw: list[Block] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            # MuPDF can put a heading and the paragraph under it in one block: split at changes of font size / boldness,
            # so a chapter heading in the middle of a page becomes a block of its own
            runs: list[dict] = []
            for ln in b.get("lines", []):
                chars = sum(len(s["text"].strip()) for s in ln["spans"])
                if not chars:
                    continue
                size = sum(s["size"] * len(s["text"].strip()) for s in ln["spans"]) / chars
                bold = sum(len(s["text"].strip()) for s in ln["spans"] if (s["flags"] & 16) or "bold" in s["font"].lower()) >= 0.6 * chars
                text = "".join(s["text"] for s in ln["spans"])
                if runs and abs(runs[-1]["size"] - size) <= 0.8 and runs[-1]["bold"] == bold:
                    runs[-1]["lines"].append(text)
                    runs[-1]["y1"] = ln["bbox"][3]
                else:
                    runs.append({"lines": [text], "size": size, "bold": bold, "y0": ln["bbox"][1], "y1": ln["bbox"][3]})
            for r in runs:
                text = unicodedata.normalize("NFKC", _join_lines(r["lines"]))
                if len(re.sub(r"\W", "", text)) >= 2:
                    raw.append(Block(pno + 1, text, r["y0"], r["y1"], page.rect.height, len(r["lines"])))

    # running headers / footers: page numbers, and margin text that repeats (digits ignored) on many pages
    def key(b: Block) -> str:
        return re.sub(r"\d+", "#", b.text.lower())

    pages_with: Counter[str] = Counter()
    for k, _ in {(key(b), b.page) for b in raw if _in_margin(b)}:
        pages_with[k] += 1
    threshold = max(3, int(0.25 * doc.page_count))
    return [b for b in raw if not (_in_margin(b) and (PAGE_NUM_ONLY.match(b.text.strip()) or pages_with[key(b)] >= threshold))]


# --------------------------------------------------------------------------- #
# 3. Table of contents -> [(level, title, PDF page)]
# --------------------------------------------------------------------------- #
def _page_labels(doc: pymupdf.Document) -> dict[str, str]:
    """Printed page labels that differ from the PDF page number ({'15': 'xiii'})."""
    try:
        return {str(i + 1): doc[i].get_label() for i in range(doc.page_count) if doc[i].get_label() not in ("", str(i + 1))}
    except Exception:  # labels are a nicety; never fail ingestion because of them
        return {}


# ---- printed contents page: rows like "Chapter 2  Bonding ........ 31" ---------------- #
_LEADER = re.compile(r"^[.·…_\-–—]+$")
_PAGE_TOKEN = re.compile(r"^[.·…_]*(\d{1,4})$")


def _toc_rows(page: pymupdf.Page) -> list[list[tuple]]:
    """Words of a page grouped into visual rows, left to right."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    cy = 0.0
    for w in words:
        yc = (w[1] + w[3]) / 2
        if not rows or abs(yc - cy) > 0.6 * max(w[3] - w[1], 1):
            rows.append([])
            cy = yc
        rows[-1].append(w)
    return [sorted(r, key=lambda w: w[0]) for r in rows]


def _parse_toc_row(row: list[tuple]) -> tuple[str, int] | None:
    """'Chapter 2  Bonding ...... 31' -> ('Chapter 2 Bonding', 31). 'Chapter 2' alone is a heading, not an entry."""
    if len(row) < 2 or not (m := _PAGE_TOKEN.match(row[-1][4])):
        return None
    last, prev = row[-1], row[-2]
    leader = last[4][0] in ".·…_" or bool(_LEADER.match(prev[4])) or prev[4].endswith("..")
    if not leader and last[0] - prev[2] < 12:  # without leader dots the page number must be set apart from the title
        return None
    title = re.sub(r"[.·…_\s]{2,}$", "", " ".join(w[4] for w in row[:-1] if not _LEADER.match(w[4]))).strip(" .·…_")
    return (title, int(m.group(1))) if re.search(r"[A-Za-z]", title) else None


def _toc_level(title: str) -> int:
    m = NUM_HEADING.match(title)  # 1.2 -> 2, 1.2.3 -> 3; "Chapter 4", "4 Title" and unnumbered titles -> 1
    return min(m.group(1).count(".") + 1, 3) if m else 1


def _printed_toc(doc: pymupdf.Document) -> tuple[list[tuple[int, str, int]], int]:
    """Entries [(level, title, PRINTED page)] from the contents page(s) near the front, and the last contents page."""
    rows_by_page = {}
    for i in range(min(doc.page_count, 40)):
        parsed = [r for r in map(_parse_toc_row, _toc_rows(doc[i])) if r]
        if len(parsed) >= 4:
            rows_by_page[i] = parsed
    if not rows_by_page:
        return [], 0
    first = last = min(rows_by_page)
    while last + 1 in rows_by_page:
        last += 1
    rows = [r for i in range(first, last + 1) for r in rows_by_page[i]]
    pages = [p for _, p in rows]
    if len(rows) < 3 or sum(b >= a for a, b in zip(pages, pages[1:])) < 0.8 * (len(rows) - 1):  # real lists ascend
        return [], 0
    return [(_toc_level(t), t, p) for t, p in rows], last + 1


class TocEntry(BaseModel):
    level: int = Field(description="1 = chapter, 2 = section, 3 = subsection")
    title: str = Field(description="The entry title without its page number or dot leaders")
    printed_page: int = Field(description="The page number printed to the right of the entry")


class TocList(BaseModel):
    entries: list[TocEntry]


def _toc_with_gemini(doc: pymupdf.Document) -> tuple[list[tuple[int, str, int]], int]:
    """Fallback for contents pages whose layout the row parser cannot read."""
    start = next((i for i in range(min(doc.page_count, 30)) if re.search(r"\bcontents\b", doc[i].get_text()[:500], re.I)), None)
    if start is None:
        return [], 0
    end = min(start + 4, doc.page_count)
    text = "\n\n".join(f"--- page {i + 1} ---\n{doc[i].get_text()}" for i in range(start, end))
    prompt = (
        "Below is the text of a textbook's table of contents. List every chapter and section entry in order with the page "
        "number printed next to it (level 1 = chapter, 2 = section, 3 = subsection). Ignore roman-numeral pages and anything "
        "that is not a contents entry.\n\n" + text
    )
    try:
        out = gen.generate_json(prompt, TocList, temperature=0.0)
    except (gen.LLMError, gen.ConfigError) as e:
        log.warning("Gemini could not read the contents page: %s", e)
        return [], 0
    return [(min(max(e.level, 1), 3), e.title.strip(), e.printed_page) for e in out.entries if e.title.strip() and e.printed_page > 0], end


def _matches(title: str, text: str) -> bool:
    nt, nb = _norm(title), _norm(text)
    return bool(nb) and (nb == nt or fuzz.ratio(nb, nt) >= 85 or (len(nb) >= 4 and nb in nt and len(nb) >= 0.5 * len(nt)))


def _to_pdf_pages(doc: pymupdf.Document, blocks: list[Block], rows: list[tuple[int, str, int]], toc_end: int):
    """Printed page numbers -> PDF pages: ([(level, title, PDF page)], page_labels), or None if they cannot be matched."""
    n = doc.page_count
    by_label = {}
    for i in range(n):
        if label := doc[i].get_label():
            by_label.setdefault(label, i + 1)
    if sum(str(p) in by_label for _, _, p in rows) >= 0.7 * len(rows):  # the PDF defines its own page labels
        return [(lvl, t, by_label[str(p)]) for lvl, t, p in rows if str(p) in by_label], _page_labels(doc)

    # otherwise look where chapter titles really are and vote for the offset (PDF page - printed page)
    titles_on: dict[int, list[str]] = {}
    for b in blocks:
        if not _in_margin(b) and len(b.text.split()) <= 25:
            titles_on.setdefault(b.page, []).append(b.text)
    top = [r for r in rows if r[0] == min(lvl for lvl, _, _ in rows)]
    sample = top[:: max(1, len(top) // 8)][:8]
    votes: Counter[int] = Counter()
    for _, title, printed in sample:
        for p in range(max(toc_end + 1, printed), min(n, printed + 120) + 1):
            if any(_matches(title, t) for t in titles_on.get(p, [])):
                votes[p - printed] += 1
                break
    if not votes or votes.most_common(1)[0][1] < max(1, len(sample) // 2):
        return None
    offset = votes.most_common(1)[0][0]
    labels = _page_labels(doc) or ({str(p): str(p - offset) for p in range(offset + 1, n + 1)} if offset else {})
    return [(lvl, t, p + offset) for lvl, t, p in rows if 1 <= p + offset <= n], labels


def read_toc(doc: pymupdf.Document, blocks: list[Block]) -> tuple[list[tuple[int, str, int]], str, dict[str, str]]:
    """Returns (entries [(level, title, PDF page)], source 'toc' | 'printed_toc' | 'none', page_labels)."""
    bookmarks = [(lvl, t.strip(), p) for lvl, t, p in doc.get_toc(simple=True) if t.strip() and 1 <= p <= doc.page_count]
    if len(bookmarks) >= 2:
        return bookmarks, "toc", _page_labels(doc)
    rows, toc_end = _printed_toc(doc)
    if len(rows) < 3:
        rows, toc_end = _toc_with_gemini(doc)
    if len(rows) >= 3 and (mapped := _to_pdf_pages(doc, blocks, rows, toc_end)) and len(mapped[0]) >= 2:
        return mapped[0], "printed_toc", mapped[1]
    return [], "none", _page_labels(doc)


# --------------------------------------------------------------------------- #
# 4. Chapters: each starts at its heading block, so a chapter that begins mid-page only takes the text below its heading
# --------------------------------------------------------------------------- #
@dataclass
class Chapter:
    name: str
    start: int  # index of the first block that belongs to this chapter
    skip: set[int] = field(default_factory=set)  # heading blocks, which are not body text
    number: int | None = None
    title: str = ""
    blocks: list[Block] = field(default_factory=list)


def _chapter_level(toc: list[tuple[int, str, int]]) -> int:
    levels = sorted({lvl for lvl, _, _ in toc})
    for lvl in levels[:3]:  # shallowest level whose titles look like "Chapter 3" / "3 Title"
        titles = [t for l, t, _ in toc if l == lvl]
        if sum(bool(CHAPTERISH.match(t)) for t in titles) >= 0.5 * len(titles):
            return lvl
    return next((lvl for lvl in levels if sum(l == lvl for l, _, _ in toc) >= 2), levels[0])


def _locate(title: str, page: int, blocks: list[Block], by_page: dict[int, list[int]]) -> tuple[int | None, set[int]]:
    """Block where a chapter starts: its heading on the start page (plus a separate 'Chapter 3' label block above it)."""
    for i in by_page.get(page, []):
        b = blocks[i]
        if len(b.text.split()) <= 25 and b.n_lines <= 3 and not _in_margin(b) and _matches(title, b.text):
            if i > 0 and blocks[i - 1].page == b.page and CHAPTER_LABEL_ONLY.match(blocks[i - 1].text):
                return i - 1, {i - 1, i}
            return i, {i}
    # heading text not found on that page: the chapter starts at the top of the page
    return next((by_page[p][0] for p in range(page, page + 4) if by_page.get(p)), None), set()


def build_chapters(toc: list[tuple[int, str, int]], blocks: list[Block]) -> list[Chapter]:
    if len(toc) < 2:
        return []
    level = _chapter_level(toc)
    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b.page, []).append(i)
    chapters: list[Chapter] = []
    last = 0
    for lvl, title, page in toc:
        if lvl != level:
            continue  # sections are not used: the chapter is the only structure we keep
        idx, skip = _locate(title, page, blocks, by_page)
        if idx is not None:
            chapters.append(Chapter(title, max(idx, last), skip))
            last = chapters[-1].start
    return chapters


def number_chapters(chapters: list[Chapter]) -> str:
    """Use the numbers printed in the titles when most chapters have one, otherwise count 1, 2, 3 in order."""
    parsed = [NUMBERED.match(c.name) for c in chapters]
    if sum(m is not None for m in parsed) >= 0.5 * len(chapters):
        for c, m in zip(chapters, parsed):
            c.number, c.title = (int(m.group(1)), m.group(2).strip() or c.name) if m else (None, c.name)
        return "title"
    for n, c in enumerate(chapters, start=1):
        c.number, c.title = n, c.name
    return "ordinal"


# --------------------------------------------------------------------------- #
# 5. Chunks with exact page maps
# --------------------------------------------------------------------------- #
def _split_long(text: str) -> list[str]:
    """A paragraph over CHUNK_MAX_WORDS is split at sentence ends (or by word count if it has no punctuation)."""
    if len(text.split()) <= CHUNK_MAX_WORDS:
        return [text]
    pieces, cur, count = [], [], 0
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text):
        words = sent.split()
        if len(words) > CHUNK_MAX_WORDS:
            pieces += [" ".join(words[j : j + CHUNK_WORDS]) for j in range(0, len(words), CHUNK_WORDS)]
            continue
        if cur and count + len(words) > CHUNK_WORDS:
            pieces.append(" ".join(cur))
            cur, count = [], 0
        cur.append(sent)
        count += len(words)
    return pieces + ([" ".join(cur)] if cur else [])


def _join_pieces(pieces: list[tuple[int, str]]) -> tuple[str, list[list[int]]]:
    """[(page, text), ...] -> (text, page_map). page_map [[4, 0], [5, 310]] = text from offset 310 is on page 5."""
    parts, page_map, n = [], [], 0
    for page, t in pieces:
        if parts:
            n += 2  # the "\n\n" between pieces
        if not page_map or page_map[-1][0] != page:
            page_map.append([page, n])
        parts.append(t)
        n += len(t)
    return "\n\n".join(parts), page_map


def make_chunks(blocks: list[Block]) -> list[tuple[str, list[list[int]]]]:
    chunks, cur, words = [], [], 0
    for b in blocks:
        for piece in _split_long(b.text):
            w = len(piece.split())
            if cur and words + w > CHUNK_WORDS:
                chunks.append(_join_pieces(cur))
                cur, words = [], 0
            cur.append((b.page, piece))
            words += w
    if cur:
        chunks.append(_join_pieces(cur))
    return [c for c in chunks if len(c[0].split()) >= 5]


def embedding_input(c: dict) -> str:
    return f"{c['document_name']} | {c['chapter_name']}\n{c['text']}"


# --------------------------------------------------------------------------- #
# 6. Whole ingestion
# --------------------------------------------------------------------------- #
def _clean_name(filename: str) -> str:
    return (re.sub(r"[_\-]+", " ", Path(filename).stem).strip() or "Untitled textbook")[:80]


def parse_pdf(data: bytes, filename: str) -> dict:
    """Everything except embedding and storage: returns the document record and its chunks (tests use this directly)."""
    doc = validate_pdf(data, filename)
    document_id = hashlib.sha1(data).hexdigest()[:12]
    name = _clean_name(filename)
    blocks = extract_blocks(doc)
    if sum(len(b.text.split()) for b in blocks) < 15 * doc.page_count:
        raise IngestionError("This PDF has little or no extractable text (it looks like a scan). OCR is not supported - upload a text-based PDF.")

    toc, source, page_labels = read_toc(doc, blocks)
    chapters = build_chapters(toc, blocks)
    if not chapters:
        chapters, source = [Chapter(name, 0)], "none"  # no usable table of contents: the whole book is one chapter
    for k, ch in enumerate(chapters):
        end = chapters[k + 1].start if k + 1 < len(chapters) else len(blocks)
        ch.blocks = [b for i, b in enumerate(blocks[ch.start : end], ch.start) if i not in ch.skip]
    chapters = [c for c in chapters if c.blocks and not BACK_MATTER.match(c.name)]
    if not chapters:
        raise IngestionError("No readable text could be extracted from this PDF.")
    number_source = number_chapters(chapters)

    outline, chunks = [], []
    for ci, ch in enumerate(chapters):
        chapter_id = f"{document_id}-c{ci:02d}"
        made = make_chunks(ch.blocks)
        for k, (text, page_map) in enumerate(made):
            chunks.append({
                "chunk_id": f"{chapter_id}-k{k:04d}", "document_id": document_id, "document_name": name,
                "chapter_id": chapter_id, "chapter_number": ch.number, "chapter_name": ch.name,
                "text": text, "page_start": page_map[0][0], "page_end": page_map[-1][0], "page_map": page_map,
                "order": len(chunks), "word_count": len(text.split()),
            })
        pages = [p for _, pm in made for p, _ in pm] or [ch.blocks[0].page]
        outline.append({"chapter_id": chapter_id, "number": ch.number, "name": ch.name, "title": ch.title,
                        "page_start": min(pages), "page_end": max(pages), "num_chunks": len(made)})

    warnings = []
    if source == "none":
        warnings.append("No table of contents was found, so the whole book is treated as one chapter and chapter filtering is not available for it.")
    elif source == "printed_toc":
        warnings.append("This PDF has no bookmarks; chapters were read from the printed contents page. Check them in the Chapters tab.")
    if number_source == "ordinal" and source != "none":
        warnings.append("Chapter numbers are not printed in the titles; chapters are numbered in order of appearance.")

    record = {
        "document_id": document_id, "name": name, "filename": filename, "num_pages": doc.page_count,
        "num_chunks": len(chunks), "num_chapters": len(outline), "structure_source": source,
        "chapter_number_source": number_source, "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(data), "page_labels": page_labels, "chapters": outline, "warnings": warnings,
    }
    return {"record": record, "chunks": chunks}


def ingest_pdf(store: Store, data: bytes, filename: str, progress=None) -> dict:
    """Parse, embed and index one textbook. ``progress(message, fraction)`` is optional. Returns the document record."""
    progress = progress or (lambda msg, frac: None)
    progress("Validating and reading the PDF", 0.02)
    parsed = parse_pdf(data, filename)
    record, chunks = parsed["record"], parsed["chunks"]
    if existing := store.get_document(record["document_id"]):
        raise IngestionError(f"This PDF is already in your library as “{existing['name']}”.")
    progress(f"Found {record['num_chapters']} chapters and {len(chunks)} passages", 0.15)

    vectors = gen.embed_texts(
        [embedding_input(c) for c in chunks], task_type="RETRIEVAL_DOCUMENT",
        progress=lambda done, total: progress(f"Embedding passages ({done}/{total})", 0.15 + 0.75 * done / total),
    )
    for c, v in zip(chunks, vectors):
        c["embedding"] = v

    progress("Saving to MongoDB", 0.93)
    try:
        store.save_pdf(record["document_id"], data)
        store.insert_chunks(chunks)
        store.save_document(record)  # written last: a document record means ingestion completed
    except Exception:
        store.delete_document(record["document_id"])  # no half-ingested books
        raise
    progress("Done", 1.0)
    return record

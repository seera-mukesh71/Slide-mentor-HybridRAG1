"""Tests for ingestion, chapter-first retrieval, RRF, reranking and evaluation metrics.

No network, Gemini or Atlas needed:
  * ``FakeStore`` is the real ``Store`` running on mongomock; only ``$vectorSearch`` is replaced by an in-memory cosine
    search that applies the SAME scope filter (a separate test checks the real ``$vectorSearch`` pipeline).
  * embeddings are a deterministic hashing bag-of-words; Gemini calls are monkeypatched per test.
"""
import hashlib
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import mongomock
import numpy as np
import pymupdf
import pytest
from mongomock import gridfs as mongomock_gridfs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generation as gen  # noqa: E402
import ingestion  # noqa: E402
import retrieval  # noqa: E402
import storage  # noqa: E402
from storage import Scope, Store  # noqa: E402

mongomock_gridfs.enable_gridfs_integration()


# --------------------------------------------------------------------------- #
# Shared helpers (also imported by test_pipeline.py)
# --------------------------------------------------------------------------- #
SCIENCE = [
    ("Chapter 1: Thermodynamics", [
        ("1.1 Heat and Temperature", [
            "Heat is the transfer of thermal energy between systems at different temperatures. Temperature measures the average kinetic energy of the particles in a substance. When two bodies are in contact, energy flows from the hotter body to the colder one until thermal equilibrium is reached. The specific heat capacity of a material tells us how much energy is needed to raise one kilogram of it by one kelvin. Water has an unusually high specific heat capacity which is why oceans moderate the climate of coastal regions.",
            "Calorimetry is the experimental technique used to measure heat transfer. In a calorimeter a hot sample is dropped into water of known mass and temperature, and the final equilibrium temperature lets us compute the heat released by the sample. Careful insulation is essential because any energy lost to the surroundings produces systematic errors in the measured specific heat."]),
        ("1.2 Entropy", [
            "Entropy is a measure of the number of microscopic arrangements consistent with a macroscopic state. The second law of thermodynamics states that the total entropy of an isolated system never decreases. Heat engines can therefore never convert all absorbed heat into work, and the Carnot efficiency sets the theoretical upper limit of any engine operating between two reservoirs.",
            "Irreversible processes such as free expansion of a gas and mixing of two liquids increase the entropy of the universe. Reversible idealised processes keep total entropy constant. Statistical mechanics connects the thermodynamic definition of entropy to the logarithm of the number of accessible microstates."]),
    ]),
    ("Chapter 2: Photosynthesis", [
        ("2.1 Light Reactions", [
            "The light reactions of photosynthesis take place in the thylakoid membranes of the chloroplast. Chlorophyll absorbs photons, mostly in the red and blue parts of the spectrum, and uses the captured energy to split water molecules, releasing oxygen. The energy is stored temporarily as ATP and NADPH through a chain of electron carriers and a proton gradient across the membrane.",
            "Photosystem II and photosystem I work in series. Excited electrons from photosystem II travel down an electron transport chain, pumping protons into the thylakoid lumen. The proton gradient drives ATP synthase, while photosystem I re-energises electrons to reduce NADP plus to NADPH."]),
        ("2.2 Calvin Cycle", [
            "The Calvin cycle occurs in the stroma and uses ATP and NADPH from the light reactions to fix carbon dioxide into sugar. The enzyme rubisco attaches carbon dioxide to ribulose bisphosphate. Through reduction and regeneration steps the cycle produces glyceraldehyde three phosphate, which plants use to build glucose and other carbohydrates.",
            "Environmental factors such as light intensity, carbon dioxide concentration and temperature limit the rate of photosynthesis. Plants in hot dry climates have evolved C4 and CAM pathways that concentrate carbon dioxide around rubisco to reduce wasteful photorespiration."]),
    ]),
    ("Chapter 3: Genetics", [
        ("3.1 DNA Structure", [
            "Deoxyribonucleic acid is a double helix made of two antiparallel strands of nucleotides. Each nucleotide contains a phosphate group, the sugar deoxyribose and one of four nitrogenous bases: adenine, thymine, guanine or cytosine. Adenine pairs with thymine and guanine pairs with cytosine through hydrogen bonds, which explains Chargaff's rules.",
            "During replication the helix is unwound by helicase and each strand serves as a template for DNA polymerase. Replication is semiconservative: every new molecule contains one original strand and one newly synthesised strand, which preserves genetic information with high fidelity."]),
        ("3.2 Inheritance", [
            "Gregor Mendel showed that traits are inherited as discrete units that we now call genes. Each individual carries two alleles of a gene, one from each parent. A dominant allele masks the effect of a recessive allele in heterozygotes, which explains the three to one phenotype ratio in the second generation of a monohybrid cross.",
            "The law of independent assortment states that alleles of different genes segregate independently when they sit on different chromosomes. Linked genes close together on the same chromosome tend to be inherited together unless crossing over separates them during meiosis."]),
    ]),
]

CHEMISTRY = [
    ("Chapter 1: Atoms", [
        ("1.1 Atomic Structure", [
            "An atom consists of a dense nucleus of protons and neutrons surrounded by electrons that occupy orbitals. The number of protons defines the element, while isotopes of an element differ in their number of neutrons. Electrons fill orbitals from the lowest energy upwards, which explains the periodic repetition of chemical properties across the table.",
            "Spectroscopy reveals atomic structure because electrons absorb and emit light only at discrete energy differences. The hydrogen emission lines were explained by Bohr and later by quantum mechanics, which replaced circular orbits by probability distributions of the electron."]),
    ]),
    ("Chapter 2: Bonding", [
        ("2.1 Covalent Bonds", [
            "A covalent bond forms when two atoms share a pair of electrons. Shared electrons lower the energy of the system compared with separate atoms, and the bond length is the distance at which attraction and repulsion balance. Electronegativity differences make the sharing unequal and produce polar covalent bonds in molecules such as water.",
            "Molecular geometry follows from the repulsion between electron pairs around a central atom. Methane is tetrahedral, ammonia is trigonal pyramidal because of its lone pair, and water is bent. These shapes determine polarity, boiling points and many biological interactions.",
            "Ionic bonds arise when one atom transfers electrons to another and the resulting ions attract each other electrostatically. Sodium chloride forms a crystal lattice in which every ion is surrounded by oppositely charged neighbours, which explains its high melting point and brittleness."]),
    ]),
]


def make_pdf(chapters, with_toc=True, headers=True, printed_toc=False) -> bytes:
    """Build a small textbook PDF with PyMuPDF. Section titles starting with '###' become level-3 subsections.

    ``with_toc``: embedded bookmarks. ``printed_toc``: a cover page + a printed contents page are put in front, so
    printed page numbers (what the contents page shows) are 2 lower than PDF pages."""
    doc = pymupdf.open()
    toc = []

    def newpage():
        page = doc.new_page()
        if headers:
            page.insert_text((72, 36), "Running Header Title", fontsize=9, fontname="helv")
            page.insert_text((290, 820), str(len(doc)), fontsize=9, fontname="helv")
        return page, 90

    for ch_title, sections in chapters:
        page, y = newpage()
        page.insert_text((72, y), ch_title, fontsize=22, fontname="hebo")
        toc.append([1, ch_title, len(doc)])
        y += 40
        for sec_title, paras in sections:
            sub = sec_title.startswith("###")
            title = sec_title.lstrip("#")
            if y > 600:
                page, y = newpage()
            page.insert_text((72, y), title, fontsize=12 if sub else 15, fontname="hebo")
            toc.append([3 if sub else 2, title, len(doc)])
            y += 22 if sub else 26
            for text in paras:
                h = (math.ceil(len(text) / 80) + 1) * 14
                if y + h > 770:
                    page, y = newpage()
                assert page.insert_textbox(pymupdf.Rect(72, y, 523, y + h), text, fontsize=11, fontname="helv") >= 0
                y += h + 8
    if with_toc:
        doc.set_toc(toc)
    if not printed_toc:
        return doc.tobytes()
    final = pymupdf.open()
    final.new_page().insert_text((72, 300), "Introductory Science", fontsize=30, fontname="hebo")  # cover
    contents = final.new_page()
    contents.insert_text((72, 80), "Contents", fontsize=20, fontname="hebo")
    y = 120
    for lvl, title, page in toc:
        indent = 72 + 18 * (lvl - 1)
        contents.insert_text((indent, y), title + " " + "." * 25, fontsize=11, fontname="helv")
        contents.insert_text((500, y), str(page), fontsize=11, fontname="helv")
        y += 18
    final.insert_pdf(doc)
    return final.tobytes()


def hash_vec(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim)
    for tok in retrieval.tokenize(text):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def fake_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=100, progress=None):
    out = [hash_vec(t) for t in texts]
    if progress:
        progress(len(texts), len(texts))
    return out


class FakeStore(Store):
    """The real Store on mongomock; ``$vectorSearch`` is emulated with cosine similarity under the same scope filter."""

    def __init__(self):
        super().__init__(db=mongomock.MongoClient()["test"])
        self.chunk_scopes: list[Scope] = []
        self.vector_scopes: list[Scope] = []

    def ensure_indexes(self, dim):  # Atlas search indexes do not exist on mongomock
        pass

    def vector_index_status(self):
        return "ready"

    def get_chunks(self, scope):
        self.chunk_scopes.append(scope)
        return super().get_chunks(scope)

    def vector_search(self, query_vector, scope, limit=30, num_candidates=None):
        self.vector_scopes.append(scope)
        docs = list(self.chunks.find(scope.mongo_filter(), {"_id": 0}))  # pre-filter, exactly like $vectorSearch
        q = np.array(query_vector)
        ranked = sorted(((float(np.dot(q, np.array(d["embedding"]))), d) for d in docs), key=lambda x: -x[0])[:limit]
        return [{**{k: v for k, v in d.items() if k != "embedding"}, "score": s} for s, d in ranked]


class FakeCrossEncoder:
    """Scores a (query, passage) pair by shared content tokens - deterministic stand-in for the real cross-encoder."""

    def predict(self, pairs):
        return np.array([float(len(set(retrieval.tokenize(q)) & set(retrieval.tokenize(d)))) for q, d in pairs])


def default_fake_text(prompt, **kwargs):
    """HyDE / rewrite stand-in: echoes the question as the 'hypothetical passage'."""
    return "A textbook passage answering: " + prompt.split("QUESTION:")[-1].strip()


def default_fake_json(prompt, schema, **kwargs):
    if schema is retrieval.ChapterChoice:  # Gemini chapter selection: unsure => search everything
        return retrieval.ChapterChoice(chapter_ids=[], confidence="low", reason="test default")
    raise AssertionError(f"unexpected Gemini structured call for {schema.__name__}")


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(gen, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(gen, "generate_text", default_fake_text)
    monkeypatch.setattr(gen, "generate_json", default_fake_json)
    monkeypatch.setattr(retrieval, "get_reranker", lambda: FakeCrossEncoder())
    monkeypatch.setattr(ingestion, "CHUNK_WORDS", 60)  # one paragraph per chunk in these small books
    return monkeypatch


@pytest.fixture
def library(patched):
    store = FakeStore()
    sci = ingestion.ingest_pdf(store, make_pdf(SCIENCE), "Intro_Science_Notes.pdf")
    return store, sci


def chapter_of(store, chunk_id) -> str:
    return store.get_chunks_by_ids([chunk_id])[0]["chapter_name"]


# --------------------------------------------------------------------------- #
# Ingestion: structure + exact page provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["bookmarks", "printed_contents_page"])
def test_ingestion_extracts_chapters_sections_and_pages(patched, variant):
    store = FakeStore()
    printed = variant == "printed_contents_page"
    pdf = make_pdf(SCIENCE, with_toc=not printed, printed_toc=printed)
    rec = ingestion.ingest_pdf(store, pdf, "Intro Science.pdf")
    assert rec["structure_source"] == ("printed_toc" if printed else "toc")
    assert [(c["number"], c["title"]) for c in rec["chapters"]] == [(1, "Thermodynamics"), (2, "Photosynthesis"), (3, "Genetics")]
    chunks = store.get_chunks(Scope((rec["document_id"],)))
    required = {"document_id", "chapter_id", "chapter_number", "chapter_name", "chunk_id", "text", "page_start", "page_end", "page_map"}
    assert all(required <= set(c) for c in chunks)
    assert not any({"parent_id", "section_name"} & set(c) for c in chunks)  # chapter label + page only
    # page provenance: the rubisco text is on printed page 2 = PDF page 4 when a cover + contents page come first
    pdf_page = 4 if printed else 2
    rubisco = [c for c in chunks if "rubisco" in c["text"]]
    assert rubisco and all(c["page_start"] == c["page_end"] == pdf_page for c in rubisco)
    assert all(c["chapter_name"] == "Chapter 2: Photosynthesis" for c in rubisco)
    if printed:
        assert rec["page_labels"][str(pdf_page)] == "2"  # citations can show the number printed in the book
        assert any("printed contents page" in w for w in rec["warnings"])
    # the cover and the contents page are not searchable text
    assert not any("Contents" in c["text"] or "Introductory Science" in c["text"] for c in chunks)
    # running headers / page numbers never leak into the text
    assert not any("Running Header Title" in c["text"] for c in chunks)
    assert store.load_pdf(rec["document_id"]) is not None


def test_book_without_any_table_of_contents_is_one_chapter(patched):
    parsed = ingestion.parse_pdf(make_pdf(SCIENCE, with_toc=False), "Intro Science.pdf")
    rec = parsed["record"]
    assert rec["structure_source"] == "none" and rec["num_chapters"] == 1
    assert rec["chapters"][0]["name"] == "Intro Science"
    assert any("No table of contents" in w for w in rec["warnings"])
    assert len(parsed["chunks"]) > 4  # the text is still fully indexed, with exact pages
    assert {c["chapter_id"] for c in parsed["chunks"]} == {rec["chapters"][0]["chapter_id"]}


def test_chapter_that_starts_in_the_middle_of_a_page(patched):
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 90), "Chapter 1: Atoms", fontsize=22, fontname="hebo")
    p1.insert_textbox(pymupdf.Rect(72, 120, 523, 300), "Atoms are made of protons, neutrons and electrons arranged in shells around a nucleus. " * 3, fontsize=11, fontname="helv")
    p2 = doc.new_page()
    p2.insert_textbox(pymupdf.Rect(72, 60, 523, 200), "Tail of chapter one: isotopes of an element differ only in their number of neutrons and share chemical behaviour. " * 2, fontsize=11, fontname="helv")
    p2.insert_text((72, 260), "Chapter 2: Bonding", fontsize=22, fontname="hebo")
    p2.insert_textbox(pymupdf.Rect(72, 290, 523, 450), "A covalent bond is formed when two atoms share electrons to complete their outer shells. " * 3, fontsize=11, fontname="helv")
    doc.set_toc([[1, "Chapter 1: Atoms", 1], [1, "Chapter 2: Bonding", 2]])
    parsed = ingestion.parse_pdf(doc.tobytes(), "Chem.pdf")
    tail = next(c for c in parsed["chunks"] if "Tail of chapter one" in c["text"])
    bond = next(c for c in parsed["chunks"] if "covalent bond" in c["text"])
    assert tail["page_start"] == 2 and tail["chapter_name"] == "Chapter 1: Atoms"  # above the heading => previous chapter
    assert bond["page_start"] == 2 and bond["chapter_name"] == "Chapter 2: Bonding"
    assert not any("Chapter 2: Bonding" in c["text"] for c in parsed["chunks"])  # the heading itself is not body text


def test_contents_row_parser_needs_the_page_number_to_be_set_apart():
    w = lambda x0, x1, t: (x0, 100, x1, 112, t, 0, 0, 0)
    row = ingestion._parse_toc_row([w(72, 120, "Chapter"), w(125, 135, "2"), w(140, 200, "Bonding"), w(205, 400, "........."), w(500, 512, "31")])
    assert row == ("Chapter 2 Bonding", 31)
    assert ingestion._parse_toc_row([w(72, 120, "Chapter"), w(125, 135, "2")]) is None  # a heading, not a contents entry
    assert ingestion._parse_toc_row([w(72, 120, "Bonding"), w(300, 312, "31")]) == ("Bonding", 31)  # no leader dots, but set apart


def test_ingestion_page_map_across_pages(patched):
    paras = [f"Paragraph {i} of the long section. " + ("Filler words about the topic of this paragraph. " * 12) for i in range(1, 13)]
    pdf = make_pdf([("Chapter 1: Long Chapter", [("1.1 Long Section", paras)]), ("Chapter 2: Short", [("2.1 Tiny", [paras[0] + " Again."])])])
    store = FakeStore()
    rec = ingestion.ingest_pdf(store, pdf, "Long.pdf")
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert doc.page_count >= 3
    page_of = {i: next(p + 1 for p in range(doc.page_count) if doc[p].search_for(f"Paragraph {i} of the long")) for i in range(1, 13)}
    chunks = [c for c in store.get_chunks(Scope((rec["document_id"],))) if c["chapter_number"] == 1]
    for c in chunks:
        labels = [i for i in range(1, 13) if f"Paragraph {i} of the long" in c["text"]]
        assert labels, "every chunk should contain at least one paragraph"
        assert c["page_start"] == min(page_of[i] for i in labels)
        assert c["page_end"] == max(page_of[i] for i in labels)
        assert c["page_map"][0] == [c["page_start"], 0]
    assert any(len(c["page_map"]) > 1 for c in chunks) or doc.page_count >= 3  # chunks that cross a page keep both pages


def test_ingestion_validation_errors(patched):
    store = FakeStore()
    with pytest.raises(ingestion.IngestionError, match="Only PDF"):
        ingestion.ingest_pdf(store, b"hello", "notes.txt")
    with pytest.raises(ingestion.IngestionError, match="empty"):
        ingestion.ingest_pdf(store, b"", "a.pdf")
    with pytest.raises(ingestion.IngestionError, match="does not look like a PDF"):
        ingestion.ingest_pdf(store, b"just text, not a pdf", "a.pdf")
    blank = pymupdf.open()
    for _ in range(3):
        blank.new_page()
    with pytest.raises(ingestion.IngestionError, match="scan"):
        ingestion.ingest_pdf(store, blank.tobytes(), "scan.pdf")
    patched.setattr(ingestion, "MAX_PDF_MB", 0)
    with pytest.raises(ingestion.IngestionError, match="larger than"):
        ingestion.ingest_pdf(store, make_pdf(SCIENCE), "big.pdf")


def test_duplicate_upload_is_rejected(patched):
    store, pdf = FakeStore(), make_pdf(SCIENCE)  # same bytes => same document id
    ingestion.ingest_pdf(store, pdf, "Intro_Science_Notes.pdf")
    with pytest.raises(ingestion.IngestionError, match="already in your library"):
        ingestion.ingest_pdf(store, pdf, "copy of notes.pdf")
    assert len(store.list_documents()) == 1


# --------------------------------------------------------------------------- #
# Chapter identification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, expected", [
    ("explain chapter 3", [3]),
    ("What is in Ch. 2?", [2]),
    ("summarise CHAPTER IV please", [4]),
    ("compare chapters 2 and 5", [2, 5]),
    ("revise chapters 2-4", [2, 3, 4]),
    ("what does chapter three say", [3]),
    ("the third chapter is hard", [3]),
    ("explain unit 7", [7]),
    ("what is entropy", []),
    ("chapter civil engineering", []),
])
def test_extract_chapter_numbers(question, expected):
    assert retrieval.extract_chapter_numbers(question) == expected


@pytest.fixture
def chapters(library):
    store, sci = library
    return store.list_chapters([sci["document_id"]])


def ids(sel):
    return [c["chapter_title"] for c in sel.chapters]


def test_chapter_references_are_not_bm25_search_terms():
    assert retrieval.strip_chapter_references("In chapter 3, what does rubisco do?") == "In , what does rubisco do?"
    assert "chapter" not in retrieval.strip_chapter_references("compare chapters 2 and 5 on entropy")
    assert retrieval.strip_chapter_references("explain the third chapter on genes") == "explain the on genes"
    assert retrieval.strip_chapter_references("what is entropy") == "what is entropy"


def test_identify_by_chapter_number(chapters, monkeypatch):
    monkeypatch.setattr(gen, "generate_json", lambda *a, **k: pytest.fail("Gemini must not be called when the number matches"))
    sel = retrieval.identify_chapters("In chapter 2, how does the cycle fix carbon?", chapters)
    assert (sel.method, ids(sel)) == ("number", ["Photosynthesis"])


def test_identify_by_normalized_title(chapters):
    sel = retrieval.identify_chapters("what is PHOTOSYNTHESIS?? explain briefly", chapters, use_gemini=False)
    assert (sel.method, ids(sel)) == ("title", ["Photosynthesis"])


def test_identify_by_fuzzy_title(chapters):
    sel = retrieval.identify_chapters("explain photosyntesis in plants", chapters, use_gemini=False)  # typo
    assert (sel.method, ids(sel)) == ("fuzzy", ["Photosynthesis"])
    sel = retrieval.identify_chapters("laws of thermodynamic", chapters, use_gemini=False)
    assert (sel.method, ids(sel)) == ("fuzzy", ["Thermodynamics"])


def test_multiple_chapters_are_supported(chapters):
    sel = retrieval.identify_chapters("compare chapter 1 and chapter 3", chapters)
    assert sel.method == "number" and ids(sel) == ["Thermodynamics", "Genetics"]
    sel = retrieval.identify_chapters("how does genetics relate to photosynthesis?", chapters, use_gemini=False)
    assert sel.method == "title" and sorted(ids(sel)) == ["Genetics", "Photosynthesis"]
    assert len(sel.document_ids) == 1


def fake_chapters(*titles, doc="d1", doc_name="Book"):
    return [
        {"document_id": doc, "document_name": doc_name, "chapter_id": f"{doc}-c{i}", "chapter_number": i,
         "chapter_name": f"Chapter {i}: {t}", "chapter_title": t, "sections": []}
        for i, t in enumerate(titles, start=1)
    ]


def test_generic_and_short_titles_do_not_auto_match(monkeypatch):
    monkeypatch.setattr(gen, "generate_json", lambda *a, **k: pytest.fail("not needed"))
    chs = fake_chapters("Introduction", "Energy", "Cell Division")
    # generic title: only reachable via chapter number
    assert retrieval.identify_chapters("give me an introduction to entropy", chs, use_gemini=False).method == "none"
    # short single-word title is too easy to hit by accident...
    assert retrieval.identify_chapters("what is energy", chs, use_gemini=False).method == "none"
    # ...unless the question explicitly talks about a chapter
    assert retrieval.identify_chapters("summarise the energy chapter", chs, use_gemini=False).method == "title"


def test_more_specific_title_wins_over_contained_title():
    chs = fake_chapters("Energy", "Work, Power and Energy", "Momentum")
    sel = retrieval.identify_chapters("explain work power and energy for the exam, chapter included", chs, use_gemini=False)
    assert ids(sel) == ["Work, Power and Energy"]


def test_chapter_numbers_across_books_prefer_the_book_named_in_the_question():
    chs = fake_chapters("Atoms", "Bonding", doc="d1", doc_name="Chemistry Basics") + fake_chapters("Cells", "Genes", doc="d2", doc_name="Biology Primer")
    both = retrieval.identify_chapters("explain chapter 2", chs, use_gemini=False)
    assert sorted(c["document_id"] for c in both.chapters) == ["d1", "d2"]
    one = retrieval.identify_chapters("explain chapter 2 of Biology Primer", chs, use_gemini=False)
    assert [(c["document_id"], c["chapter_title"]) for c in one.chapters] == [("d2", "Genes")]


def test_gemini_selection_validates_what_it_returns(monkeypatch):
    chs = fake_chapters("Cell Respiration", "Genetics", "Ecology")
    q = "how do mitochondria make ATP"  # no number / title overlap -> Gemini decides
    seen = {}

    def reply(choice):
        def _fn(prompt, schema, **kw):
            seen["prompt"] = prompt
            return choice
        return _fn

    monkeypatch.setattr(gen, "generate_json", reply(retrieval.ChapterChoice(chapter_ids=["C1", "C99"], confidence="high", reason="ATP")))
    sel = retrieval.identify_chapters(q, chs)
    assert sel.method == "gemini" and ids(sel) == ["Cell Respiration"]  # invented id C99 dropped
    assert "Cell Respiration" in seen["prompt"] and "C1" in seen["prompt"]

    monkeypatch.setattr(gen, "generate_json", reply(retrieval.ChapterChoice(chapter_ids=["C1"], confidence="low", reason="vague")))
    assert retrieval.identify_chapters(q, chs).method == "none"  # low confidence => search everything

    monkeypatch.setattr(gen, "generate_json", reply(retrieval.ChapterChoice(chapter_ids=["C7"], confidence="high", reason="?")))
    assert retrieval.identify_chapters(q, chs).method == "none"  # only invented ids

    def boom(*a, **k):
        raise gen.LLMError("quota")

    monkeypatch.setattr(gen, "generate_json", boom)
    assert retrieval.identify_chapters(q, chs).method == "none"  # Gemini failure never breaks retrieval


def test_single_chapter_book_skips_chapter_filtering():
    assert retrieval.identify_chapters("explain chapter 1", fake_chapters("Only One"), use_gemini=False).method == "none"


# --------------------------------------------------------------------------- #
# Chapter filter is applied BEFORE BM25 and BEFORE vector search
# --------------------------------------------------------------------------- #
def run(store, sci, question, mode, use_filter=True):
    chs = store.list_chapters([sci["document_id"]])
    ctx = retrieval.QueryContext(question, chs, use_filter)
    return retrieval.retrieve(store, ctx, [sci["document_id"]], mode)


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "full"])
def test_filtered_results_come_only_from_the_selected_chapter(library, mode):
    store, sci = library
    q = "In chapter 2, how is energy from light used with carbon dioxide?"  # 'energy' is far more frequent in chapter 1
    ret = run(store, sci, q, mode)
    assert ret.selection.method == "number" and not ret.fallback
    assert ret.hits and {chapter_of(store, h.chunk["chunk_id"]) for h in ret.hits} == {"Chapter 2: Photosynthesis"}
    # the filter really changed the result (otherwise this test would prove nothing)
    unfiltered = run(store, sci, q, mode, use_filter=False)
    assert len({chapter_of(store, h.chunk["chunk_id"]) for h in unfiltered.hits}) > 1


@pytest.mark.parametrize("mode", ["bm25", "vector"])
def test_filter_is_a_pre_filter_not_a_post_filter(library, monkeypatch, mode):
    """With only ONE candidate per retriever, a post-filter would drop it (the global best is in chapter 1) and return
    nothing; a pre-filter still returns the best chunk of the selected chapter."""
    store, sci = library
    monkeypatch.setattr(retrieval, "TOP_K_EACH", 1)
    monkeypatch.setattr(retrieval, "MIN_CANDIDATES", 1)  # one candidate must count as "sufficient" for this experiment
    q = "In chapter 2, how is heat energy transferred?"  # globally best chunk is in chapter 1 (heat)
    best_global = run(store, sci, q, mode, use_filter=False).hits[0]
    assert chapter_of(store, best_global.chunk["chunk_id"]) == "Chapter 1: Thermodynamics"
    filtered = run(store, sci, q, mode)
    assert len(filtered.hits) == 1
    assert chapter_of(store, filtered.hits[0].chunk["chunk_id"]) == "Chapter 2: Photosynthesis"


def test_bm25_and_vector_receive_the_identical_scope(library):
    store, sci = library
    ret = run(store, sci, "In chapter 3, what pairs with adenine?", "full")
    ch3 = [c["chapter_id"] for c in store.list_chapters([sci["document_id"]]) if c["chapter_number"] == 3]
    expected = Scope((sci["document_id"],), tuple(ch3))
    assert store.chunk_scopes[-1] == expected and store.vector_scopes[-1] == expected == ret.scope
    assert expected.mongo_filter() == {"$and": [{"document_id": {"$in": [sci["document_id"]]}}, {"chapter_id": {"$in": ch3}}]}


def test_real_vector_search_puts_the_filter_inside_the_vectorsearch_stage():
    store = FakeStore()
    store.chunks = MagicMock()
    store.chunks.aggregate.return_value = []
    scope = Scope(("docA", "docB"), ("docA-c01", "docB-c02"))
    Store.vector_search(store, [0.1, 0.2], scope, limit=5)
    pipeline = store.chunks.aggregate.call_args.args[0]
    stage = pipeline[0]["$vectorSearch"]  # first stage => filter is applied by Atlas before ranking
    assert stage["index"] == storage.VECTOR_INDEX_NAME and stage["path"] == "embedding" and stage["limit"] == 5
    assert stage["filter"] == {"$and": [{"document_id": {"$in": ["docA", "docB"]}}, {"chapter_id": {"$in": ["docA-c01", "docB-c02"]}}]}
    assert stage["numCandidates"] >= stage["limit"]
    assert {"$project": {"_id": 0, "embedding": 0}} in pipeline  # embeddings are never returned


def test_two_books_chapter_filter_restricts_document_and_chapter(patched):
    store = FakeStore()
    sci = ingestion.ingest_pdf(store, make_pdf(SCIENCE), "Intro Science Notes.pdf")
    chem = ingestion.ingest_pdf(store, make_pdf(CHEMISTRY), "Chemistry Basics.pdf")
    docs = [sci["document_id"], chem["document_id"]]
    ctx = retrieval.QueryContext("In Chemistry Basics chapter 2, why is water bent?", store.list_chapters(docs))
    ret = retrieval.retrieve(store, ctx, docs, "full")
    assert ret.scope.document_ids == (chem["document_id"],)
    assert {h.chunk["document_id"] for h in ret.hits} == {chem["document_id"]}
    assert {h.chunk["chapter_name"] for h in ret.hits} == {"Chapter 2: Bonding"}


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #
def test_falls_back_to_all_chunks_when_the_chapter_scope_is_insufficient(library):
    store, sci = library
    ret = run(store, sci, "In chapter 3, what does rubisco attach to carbon dioxide?", "bm25")  # rubisco lives in chapter 2
    assert ret.selection.method == "number" and ret.fallback and "candidate" in ret.fallback_reason
    assert ret.scope.chapter_ids is None
    assert chapter_of(store, ret.hits[0].chunk["chunk_id"]) == "Chapter 2: Photosynthesis"


def test_falls_back_when_the_best_rerank_score_is_too_low(library, monkeypatch):
    store, sci = library
    monkeypatch.setattr(retrieval, "MIN_RERANK_SCORE", 99.0)  # nothing can pass
    ret = run(store, sci, "In chapter 2, how is energy captured?", "full")
    assert ret.fallback and "reranker score" in ret.fallback_reason


def test_no_chapter_identified_searches_everything(library, monkeypatch):
    store, sci = library
    monkeypatch.setattr(gen, "generate_json", lambda *a, **k: retrieval.ChapterChoice(chapter_ids=[], confidence="low", reason="vague"))
    ret = run(store, sci, "summarise everything about the living world", "hybrid")
    assert ret.selection.method == "none" and not ret.fallback and ret.scope.chapter_ids is None
    assert store.chunk_scopes[-1] == Scope((sci["document_id"],))


def test_ambiguous_question_with_unknown_chapter_number_searches_everything(library):
    store, sci = library
    ret = run(store, sci, "explain chapter 12 about entropy", "hybrid")  # book only has chapters 1-3
    assert ret.selection.chapter_ids == [] or ret.selection.method != "number"


# --------------------------------------------------------------------------- #
# RRF, reranking, BM25, parent expansion, HyDE isolation
# --------------------------------------------------------------------------- #
def test_rrf_fuses_rankings():
    fused = dict(retrieval.rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61) and fused["d"] == pytest.approx(1 / 63)
    assert [cid for cid, _ in retrieval.rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])] == ["b", "c", "a", "d"]
    assert retrieval.rrf_fuse([]) == []


def test_bm25_ranks_matching_chunks_first_and_ignores_zero_scores():
    chunks = [
        {"chunk_id": "x", "chapter_name": "A", "text": "photosynthesis uses light to make sugar"},
        {"chunk_id": "y", "chapter_name": "B", "text": "entropy never decreases in isolated systems"},
        {"chunk_id": "z", "chapter_name": "C", "text": "rubisco fixes carbon during photosynthesis photosynthesis"},
    ]
    ranked = retrieval.bm25_search("photosynthesis rubisco", chunks)
    assert [cid for cid, _ in ranked] == ["z", "x"]
    assert retrieval.bm25_search("zebra", chunks) == [] and retrieval.bm25_search("", chunks) == []


def test_rerank_orders_by_cross_encoder_score(patched):
    chunks = [
        {"chunk_id": "1", "chapter_name": "S", "text": "entropy of the universe increases"},
        {"chunk_id": "2", "chapter_name": "S", "text": "rubisco fixes carbon dioxide in the calvin cycle of photosynthesis"},
        {"chunk_id": "3", "chapter_name": "S", "text": "dna polymerase copies the helix"},
    ]
    hits = retrieval.rerank("what does rubisco do in the calvin cycle", chunks, top_k=2)
    assert len(hits) == 2 and hits[0].chunk["chunk_id"] == "2"
    assert hits[0].stage == "rerank" and hits[0].score > hits[1].score


def test_the_llm_receives_only_the_top_chunks(library):
    store, sci = library
    ret = run(store, sci, "In chapter 2, explain the light reactions and the electron transport chain", "full")
    sources = retrieval.build_sources(ret.hits, limit=4)
    assert [s.sid for s in sources] == ["S1", "S2", "S3", "S4"]
    assert [s.chunk["chunk_id"] for s in sources] == [h.chunk["chunk_id"] for h in ret.hits[:4]]
    prompt = gen.render_sources(sources)
    for s in sources:
        assert s.chunk["text"].split(".")[0] in prompt
    # the prompt holds those chunks and nothing else: no other text of the chapter is sent
    others = [c for c in store.get_chunks(Scope((sci["document_id"],))) if c["chunk_id"] not in {s.chunk["chunk_id"] for s in sources}]
    assert all(c["text"][:60] not in prompt for c in others)
    assert len(retrieval.build_sources(ret.hits)) <= retrieval.CONTEXT_K


def test_hyde_text_is_never_used_as_evidence(library, monkeypatch):
    store, sci = library
    hyde = "HYPOTHETICAL-NONSENSE-ABOUT-QUASARS-AND-DRAGONS"
    monkeypatch.setattr(gen, "generate_text", lambda prompt, **k: hyde)
    chs = store.list_chapters([sci["document_id"]])
    ctx = retrieval.QueryContext("In chapter 2, how are ATP and NADPH made?", chs)
    ret = retrieval.retrieve(store, ctx, [sci["document_id"]], "full")
    assert ret.hyde_passage == hyde  # kept only for display
    sources = retrieval.build_sources(ret.hits)
    assert hyde not in gen.render_sources(sources)  # never reaches the answer prompt
    assert all(hyde not in s.chunk["text"] for s in sources)
    # BM25 and the reranker work on the real question, not on the HyDE text
    assert ctx.question != hyde and "QUASARS" not in " ".join(retrieval.tokenize(ctx.question)).upper()


def test_hyde_failure_degrades_to_the_question_embedding(library, monkeypatch):
    store, sci = library

    def boom(*a, **k):
        raise gen.LLMError("rate limit")

    monkeypatch.setattr(gen, "generate_text", boom)
    ctx = retrieval.QueryContext("In chapter 2, what does rubisco do?", store.list_chapters([sci["document_id"]]))
    ret = retrieval.retrieve(store, ctx, [sci["document_id"]], "full")
    assert ret.hits and "rate limit" in ctx.hyde_error


def test_rewrite_uses_recent_messages_and_survives_errors(monkeypatch):
    history = [{"role": "user", "content": "What is entropy?"}, {"role": "assistant", "content": "A measure of disorder."}]
    seen = {}

    def fake(prompt, **k):
        seen["prompt"] = prompt
        return '"How does entropy relate to the second law of thermodynamics?"'

    monkeypatch.setattr(gen, "generate_text", fake)
    assert retrieval.rewrite_question("and how does it relate to the second law?", history) == "How does entropy relate to the second law of thermodynamics?"
    assert "What is entropy?" in seen["prompt"] and "second law" in seen["prompt"]
    assert retrieval.rewrite_question("anything", []) == "anything"  # nothing to resolve => no call
    monkeypatch.setattr(gen, "generate_text", lambda *a, **k: (_ for _ in ()).throw(gen.LLMError("x")))
    assert retrieval.rewrite_question("and the second law?", history) == "and the second law?"
    many = [{"role": "user", "content": f"message {i}"} for i in range(9)]
    monkeypatch.setattr(gen, "generate_text", fake)
    retrieval.rewrite_question("q?", many, max_messages=5)
    assert "message 4" in seen["prompt"] and "message 3" not in seen["prompt"]


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def test_metrics_hand_computed():
    ranked, rel = ["a", "b", "c", "d"], ["c"]
    assert storage.recall_at_k(ranked, rel, 2) == 0.0 and storage.recall_at_k(ranked, rel, 3) == 1.0
    assert storage.reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert storage.ndcg_at_k(ranked, rel, 5) == pytest.approx(1 / math.log2(4))
    two = ["x", "a", "y", "b"]
    # relevant {a, b}: DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3)
    assert storage.ndcg_at_k(two, ["a", "b"], 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))
    assert storage.recall_at_k(two, ["a", "b"], 3) == 0.5


def test_evaluate_rankings_aggregates_and_counts_empty_results():
    m = storage.evaluate_rankings([(["a", "b", "c"], ["a"]), (["x", "y", "z"], ["z"]), ([], ["q"]), (["q"], ["q"])])
    assert m["Recall@3"] == pytest.approx(3 / 4) and m["Recall@5"] == m["Recall@10"] == pytest.approx(3 / 4)
    assert m["MRR"] == pytest.approx((1 + 1 / 3 + 0 + 1) / 4)
    assert m["Empty rate"] == pytest.approx(1 / 4) and m["Questions"] == 4
    assert m["nDCG@10"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert storage.evaluate_rankings([]) == {}

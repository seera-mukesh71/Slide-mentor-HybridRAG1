"""End-to-end tests of the whole workflow with a scripted fake Gemini (no network, no Atlas):

upload PDF -> extract chapter metadata -> identify chapter -> filter before retrieval -> HyDE + BM25 + vector search
-> RRF -> reranking -> grounded answer -> self-critique -> validated, clickable page citations.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generation as gen  # noqa: E402
import retrieval  # noqa: E402
from rag_pipeline import EVAL_SYSTEMS, RAGPipeline  # noqa: E402
from storage import Scope  # noqa: E402
from test_retrieval import SCIENCE, FakeStore, make_pdf, patched  # noqa: E402,F401  (patched = shared fixture)

HYDE_MARKER = "HYPOTHETICAL-PASSAGE-ABOUT-DRAGONS"
SOURCE_TAG = re.compile(r'<source id="(S\d+)" textbook="([^"]*)" chapter="([^"]*)" pages="(\d+)(?:-(\d+))?">')


def sources_in(prompt: str) -> list[dict]:
    return [
        {"sid": m[0], "book": m[1], "chapter": m[2], "start": int(m[3]), "end": int(m[4] or m[3])}
        for m in SOURCE_TAG.findall(prompt)
    ]


def draft(answer, cites, coverage="full", missing=""):
    return gen.AnswerDraft(answer=answer, citations=[gen.CitedPage(source_id=s, page=p) for s, p in cites], coverage=coverage, missing=missing)


def critique(verdict="pass", unsupported=(), incorrect=(), missing=(), feedback=""):
    return gen.Critique(verdict=verdict, unsupported_claims=list(unsupported), incorrect_citations=list(incorrect),
                        missing_from_answer=list(missing), feedback=feedback)


class FakeGemini:
    """Scripted stand-in for generation.generate_text / generate_json. Records every call."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.drafts: list = []  # AnswerDraft or callable(prompt) -> AnswerDraft, consumed in order
        self.critiques: list = []
        self.rewrite_to: str | None = None
        self.raise_on: set[str] = set()
        self.chapter_choice = None  # optional callable(prompt) -> ChapterChoice

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)

    def prompts(self, kind: str) -> list[str]:
        return [p for k, p in self.calls if k == kind]

    def text(self, prompt, **kw):
        if prompt.startswith("Rewrite the student's latest question"):
            self.calls.append(("rewrite", prompt))
            return self.rewrite_to or prompt.rsplit("LATEST QUESTION:", 1)[-1].strip()
        self.calls.append(("hyde", prompt))
        return f"{HYDE_MARKER} " + prompt.rsplit("QUESTION:", 1)[-1].strip()

    def json(self, prompt, schema, **kw):
        name = schema.__name__
        self.calls.append((name, prompt))
        if name in self.raise_on:
            raise gen.LLMError(f"{name} unavailable")
        if name == "ChapterChoice":
            if self.chapter_choice:
                return self.chapter_choice(prompt)
            return retrieval.ChapterChoice(chapter_ids=[], confidence="low", reason="fake")
        if name == "AnswerDraft":
            item = self.drafts.pop(0)
            return item(prompt) if callable(item) else item
        if name == "Critique":
            return self.critiques.pop(0) if self.critiques else critique()
        if name == "EvalQuestion":
            passage = prompt.split("PASSAGE:\n", 1)[1]
            words = re.findall(r"[A-Za-z]{5,}", passage)[:9]
            from rag_pipeline import EvalQuestion

            return EvalQuestion(question="Explain " + " ".join(words) + "?")
        raise AssertionError(f"unexpected schema {name}")


def cite_first_source(text="Rubisco attaches carbon dioxide to ribulose bisphosphate in the Calvin cycle"):
    def make(prompt):
        s = sources_in(prompt)[0]
        return draft(f"{text} [{s['sid']}].", [(s["sid"], s["start"])])
    return make


@pytest.fixture
def llm(patched):
    fake = FakeGemini()
    patched.setattr(gen, "generate_text", fake.text)
    patched.setattr(gen, "generate_json", fake.json)
    return fake


@pytest.fixture
def app_env(llm):
    store = FakeStore()
    pipeline = RAGPipeline(store, ensure_indexes=False)
    progress = []
    rec = pipeline.ingest(make_pdf(SCIENCE), "Intro_Science_Notes.pdf", progress=lambda m, f: progress.append((m, f)))
    return pipeline, store, rec, llm, progress


Q = "In chapter 2, what does rubisco do in the Calvin cycle?"


# --------------------------------------------------------------------------- #
# The complete flow
# --------------------------------------------------------------------------- #
def test_complete_flow_upload_to_cited_answer(app_env):
    pipeline, store, rec, llm, progress = app_env

    # 1. upload -> chapter metadata extracted and indexed
    assert progress[0][1] < progress[-1][1] == 1.0 and any("Embedding passages" in m for m, _ in progress)
    assert [(c["number"], c["name"]) for c in rec["chapters"]] == [(1, "Chapter 1: Thermodynamics"), (2, "Chapter 2: Photosynthesis"), (3, "Chapter 3: Genetics")]
    ch2 = rec["chapters"][1]["chapter_id"]

    # 2-8. ask
    llm.drafts = [cite_first_source()]
    llm.critiques = [critique("pass")]
    msg = pipeline.ask(Q, "sess1", [rec["document_id"]])
    meta, trace = msg["meta"], msg["meta"]["trace"]

    # chapter identified from the stored chapter names/numbers, and the scope was filtered BEFORE retrieval
    assert trace["chapter_method"] == "number" and trace["scope_chapter_ids"] == [ch2] and not trace["fallback"]
    expected_scope = Scope((rec["document_id"],), (ch2,))
    assert store.chunk_scopes[-1] == expected_scope and store.vector_scopes[-1] == expected_scope  # BM25 and vector, same scope
    assert llm.count("ChapterChoice") == 0  # a chapter number is confident enough; Gemini selection not needed

    # HyDE was generated and used for the vector query, but never shown to the answering model as evidence
    assert llm.count("hyde") == 1 and trace["hyde_passage"].startswith(HYDE_MARKER)
    answer_prompt = llm.prompts("AnswerDraft")[0]
    assert HYDE_MARKER not in answer_prompt

    # the passages given to the model are real textbook text from chapter 2 only
    srcs = sources_in(answer_prompt)
    assert srcs and {s["chapter"] for s in srcs} == {"Chapter 2: Photosynthesis"}
    assert "rubisco attaches carbon dioxide to ribulose bisphosphate" in answer_prompt.lower()

    # grounded answer + self-critique with the same model, no regeneration needed
    assert meta["status"] == "answered" and llm.count("Critique") == 1 and not trace["regenerated"]
    assert llm.count("AnswerDraft") == 1

    # citations: textbook name, chapter, page - validated against the stored metadata
    assert "[1]" in msg["content"] and "[S1]" not in msg["content"]
    c = meta["citations"][0]
    assert (c["document_name"], c["chapter_name"], c["chapter_number"], c["page"]) == ("Intro Science Notes", "Chapter 2: Photosynthesis", 2, 2)
    assert "rubisco" in c["excerpt"].lower() and c["page_start"] <= c["page"] <= c["page_end"]
    chunk = store.get_chunks_by_ids([c["chunk_id"]])[0]
    assert (chunk["document_id"], chunk["page_start"]) == (c["document_id"], 2)

    # the PDF needed by the page viewer is stored, and the cited page really contains the excerpt
    import pymupdf

    pdf = pymupdf.open(stream=store.load_pdf(rec["document_id"]), filetype="pdf")
    assert "rubisco" in pdf[c["page"] - 1].get_text().lower()

    # conversation memory persisted with the citation metadata (so reloads restore the clickable sources)
    stored = store.get_messages("sess1")
    assert [m["role"] for m in stored] == ["user", "assistant"] and stored[1]["meta"]["citations"][0]["page"] == 2


def test_chapter_is_found_without_a_number_via_gemini_and_still_filters(app_env):
    pipeline, store, rec, llm, _ = app_env
    ch3 = rec["chapters"][2]["chapter_id"]

    def choose_ch3(prompt):
        alias = re.search(r"(C\d+) \| ch 3 \|", prompt).group(1)  # Gemini sees ids, numbers and titles - never raw db ids
        return retrieval.ChapterChoice(chapter_ids=[alias], confidence="high", reason="alleles")

    llm.chapter_choice = choose_ch3
    llm.drafts = [cite_first_source("Alleles are alternative forms of a gene")]
    msg = pipeline.ask("What is an allele and how do dominant ones behave?", "s", [rec["document_id"]])
    assert msg["meta"]["trace"]["chapter_method"] == "gemini" and msg["meta"]["trace"]["scope_chapter_ids"] == [ch3]
    assert llm.count("ChapterChoice") == 1 and "Genetics" in llm.prompts("ChapterChoice")[0]
    assert {s["chapter"] for s in sources_in(llm.prompts("AnswerDraft")[0])} == {"Chapter 3: Genetics"}


# --------------------------------------------------------------------------- #
# Self-critique, regeneration, citation validation
# --------------------------------------------------------------------------- #
def test_invalid_ids_and_pages_are_repaired_in_python_and_trigger_one_regeneration(app_env):
    pipeline, store, rec, llm, _ = app_env

    def bad_first_draft(prompt):
        s = sources_in(prompt)[0]
        return draft(f"Rubisco fixes carbon [{s['sid']}] and also does magic [S9] (p. 77).", [(s["sid"], 99), ("S9", 3)])

    llm.drafts = [bad_first_draft, cite_first_source()]
    llm.critiques = [critique("pass")]  # the critic is happy, but Python found an unknown source id
    msg = pipeline.ask(Q, "s", [rec["document_id"]])
    assert llm.count("AnswerDraft") == 2 and llm.count("Critique") == 1  # exactly one regeneration
    second_prompt = llm.prompts("AnswerDraft")[1]
    assert "unknown source id S9" in second_prompt.lower() or "Unknown source id S9" in second_prompt
    assert "REVIEWER FEEDBACK" in second_prompt and "also does magic" in second_prompt  # previous answer + feedback were provided
    assert msg["meta"]["trace"]["regenerated"] is True and msg["meta"]["status"] == "answered"
    assert "S9" not in msg["content"] and "magic" not in msg["content"]


def test_validate_citations_directly_removes_unknown_ids_and_bad_pages():
    chunk = {"chunk_id": "k1", "document_id": "d", "document_name": "Book", "chapter_id": "d-c00", "chapter_number": 1,
             "chapter_name": "Chapter 1: A", "page_start": 4, "page_end": 5,
             "text": "page four text\n\npage five text", "page_map": [[4, 0], [5, 16]]}
    src = gen.Source("S1", chunk)
    d = draft("Fact one [S1]. Fact two [S2][S1]. See (p. 99) and (page 6).", [("S1", 123), ("S2", 1)])
    v = gen.validate_citations(d, [src])
    assert v.unknown_ids and [c.sid for c in v.citations] == ["S1"]
    cit = v.citations[0]
    assert cit.page == 4 and cit.corrected and not cit.page_exact  # 123 is not in the source -> corrected to its first page
    assert "[S2]" not in v.text_sids and "p. 99" not in v.text and "(page 6)" not in v.text  # page 6 is not in the chunk the model was shown
    assert v.text.startswith("Fact one [1]. Fact two [1]")
    # a page inside the chunk is accepted, with the excerpt taken from that page
    child_page = gen.validate_citations(draft("Detail [S1].", [("S1", 5)]), [src]).citations[0]
    assert child_page.excerpt == "page five text"


def test_critique_can_force_one_regeneration_but_never_two(app_env):
    pipeline, store, rec, llm, _ = app_env
    llm.drafts = [cite_first_source("A weak first answer"), cite_first_source("An improved answer"), cite_first_source("must never be requested")]
    llm.critiques = [critique("revise", unsupported=["claim X"], feedback="Remove claim X."), critique("revise", feedback="still bad")]
    msg = pipeline.ask(Q, "s", [rec["document_id"]])
    assert llm.count("AnswerDraft") == 2 and llm.count("Critique") == 1
    assert "claim X" in llm.prompts("AnswerDraft")[1] and "Remove claim X." in llm.prompts("AnswerDraft")[1]
    assert msg["content"].startswith("An improved answer") and msg["meta"]["trace"]["regenerated"]


def test_critique_outage_does_not_lose_the_answer(app_env):
    pipeline, store, rec, llm, _ = app_env
    llm.raise_on = {"Critique"}
    llm.drafts = [cite_first_source()]
    msg = pipeline.ask(Q, "s", [rec["document_id"]])
    assert msg["meta"]["status"] == "answered" and msg["meta"]["trace"]["critique"] is None


def test_partial_answer_when_the_textbook_covers_only_part(app_env):
    pipeline, store, rec, llm, _ = app_env
    s1 = lambda p: (lambda s: draft(f"Rubisco fixes carbon dioxide [{s['sid']}].", [(s["sid"], s["start"])], "partial", "The book does not give rubisco's reaction rate."))(sources_in(p)[0])  # noqa: E731
    llm.drafts = [s1]
    msg = pipeline.ask(Q, "s", [rec["document_id"]])
    assert msg["meta"]["status"] == "partial" and "reaction rate" in msg["meta"]["explanation"] and msg["meta"]["citations"]


def test_abstains_with_an_explanation_when_the_book_has_no_evidence(app_env):
    pipeline, store, rec, llm, _ = app_env
    llm.drafts = [draft("", [], "none", "The retrieved passages never mention black holes.")]
    msg = pipeline.ask("What is the event horizon of a black hole?", "s", [rec["document_id"]])
    meta = msg["meta"]
    assert meta["status"] == "abstained" and meta["citations"] == []
    assert "can't answer this from your uploaded textbook" in msg["content"] and "black holes" in msg["content"]
    assert "I searched:" in msg["content"]
    assert 1 <= len(meta["closest"]) <= 3 and all(c["excerpt"] for c in meta["closest"])  # nearest passages shown, not cited as support
    assert llm.count("Critique") == 0 and llm.count("AnswerDraft") == 1  # nothing to critique, nothing to regenerate


def test_abstains_without_calling_the_model_when_nothing_was_retrieved(llm):
    ans = gen.generate_grounded_answer("anything", [], "all selected books")
    assert ans.status == "abstained" and "No passages" in ans.explanation and llm.calls == []


def test_answer_without_any_valid_citation_is_not_shown(app_env):
    pipeline, store, rec, llm, _ = app_env
    uncited = draft("Rubisco fixes carbon dioxide.", [])
    llm.drafts = [uncited, uncited]
    msg = pipeline.ask(Q, "s", [rec["document_id"]])
    assert msg["meta"]["status"] == "abstained" and "could not produce an answer that I could verify" in msg["content"]
    assert llm.count("AnswerDraft") == 2  # one regeneration, then give up honestly


# --------------------------------------------------------------------------- #
# Conversation memory
# --------------------------------------------------------------------------- #
def test_follow_up_is_rewritten_from_recent_messages(app_env):
    pipeline, store, rec, llm, _ = app_env
    llm.drafts = [cite_first_source(), cite_first_source("The cycle is limited by light and carbon dioxide")]
    pipeline.ask(Q, "mem", [rec["document_id"]])
    assert llm.count("rewrite") == 0  # first question has no history, so nothing to resolve
    llm.rewrite_to = "In chapter 2, what limits the rate of the Calvin cycle?"
    second = pipeline.ask("and what limits it?", "mem", [rec["document_id"]])
    rewrite_prompt = llm.prompts("rewrite")[0]
    assert "rubisco" in rewrite_prompt.lower() and "and what limits it?" in rewrite_prompt  # earlier turns + new question
    t = second["meta"]["trace"]
    assert t["original_question"] == "and what limits it?" and t["standalone_question"] == llm.rewrite_to
    assert t["chapter_method"] == "number"  # the rewritten, standalone question carried the chapter reference
    assert [m["role"] for m in store.get_recent_messages("mem", 5)] == ["user", "assistant", "user", "assistant"]
    assert store.get_messages("other-session") == []  # sessions are isolated


def test_input_validation(app_env):
    pipeline, _, rec, _, _ = app_env
    with pytest.raises(ValueError, match="type a question"):
        pipeline.ask("   ", "s", [rec["document_id"]])
    with pytest.raises(ValueError, match="at least one textbook"):
        pipeline.ask(Q, "s", [])


# --------------------------------------------------------------------------- #
# Gemini error handling
# --------------------------------------------------------------------------- #
def test_retries_transient_errors_and_explains_permanent_ones(monkeypatch):
    from google.genai import errors

    sleeps = []
    monkeypatch.setattr(gen.time, "sleep", sleeps.append)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        return "ok"

    assert gen._with_retries(flaky, "test") == "ok" and attempts["n"] == 3 and len(sleeps) == 2

    def forbidden():
        raise errors.APIError(403, {"error": {"message": "denied", "status": "PERMISSION_DENIED"}})

    with pytest.raises(gen.LLMError, match="GEMINI_API_KEY"):
        gen._with_retries(forbidden, "answering")

    def always_busy():
        raise errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})

    with pytest.raises(gen.LLMError, match="temporarily unavailable"):
        gen._with_retries(always_busy, "answering", attempts=2)


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(gen, "_client", None)
    with pytest.raises(gen.ConfigError, match="GEMINI_API_KEY"):
        gen.get_client()


# --------------------------------------------------------------------------- #
# Evaluation workflow
# --------------------------------------------------------------------------- #
def test_evaluation_generates_questions_and_compares_four_systems(app_env):
    pipeline, store, rec, llm, _ = app_env
    items = pipeline.build_eval_set(rec["document_id"], n=6)
    assert len(items) == 6 and store.get_eval_set(rec["document_id"]) == items
    known = {c["chunk_id"] for c in store.get_chunks(Scope((rec["document_id"],)))}
    assert all(len(i["relevant_chunk_ids"]) == 1 and i["relevant_chunk_ids"][0] in known for i in items)
    assert {i["chapter_id"] for i in items} == {c["chapter_id"] for c in store.list_chapters([rec["document_id"]])}  # every chapter covered
    assert any(i["mentions_chapter"] and "In chapter" in i["question"] for i in items)

    result = pipeline.evaluate(items, [rec["document_id"]], use_chapter_filter=True)
    assert list(result["metrics"]) == [label for label, _ in EVAL_SYSTEMS]
    for label, m in result["metrics"].items():
        assert set(m) >= {"Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10", "Empty rate"}
        assert all(0.0 <= m[k] <= 1.0 for k in m if k != "Questions")
    assert result["metrics"]["BM25"]["Recall@10"] > 0.5  # questions were written from the relevant passages
    assert result["metrics"]["Hybrid + HyDE + rerank"]["MRR"] > 0.3
    cf = result["chapter_filter"]
    assert 0.0 <= cf["identified_rate"] <= 1.0 and 0.0 <= cf["fallback_rate"] <= 1.0
    if cf["correct_when_identified"] is not None:
        assert cf["correct_when_identified"] == 1.0  # identified chapters contain the gold chunk
    off = pipeline.evaluate(items, [rec["document_id"]], use_chapter_filter=False)
    assert off["chapter_filter"] is None


def test_import_eval_set_validates_chunk_ids(app_env):
    pipeline, store, rec, _, _ = app_env
    good = store.get_chunks(Scope((rec["document_id"],)))[0]["chunk_id"]
    ok = pipeline.import_eval_set(rec["document_id"], [{"question": "What is heat?", "relevant_chunk_ids": [good]}])
    assert ok[0]["id"] == "q1"
    with pytest.raises(ValueError, match="Unknown chunk ids"):
        pipeline.import_eval_set(rec["document_id"], [{"question": "?", "relevant_chunk_ids": ["nope"]}])
    with pytest.raises(ValueError, match="non-empty"):
        pipeline.import_eval_set(rec["document_id"], [{"question": "", "relevant_chunk_ids": [good]}])


def test_deleting_a_document_removes_everything(app_env):
    pipeline, store, rec, _, _ = app_env
    pipeline.delete_document(rec["document_id"])
    assert store.get_document(rec["document_id"]) is None and store.load_pdf(rec["document_id"]) is None
    assert store.get_chunks(Scope((rec["document_id"],))) == []


# --------------------------------------------------------------------------- #
# PDF viewer helpers + the Streamlit app itself (headless)
# --------------------------------------------------------------------------- #
def test_pdf_viewer_renders_and_highlights_the_cited_page(app_env):
    import app

    pipeline, store, rec, llm, _ = app_env
    llm.drafts = [cite_first_source()]
    c = pipeline.ask(Q, "v", [rec["document_id"]])["meta"]["citations"][0]
    data = store.load_pdf(rec["document_id"])
    png_header = b"\x89PNG\r\n\x1a\n"
    plain, marked = app.render_pdf_page(data, c["page"]), app.render_pdf_page(data, c["page"], highlight=c["excerpt"])
    assert plain.startswith(png_header) and marked.startswith(png_header)
    assert plain != marked  # the cited text was found on the cited page and highlighted
    other = app.render_pdf_page(data, 1)  # page 1 does not contain the excerpt: nothing to highlight
    assert app.render_pdf_page(data, 1, highlight=c["excerpt"]) == other
    with pytest.raises(Exception):
        app.render_pdf_page(data, 99)  # out-of-range pages are rejected, not silently wrong


def test_page_label_formatting_and_highlight_phrases():
    import app

    c = {"page": 45, "page_start": 45, "page_end": 46, "page_exact": True}
    assert app.page_text(c) == "p. 45"
    assert app.page_text(c, {"45": "33"}) == "p. 45 (printed 33)"  # printed page label differs from the PDF page index
    assert app.page_text({**c, "page_exact": False}) == "pp. 45–46"  # model did not name a page: show the range
    assert app._highlight_phrases("one two three four five six seven")[0] == "one two three four five"
    assert app._highlight_phrases("too short") == ["too short"] and app._highlight_phrases("") == []


# --------------------------------------------------------------------------- #
# The Streamlit app (headless)
# --------------------------------------------------------------------------- #
def test_streamlit_app_shows_clickable_page_citations(app_env, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import storage

    pipeline, store, rec, llm, _ = app_env
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(storage, "Store", lambda: store)  # the app builds Store() when it starts
    llm.drafts = [cite_first_source()]
    llm.critiques = [critique("pass")]

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.query_params["sid"] = "ui-session"
    at.run()
    assert not at.exception, at.exception
    assert any("Intro Science Notes" in c.label for c in at.sidebar.checkbox)  # library listed

    at.chat_input[0].set_value(Q).run()
    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    cite_buttons = [b for b in at.button if b.label.startswith("[1]")]
    assert cite_buttons, "the answer should show a clickable citation"
    label = cite_buttons[0].label
    assert "Intro Science Notes" in label and "Chapter 2: Photosynthesis" in label and "p. 2" in label

    cite_buttons[0].click().run()  # opens the PDF viewer dialog at the cited page
    assert not at.exception, at.exception
    assert at.session_state["v-page"] == 2 and at.session_state["v-doc"] == rec["document_id"]
    assert len(at.get("image")) == 1 and any(b.label == "Download PDF" for b in at.get("download_button"))  # the page is rendered
    assert any("page 2 of 3" in c.value for c in at.caption)
    next(b for b in at.button if b.label == "Next ▶").click().run()  # page navigation inside the viewer
    assert at.session_state["v-page"] == 3

    # the Chapters view lists the extracted structure
    at.sidebar.radio[0].set_value("📖 Chapters").run()
    assert not at.exception, at.exception
    assert any("Chapter 2: Photosynthesis" in m.value for m in at.markdown)
    assert any(b.label == "Open" for b in at.button)  # each chapter can be opened in the PDF viewer


def _run_app(store, llm, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import storage

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(storage, "Store", lambda: store)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.query_params["sid"] = "ui-eval"
    at.run()
    assert not at.exception, at.exception
    return at


def test_streamlit_evaluation_view_runs_and_shows_the_comparison(app_env, monkeypatch):
    pipeline, store, rec, llm, _ = app_env
    at = _run_app(store, llm, monkeypatch)
    at.sidebar.radio[0].set_value("📊 Evaluation").run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Generate questions").click().run()
    assert not at.exception, at.exception
    n_items = len(store.get_eval_set(rec["document_id"]))
    usable = [c for c in store.get_chunks(Scope((rec["document_id"],))) if 50 <= c["word_count"] <= 300]
    assert n_items == len(usable) < 15  # the tiny sample book has fewer passages than the 15 questions asked for
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    assert not at.exception, at.exception
    result = at.session_state["eval_result"]
    assert list(result["metrics"]) == [label for label, _ in EVAL_SYSTEMS] and result["questions"] == n_items
    assert len(at.dataframe) >= 1  # the comparison table is displayed


def test_streamlit_delete_button_removes_the_book(app_env, monkeypatch):
    pipeline, store, rec, llm, _ = app_env
    at = _run_app(store, llm, monkeypatch)
    next(b for b in at.button if b.label == "Delete permanently").click().run()
    assert not at.exception, at.exception
    assert store.list_documents() == [] and store.load_pdf(rec["document_id"]) is None
    assert any("No textbooks yet" in c.value for c in at.sidebar.caption)


def test_gemini_wrapper_builds_valid_sdk_requests_and_parses_responses(monkeypatch):
    """Runs the real google-genai client against a mock HTTP transport: checks model names, thinking budget,
    structured-output schema, embedding task type / dimensionality and the parsing of each response type."""
    import json

    import httpx
    from google import genai
    from google.genai import types

    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content or b"{}")
        seen.append((request.url.path, body))
        if "mbedContent" in request.url.path:
            n = len(body.get("requests", [])) or 1
            return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0] + [0.0] * 766} for _ in range(n)]})
        schema = json.dumps(body.get("generationConfig", {}).get("responseSchema", {}))
        if "coverage" in schema:
            text = json.dumps({"answer": "X [S1].", "citations": [{"source_id": "S1", "page": 2}], "coverage": "full", "missing": ""})
        elif schema != "{}":
            text = json.dumps({"verdict": "pass", "unsupported_claims": [], "incorrect_citations": [], "missing_from_answer": [], "feedback": ""})
        else:
            text = "plain reply"
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})

    client = genai.Client(api_key="fake", http_options=types.HttpOptions(client_args={"transport": httpx.MockTransport(handler)}))
    monkeypatch.setattr(gen, "_client", client)

    assert gen.generate_text("hi", system="be brief") == "plain reply"
    d = gen.generate_json("p", gen.AnswerDraft, thinking_budget=1024)
    assert (d.coverage, d.citations[0].source_id, d.citations[0].page) == ("full", "S1", 2)
    assert gen.generate_json("p", gen.Critique).verdict == "pass"
    vecs = gen.embed_texts(["a", "b", "c"], task_type="RETRIEVAL_QUERY")
    assert len(vecs) == 3 and len(vecs[0]) == gen.EMBED_DIM == 768 and sum(x * x for x in vecs[0]) == pytest.approx(1.0)

    gen_calls = [b for p, b in seen if "generateContent" in p]
    assert all(":generateContent" in p and "gemini-2.5-flash" in p for p, _ in seen if "generateContent" in p)
    assert gen_calls[0]["generationConfig"]["thinkingConfig"] == {"thinking_budget": 0} and "systemInstruction" in gen_calls[0]
    assert gen_calls[1]["generationConfig"]["thinkingConfig"] == {"thinking_budget": 1024}
    assert gen_calls[1]["generationConfig"]["responseMimeType"] == "application/json"
    assert set(gen_calls[1]["generationConfig"]["responseSchema"]["properties"]) == {"answer", "citations", "coverage", "missing"}
    emb = next(b for p, b in seen if "mbedContent" in p)["requests"][0]
    assert emb["taskType"] == "RETRIEVAL_QUERY" and emb["outputDimensionality"] == 768 and emb["model"].endswith("gemini-embedding-001")

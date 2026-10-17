"""app.py - Streamlit interface: upload textbooks, ask exam questions, open cited PDF pages, evaluate retrieval.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import os
import uuid

import pandas as pd
import pymupdf
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import generation as gen  # noqa: E402
import ingestion  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402
from storage import Store, StorageError  # noqa: E402

log = logging.getLogger("student-rag")
st.set_page_config(page_title="Exam Prep Textbook Assistant", page_icon="📚", layout="wide")

EXPECTED_ERRORS = (ingestion.IngestionError, gen.LLMError, gen.ConfigError, StorageError, ValueError)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            st.error("**GEMINI_API_KEY is missing.** Copy `.env.example` to `.env`, add your key from https://aistudio.google.com/apikey and restart.")
            st.stop()
        try:
            st.session_state.pipeline = RAGPipeline(Store())
        except StorageError as e:
            st.error(f"**MongoDB is not ready.** {e}")
            st.stop()
    return st.session_state.pipeline


def ensure_session() -> str:
    """The chat session id lives in the URL (?sid=...) so a browser refresh resumes the same conversation."""
    sid = st.query_params.get("sid")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.query_params["sid"] = sid
    if st.session_state.get("loaded_sid") != sid:
        st.session_state.messages = get_pipeline().store.get_messages(sid)
        st.session_state.loaded_sid = sid
    return sid


# --------------------------------------------------------------------------- #
# PDF page viewer (rendered with PyMuPDF so the exact page always opens, in every browser)
# --------------------------------------------------------------------------- #
def _highlight_phrases(text: str, window: int = 5, max_phrases: int = 8) -> list[str]:
    words = [w for w in text.replace("\n", " ").split() if w]
    if len(words) < window:
        return [" ".join(words)] if words else []
    step = max(1, (len(words) - window) // max(1, max_phrases - 1))
    return [" ".join(words[i : i + window]) for i in range(0, len(words) - window + 1, step)][:max_phrases]


def render_pdf_page(data: bytes, page_no: int, highlight: str | None = None, zoom: float = 1.6) -> bytes:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        page = doc[page_no - 1]
        if highlight:  # best effort: mark the cited text on the page
            for phrase in _highlight_phrases(highlight):
                for rect in page.search_for(phrase)[:3]:
                    page.add_highlight_annot(rect)
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def page_text(c: dict, labels: dict | None = None) -> str:
    """'p. 45' (or 'pp. 45–46' when the exact page is unknown) plus the printed page label if it differs."""
    if c.get("page_exact", True) or c["page_start"] == c["page_end"]:
        txt = f"p. {c['page']}"
        label = (labels or {}).get(str(c["page"]))
        return f"{txt} (printed {label})" if label else txt
    return f"pp. {c['page_start']}–{c['page_end']}"


def _step_page(delta: int) -> None:
    st.session_state["v-page"] = st.session_state["v-page"] + delta


@st.dialog("Textbook page", width="large")
def pdf_viewer() -> None:
    meta, data = st.session_state["v-meta"], st.session_state["v-data"]
    total = meta["num_pages"]
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    c1.button("◀ Prev", key="v-prev", on_click=_step_page, args=(-1,), disabled=st.session_state["v-page"] <= 1)
    c2.button("Next ▶", key="v-next", on_click=_step_page, args=(1,), disabled=st.session_state["v-page"] >= total)
    c3.number_input("Go to page", min_value=1, max_value=total, key="v-page")
    c4.download_button("Download PDF", data, file_name=meta["filename"], mime="application/pdf")
    page_no = st.session_state["v-page"]
    label = (meta.get("page_labels") or {}).get(str(page_no))
    st.caption(f"{meta['name']} · page {page_no} of {total}" + (f" (printed page {label})" if label else ""))
    highlight = st.session_state.get("v-excerpt") if page_no == st.session_state.get("v-cited") else None
    st.image(render_pdf_page(data, page_no, highlight))


def open_viewer(document_id: str, page: int, excerpt: str = "") -> None:
    store = get_pipeline().store
    if st.session_state.get("v-doc") != document_id:  # keep the bytes only for the book being viewed
        data = store.load_pdf(document_id)
        meta = store.get_document(document_id)
        if data is None or meta is None:
            st.error("This textbook's PDF is no longer available.")
            return
        st.session_state.update({"v-doc": document_id, "v-data": data, "v-meta": meta})
    st.session_state.update({"v-page": page, "v-cited": page, "v-excerpt": excerpt})
    pdf_viewer()


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def _chapter(c: dict) -> str:
    return gen.chapter_label(c.get("chapter_number"), c["chapter_name"])


def render_citations(citations: list[dict], key: str, labels_by_doc: dict, header: str) -> None:
    st.caption(header)
    for c in citations:
        label = f"[{c['number']}] {c['document_name']} · {_chapter(c)[:70]} · {page_text(c, labels_by_doc.get(c['document_id']))}"
        if st.button(label, key=f"{key}-open-{c['number']}", help="Open this page in the textbook"):
            open_viewer(c["document_id"], c["page"], c.get("excerpt", ""))
    for c in citations:
        with st.expander(f"Source excerpt [{c['number']}] · {_chapter(c)[:60]}"):
            st.markdown(f"> {c['excerpt']}")
            st.caption(f"{c['document_name']} · {_chapter(c)} · {page_text(c, labels_by_doc.get(c['document_id']))}")


def render_trace(trace: dict) -> None:
    with st.expander("How this answer was found"):
        if trace["standalone_question"] != trace["original_question"]:
            st.markdown(f"**Understood as:** {trace['standalone_question']}")
        method = trace["chapter_method"]
        if trace["chapters"]:
            names = "; ".join(f"{gen.chapter_label(c['number'], c['name'])} ({c['document']})" for c in trace["chapters"])
            st.markdown(f"**Chapter filter ({method}):** {names}")
        else:
            st.markdown(f"**Chapter filter:** none - {trace['chapter_note'] or 'no chapter identified'}")
        st.markdown(f"**Searched:** {trace['scope_description']}")
        if trace["fallback"]:
            st.caption(f"Fell back to all selected books because {trace['fallback_reason']}.")
        crit = trace.get("critique")
        if crit:
            verdict = "revised once after self-critique" if trace["regenerated"] else "passed self-critique"
            st.markdown(f"**Self-check:** {verdict}")
            when = "in the first draft (fixed in the revision)" if trace["regenerated"] else ""
            for k, title in (("unsupported_claims", "Unsupported claims"), ("incorrect_citations", "Incorrect citations"), ("missing_from_answer", "Missing points")):
                if crit.get(k):
                    st.caption(f"{title} found {when}: ".replace(" : ", ": ") + " | ".join(crit[k]))
        for p in trace.get("problems", []):
            st.caption(f"Citation check: {p}")
        if trace.get("hyde_passage"):
            st.markdown("**HyDE passage** (a hypothetical answer used only to find similar textbook text - it is *not* evidence)")
            st.caption(trace["hyde_passage"])
        st.caption(f"{len(trace['sources'])} textbook passages used · {trace['seconds']} s")


def render_assistant(msg: dict, idx: int, labels_by_doc: dict) -> None:
    meta = msg.get("meta") or {}
    status = meta.get("status")
    if status == "error":
        st.error(msg["content"])
        return
    if status == "abstained":
        st.warning(msg["content"])
    else:
        st.markdown(msg["content"])
        if status == "partial":
            st.info(f"**Partial answer.** The textbook does not fully cover this. {meta.get('explanation', '')}")
    if meta.get("citations"):
        render_citations(meta["citations"], f"m{idx}", labels_by_doc, "Sources - click to open the textbook at that page")
    if status == "abstained" and meta.get("closest"):
        render_citations(meta["closest"], f"m{idx}c", labels_by_doc, "Closest passages searched (not enough to answer the question)")
    if meta.get("trace"):
        render_trace(meta["trace"])


# --------------------------------------------------------------------------- #
# Sidebar: library, upload, scope
# --------------------------------------------------------------------------- #
def sidebar(pipeline: RAGPipeline) -> tuple[list[dict], list[str], str]:
    store = pipeline.store
    with st.sidebar:
        st.title("📚 Exam Prep")
        if st.button("➕ New chat", use_container_width=True):
            st.query_params["sid"] = uuid.uuid4().hex[:12]
            st.rerun()
        view = st.radio("View", ["💬 Ask", "📖 Chapters", "📊 Evaluation"], label_visibility="collapsed")

        st.divider()
        st.subheader("Upload textbooks")
        st.session_state.setdefault("uploader_key", 0)
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key=f"up-{st.session_state.uploader_key}", label_visibility="collapsed")
        for kind, text in st.session_state.pop("upload_msgs", []):
            (st.success if kind == "ok" else st.error)(text)
        if st.button("Add to library", type="primary", disabled=not files, use_container_width=True):
            msgs = []
            for f in files:
                bar = st.progress(0.0, text=f.name)
                try:
                    rec = pipeline.ingest(f.getvalue(), f.name, progress=lambda m, p, b=bar, n=f.name: b.progress(min(p, 1.0), text=f"{n}: {m}"))
                    msgs.append(("ok", f"Added “{rec['name']}”: {rec['num_chapters']} chapters, {rec['num_chunks']} passages."))
                except EXPECTED_ERRORS as e:
                    msgs.append(("err", f"{f.name}: {e}"))
                except Exception as e:  # unexpected - keep the app usable and tell the student
                    log.exception("Ingestion failed")
                    msgs.append(("err", f"{f.name}: unexpected error ({e.__class__.__name__}: {e})"))
                bar.empty()
            st.session_state["upload_msgs"] = msgs
            st.session_state.uploader_key += 1
            st.rerun()

        st.divider()
        st.subheader("Your textbooks")
        docs = store.list_documents()
        if not docs:
            st.caption("No textbooks yet. Upload a PDF above.")
        if docs and not st.session_state.get("index_ready"):
            status = store.vector_index_status()
            if status == "ready":
                st.session_state["index_ready"] = True
            else:
                st.warning("The Atlas vector index is still building (about a minute after the first upload). Questions will work once it is ready - reload then.")
        selected: list[str] = []
        for d in docs:
            use = st.checkbox(d["name"], value=True, key=f"use-{d['document_id']}", help="Include this book when answering")
            st.caption(f"{d['num_pages']} pages · {d['num_chapters']} chapters" + (" · ⚠️ see Chapters tab" if d.get("warnings") else ""))
            if use:
                selected.append(d["document_id"])
            with st.popover("Remove", use_container_width=True):
                st.write(f"Delete “{d['name']}” and its index?")
                if st.button("Delete permanently", key=f"del-{d['document_id']}"):
                    pipeline.delete_document(d["document_id"])
                    st.session_state.pop("v-doc", None)
                    st.rerun()
    return docs, selected, view


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def view_ask(pipeline: RAGPipeline, docs: list[dict], selected: list[str], sid: str) -> None:
    st.header("Ask your textbook")
    labels_by_doc = {d["document_id"]: d.get("page_labels") or {} for d in docs}
    if not docs:
        st.info("Upload a textbook PDF in the sidebar to get started. Answers cite the textbook name, chapter and page, and each citation opens the page.")
        return
    if not selected:
        st.warning("Tick at least one textbook in the sidebar.")
    elif not st.session_state.messages:
        st.caption("Try: “Explain the second law of thermodynamics from chapter 1”, “What does chapter 3 say about inheritance?”, then a follow-up like “and how does it differ from linkage?”")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m, i, labels_by_doc)

    if prompt := st.chat_input("Ask a question about your textbook…", disabled=not selected):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding the right chapter, searching and checking the answer…"):
                    reply = pipeline.ask(prompt, sid, selected)
                msg = {"role": "assistant", **reply}
            except EXPECTED_ERRORS as e:
                msg = {"role": "assistant", "content": f"⚠️ {e}", "meta": {"status": "error"}}
            except Exception as e:
                log.exception("Answering failed")
                msg = {"role": "assistant", "content": f"⚠️ Unexpected error ({e.__class__.__name__}): {e}", "meta": {"status": "error"}}
            st.session_state.messages.append(msg)
            render_assistant(msg, len(st.session_state.messages) - 1, labels_by_doc)


def view_chapters(pipeline: RAGPipeline, docs: list[dict]) -> None:
    st.header("Chapters")
    if not docs:
        st.info("Upload a textbook to see its chapter structure.")
        return
    d = st.selectbox("Textbook", docs, format_func=lambda x: x["name"])
    rec = pipeline.store.get_document(d["document_id"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", rec["num_pages"])
    c2.metric("Chapters", rec["num_chapters"])
    c3.metric("Passages", rec["num_chunks"])
    c4.metric("Structure from", {"toc": "PDF bookmarks", "printed_toc": "Contents page", "none": "Not found"}.get(rec.get("structure_source"), "Older upload"))
    for w in rec.get("warnings", []):
        st.warning(w)
    if any("num_chunks" not in ch for ch in rec["chapters"]):  # saved by an earlier version of the app
        st.warning("This book was uploaded with an older version of the app. Delete it in the sidebar and upload it again.")
        return
    labels = rec.get("page_labels") or {}
    for ch in rec["chapters"]:
        a, b = ch["page_start"], ch["page_end"]
        pages = f"pp. {labels.get(str(a), a)}–{labels.get(str(b), b)}" + (f" (PDF pages {a}–{b})" if str(a) in labels else "")
        left, right = st.columns([6, 1])
        left.markdown(f"**{gen.chapter_label(ch['number'], ch['name'])}**  \n<small>{pages} · {ch['num_chunks']} passages</small>", unsafe_allow_html=True)
        if right.button("Open", key=f"ch-{ch['chapter_id']}"):
            open_viewer(rec["document_id"], ch["page_start"])


def view_eval(pipeline: RAGPipeline, docs: list[dict]) -> None:
    st.header("Retrieval evaluation")
    st.caption("Compares BM25, vector search, hybrid RRF and hybrid + HyDE + reranking on questions whose relevant passage is known.")
    if not docs:
        st.info("Upload a textbook first.")
        return
    d = st.selectbox("Textbook", docs, format_func=lambda x: x["name"], key="eval-doc")
    store = pipeline.store
    items = store.get_eval_set(d["document_id"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Generate a test set with Gemini**")
        n = st.number_input("Number of questions", 5, 40, 15)
        if st.button("Generate questions"):
            bar = st.progress(0.0, text="Writing questions…")
            try:
                items = pipeline.build_eval_set(d["document_id"], int(n), progress=lambda i, t: bar.progress(i / t, text=f"Question {i}/{t}"))
                if len(items) < int(n):
                    st.info(f"Only {len(items)} passages in this book were long enough to write questions from.")
            except EXPECTED_ERRORS as e:
                st.error(str(e))
            bar.empty()
    with right:
        st.markdown("**…or upload your own** (JSON: `[{\"question\": \"…\", \"relevant_chunk_ids\": [\"…\"]}]`)")
        up = st.file_uploader("Dataset", type=["json"], label_visibility="collapsed")
        if up and st.button("Import dataset"):
            try:
                items = pipeline.import_eval_set(d["document_id"], json.loads(up.getvalue()))
                st.success(f"Imported {len(items)} questions.")
            except (ValueError, TypeError) as e:
                st.error(f"Could not import the dataset: {e}")

    if not items:
        st.info("No test questions yet - generate or import some.")
        return
    with st.expander(f"Test questions ({len(items)})"):
        st.dataframe(pd.DataFrame([{"question": i["question"], "relevant chunk": ", ".join(i["relevant_chunk_ids"])} for i in items]), hide_index=True)

    use_filter = st.checkbox("Use chapter-first filtering (applied identically to all four systems)", value=True)
    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Evaluating…")
        try:
            st.session_state["eval_result"] = pipeline.evaluate(items, [d["document_id"]], use_filter, progress=lambda i, t: bar.progress(i / t, text=f"Question {i}/{t}"))
        except EXPECTED_ERRORS as e:
            st.error(str(e))
        bar.empty()

    res = st.session_state.get("eval_result")
    if res:
        cols = ["Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10", "Empty rate"]
        df = pd.DataFrame(res["metrics"]).T[cols]
        st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0, subset=cols[:-1], color="#d4edda"))
        st.bar_chart(df[["Recall@5", "MRR", "nDCG@5"]])
        st.download_button("Download results (CSV)", df.to_csv().encode(), "retrieval_eval.csv", "text/csv")
        cf = res.get("chapter_filter")
        if cf:
            correct = "n/a" if cf["correct_when_identified"] is None else f"{cf['correct_when_identified']:.0%}"
            st.caption(f"Chapter filter: identified a chapter for {cf['identified_rate']:.0%} of questions (correct when identified: {correct}); fell back to the whole book for {cf['fallback_rate']:.0%}.")
        st.caption(f"{res['questions']} questions. Generated questions are written from a single passage, so lexical methods such as BM25 may look stronger than on real student questions.")


# --------------------------------------------------------------------------- #
def main() -> None:
    pipeline = get_pipeline()
    sid = ensure_session()
    docs, selected, view = sidebar(pipeline)
    if view.endswith("Ask"):
        view_ask(pipeline, docs, selected, sid)
    elif view.endswith("Chapters"):
        view_chapters(pipeline, docs)
    else:
        view_eval(pipeline, docs)


if __name__ == "__main__":  # Streamlit runs the script as __main__; importing app (e.g. in tests) has no side effects
    main()

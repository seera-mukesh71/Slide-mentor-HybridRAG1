# Exam Prep Textbook Assistant

A Streamlit app that lets students upload textbook PDFs and ask exam-preparation questions. Answers are built **only** from retrieved textbook passages and carry clickable citations (textbook, chapter, page) that open the PDF at the cited page.

**Stack:** Python · Streamlit · Gemini 2.5 Flash (generation, query rewriting, HyDE, chapter selection, self-critique) · Gemini embeddings · PyMuPDF · MongoDB Atlas Vector Search · BM25 (`rank-bm25`) · cross-encoder reranker (`sentence-transformers`).

```
student-rag/
├── app.py            Streamlit UI (chat, library, chapters, evaluation, page viewer)
├── ingestion.py      PDF validation, table-of-contents chapters, page-tagged chunks, embedding, indexing
├── retrieval.py      Chapter-first filtering, HyDE, BM25, vector search, RRF, reranking
├── generation.py     Gemini client, grounded answer, self-critique, citation validation, abstention
├── rag_pipeline.py   Connects the whole workflow (ask / ingest / evaluate)
├── storage.py        MongoDB access, conversation memory, evaluation metrics
└── tests/            test_retrieval.py, test_pipeline.py
```

## How a question is answered

```
question ──► rewrite follow-up (last 5 messages)
         ──► identify chapter(s)  number → title → fuzzy → Gemini        (uses the chapter names stored at upload)
         ──► ONE scope = document ids + chapter ids
                ├─► BM25 over only the in-scope chunks
                └─► vector search over HyDE embedding, filter inside Atlas $vectorSearch
         ──► RRF fusion ──► cross-encoder rerank
         ──► (scope too thin? repeat once over all selected books)
         ──► the top chunks (nothing else) go to the LLM
         ──► Gemini answers from those passages, citing [S#]
         ──► same Gemini model critiques it; at most ONE regeneration
         ──► Python validates every citation id and page against the stored metadata
         ──► answered │ partial answer │ abstained (with the closest passages)
```

Key behaviours

* **Chapter-first.** If a chapter is confidently identified (e.g. "chapter 3", "Ch. IV", "chapters 2–4", a chapter title, a typo'd title, or Gemini picking from the real chapter list) the filter is applied *before* both BM25 and vector search. The corpus handed to BM25 is already filtered, and Atlas applies the same filter inside `$vectorSearch`. Several chapters can be selected. If no chapter is identified, the question is ambiguous, or the filtered search returns fewer than 3 candidates (or the best reranker score is below `MIN_RERANK_SCORE`), the search is repeated over all selected books. The UI's "How this answer was found" panel shows what happened.
* **HyDE is never evidence.** The hypothetical passage is only embedded to find similar real text. It is not sent to the answering model, not used by BM25, not used by the reranker.
* **Exact page provenance.** Every chunk stores a `page_map`. The model names the page of each citation, Python checks it against the pages that actually exist in that source (the chunk) and corrects or drops anything invalid, including page numbers the model wrote in prose.
* **Honest abstention.** If the passages do not contain the answer, the app says so, explains where it searched, and shows the closest passages instead of inventing an answer. Partial coverage produces a partial answer with a notice.

## 1. Prerequisites

* Python 3.11 or 3.12 recommended (3.10–3.13 should work; the project was developed on 3.13)
* A Gemini API key
* A MongoDB Atlas cluster (the free **M0** tier supports Vector Search)

## 2. Install

Windows (PowerShell, in VS Code's terminal):

```powershell
cd student-rag
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
code .env
```

macOS / Linux:

```bash
cd student-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

`sentence-transformers` pulls in PyTorch (large). To keep the download smaller on a machine without a GPU, install the CPU build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu` and then `pip install -r requirements.txt`.
The reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) is downloaded from Hugging Face the first time a question is answered.

## 3. Gemini API setup

1. Open https://aistudio.google.com/apikey and create an API key.
2. Put it in `.env` as `GEMINI_API_KEY=...`. One key is used for everything (generation, rewriting, HyDE, self-critique, chapter selection and embeddings). There is no Google login and no per-user key.
3. The free tier has small per-minute quotas. Ingesting a large book makes many embedding calls; the app retries with back-off on HTTP 429/5xx and shows a clear message if the quota is exhausted. Enabling billing on the key removes most limits.

Defaults: `GEMINI_MODEL=gemini-2.5-flash`, `EMBED_MODEL=gemini-embedding-001`, `EMBED_DIM=768`. The generation model must be a 2.5-family model (the code sets a thinking budget).

## 4. MongoDB Atlas setup

1. Sign in at https://cloud.mongodb.com, create a project and **Build a Database → M0 (Free)**.
2. **Database Access → Add New Database User** (password authentication). Give it read/write access to the database.
3. **Network Access → Add IP Address**: add your current IP (for a deployed server, add that server's outbound IP; `0.0.0.0/0` opens the cluster to the internet, so only use it with a strong password and you accept the risk).
4. **Connect → Drivers**: copy the connection string, replace `<password>` (URL-encode special characters, e.g. `@` → `%40`) and set it as `MONGODB_URI` in `.env`.
5. Start the app. On startup it creates the regular indexes and the Atlas **vector search index** `chunks_vector` on `<MONGODB_DB>.chunks` automatically. After the first upload, the index needs about a minute to become queryable; the sidebar warns until it is ready.

If you prefer to create the vector index by hand (Atlas → Search & Vector Search → Create → Vector Search → JSON editor, collection `student_rag.chunks`, index name `chunks_vector`):

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "document_id" },
    { "type": "filter", "path": "chapter_id" }
  ]
}
```

The two `filter` fields are what make chapter filtering happen *inside* the vector search. `numDimensions` must equal `EMBED_DIM`; if you change `EMBED_DIM`, drop the index and re-upload your books.

## 5. Run

```bash
streamlit run app.py
```

1. Upload one or more text-based PDFs in the sidebar and click **Add to library**.
2. Open **📖 Chapters** to check the detected structure.
3. Ask questions in **💬 Ask**. Click a citation button to open the textbook at that page (the cited text is highlighted on the page); expand "Source excerpt" to read the exact passage.
4. Tick/untick books in the sidebar to choose which ones are searched.

The chat session id is kept in the URL (`?sid=...`), so refreshing the browser resumes the conversation. **➕ New chat** starts a fresh one.

### First-run checklist (live services)

The automated tests use fakes (see below), so run these once to confirm your keys and services work:

```bash
python -c "import generation as g; print(g.generate_text('Reply with the word OK'))"
python -c "import generation as g; print(len(g.embed_texts(['hello'])[0]))"      # expect 768
python -c "from storage import Store; import generation as g; Store().ensure_indexes(g.EMBED_DIM); print('mongo ok')"
python -c "import retrieval; print(retrieval.get_reranker().predict([('what is entropy','Entropy measures disorder')]))"
```

## 6. Evaluation

Open **📊 Evaluation**, pick a book, then either **Generate questions** (Gemini writes one question per sampled passage, so the relevant chunk is known; every chapter is covered and every third question names its chapter, like real students do) or upload your own JSON:

```json
[{ "question": "What does rubisco do?", "relevant_chunk_ids": ["<chunk_id from the chunks collection>"] }]
```

**Run evaluation** compares BM25, vector search, hybrid RRF and hybrid + HyDE + rerank, all with the same chapter-filter setting (checkbox), and reports Recall@3/5/10, MRR, nDCG@5/10 and the empty-retrieval rate, plus how often a chapter was identified and whether it was the right one. Metric functions live in `storage.py` and are unit-tested against hand-computed values.

Caveat: generated questions are written from a single passage, so they share vocabulary with it and favour BM25 more than real student questions would.

## 7. Tests

```bash
python -m pytest -q
```

No network, Gemini or Atlas is needed. `FakeStore` is the real `Store` class on `mongomock` with only `$vectorSearch` replaced by an in-memory cosine search that applies the *same* scope filter; a separate test inspects the real `$vectorSearch` pipeline to prove the chapter/document filter sits inside that stage. Gemini is replaced by a scripted fake.

* `tests/test_retrieval.py`: PDF ingestion (PDF bookmarks and printed contents page, printed-to-PDF page mapping, chapters that start mid-page, books without a TOC, page provenance, header/footer removal, validation errors), chapter identification (number, title, fuzzy, Gemini, multiple chapters, generic-title guard, multi-book), proof that filtering happens before BM25 and vector search (a *pre*-filter test that a post-filter would fail), fallbacks, RRF, reranking, HyDE isolation, metrics.
* `tests/test_pipeline.py`: the full flow upload → chapter metadata → chapter identification → filtered retrieval → HyDE + BM25 + vector → RRF → rerank → grounded answer → self-critique → validated page citations; single regeneration; citation repair; abstention; partial answers; conversation memory; Gemini retry handling; evaluation; the Streamlit app driven headlessly (citation button → viewer → page navigation).

## 8. Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – (required) | Gemini API key |
| `MONGODB_URI` | – (required) | Atlas connection string |
| `MONGODB_DB` | `student_rag` | Database name |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation / rewrite / HyDE / critique model |
| `EMBED_MODEL` / `EMBED_DIM` | `gemini-embedding-001` / `768` | Embedding model and size (must match the vector index) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `MIN_RERANK_SCORE` | `-4.0` | Best reranker logit below which a chapter-scoped result counts as insufficient |
| `MAX_PDF_MB` / `MAX_PDF_PAGES` | `100` / `1500` | Upload limits (Streamlit's own limit is 200 MB: `--server.maxUploadSize`) |

## 9. Deployment

The app is a single Streamlit process; all state is in MongoDB Atlas, so any container host works. Set `GEMINI_API_KEY` and `MONGODB_URI` as environment variables / secrets on the host, never in the repository.

Needs: **at least ~2 GB RAM** (PyTorch + the reranker), outbound HTTPS to Gemini, Hugging Face (first run) and Atlas, and the host's outbound IP in Atlas **Network Access**.

Example `Dockerfile` (a template; not built or run during development):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t student-rag .
docker run -p 8501:8501 --env-file .env student-rag
```

Hosts: Render, Railway, Fly.io, Google Cloud Run or Hugging Face Spaces (Docker) all work with that image (give it ≥2 GB). Streamlit Community Cloud runs `streamlit run app.py` directly from a GitHub repo and exposes its *Secrets* as environment variables, but its small memory limit may be too tight for PyTorch.

**Security note:** by design there are no accounts. Everyone who can open the URL shares one library, can delete books, and uses your Gemini quota. Keep the deployment private (VPN, SSO proxy, or host-level access control) or run it locally.

## 10. Known limitations and anything incomplete

* **Not exercised against live services during development.** The development environment had no Gemini key, no Atlas cluster and no access to Hugging Face / PyTorch hosts. Everything is covered by the fake-based tests above, but the real Gemini calls (structured output, embeddings), Atlas `$vectorSearch` and the real cross-encoder have not been run end to end. Use the first-run checklist, and report/fix anything that differs.
* **`MIN_RERANK_SCORE` (-4.0) is an uncalibrated guess.** It only decides when a chapter-scoped result is thin enough to widen the search; tune it on your own books with the evaluation tab.
* **Scanned PDFs are rejected** (no OCR). Tables, equations and figures are not extracted as structured content; only their text is.
* **Chapters come only from the table of contents**: the PDF's bookmarks, or else the printed contents page (rows of `title ..... page`; if the row parser cannot read the layout, Gemini reads the contents text). The printed page numbers are matched to PDF pages by locating chapter titles in the book. A book with no table of contents (for example a lecture-slide PDF) is one chapter, so chapter-first filtering is a no-op for it. Contents entries that wrap onto a second line, or sections without numbers in a printed contents page, can be mis-levelled; check the Chapters view after each upload. The Gemini contents-page fallback has not been run against the live API.
* **Reading order** follows the PDF's text stream; unusual multi-column layouts can interleave text.
* **PDF viewer:** the cited page is rendered as an image (reliable in every browser, with best-effort highlighting of the cited text). It is not a selectable-text viewer; **Download PDF** gives the original file.
* **The self-critique uses the same model** as the answer (as specified), so it shares that model's blind spots; the Python citation validation is what guarantees ids and pages are real.
* **Shared library, no accounts** (see the security note). Chat sessions are separated only by the random id in the URL.
* **Scale:** BM25 is rebuilt per question over the in-scope chunks (no caching, by design). That is fast for a chapter and fine for a few books, but searching many large books without a chapter filter will be slower.
* Page numbers are PDF page indexes; if the PDF defines printed page labels they are shown too.

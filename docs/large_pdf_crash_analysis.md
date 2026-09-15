# Large PDF Upload — Server Freeze & Crash Analysis

**Date:** 2026-09-15  
**Scope:** What specific code causes the FastAPI server to freeze and crash when a user uploads a 1000-page PDF

---

## 🔴 Critical Bottleneck #1: Synchronous `ingest_document()` blocks the event loop

**File:** `api.py` — Line 110  
**Endpoint:** `POST /upload`

```python
# api.py line 95-110
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    ...
    result = ingest_document(tmp_path, source_name=file.filename)  # ← SYNC call in async endpoint
```

### Why this crashes the server

`ingest_document()` is a **regular synchronous function** called directly inside an `async def` endpoint. In FastAPI, `async def` endpoints run on the **main asyncio event loop**. When a synchronous function blocks inside that loop, **every other request is frozen** — health checks, chat, streaming — everything stops until `ingest_document()` returns.

For a 1000-page PDF, this function takes **15+ minutes** to complete (see Bottleneck #2), meaning the server is completely unresponsive for that entire duration.

### Fix approach

- Use `asyncio.to_thread(ingest_document, ...)` to offload to a thread pool, OR
- Use FastAPI `BackgroundTasks` so the endpoint returns immediately and processing happens asynchronously, OR
- Convert to a `def` (non-async) endpoint so FastAPI auto-runs it in a thread pool

---

## 🔴 Critical Bottleneck #2: Sequential embedding — one API call per chunk

**File:** `rag.py` — Lines 107-126  
**Function:** `ingest_document()`

```python
# rag.py lines 107-126
for page in pages:                               # 1000 iterations for 1000 pages
    page_text = page["text"]
    chunks = chunk_text(page_text)                # ~2-5 chunks per page

    for chunk_idx, chunk in enumerate(chunks):
        all_embeddings.append(get_embedding(chunk))  # ← ONE API call per chunk
```

### Why this is the real killer

For a 1000-page PDF:
- `chunk_text()` produces ~2-5 chunks per page → **~3,000 total chunks**
- Each `get_embedding(chunk)` makes a **single OpenAI API call** (~200-500ms each)
- **3,000 chunks × ~300ms = ~15 minutes** of sequential, blocking API calls

This entire time, the event loop is blocked (see Bottleneck #1), so the server appears completely dead.

### The `get_embedding()` function (rag.py lines 43-56)

```python
def get_embedding(text: str) -> list[float]:
    config = get_config()
    if config.EMBEDDING_PROVIDER == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text                        # ← single text, single API call
        )
        return response.data[0].embedding
    else:
        model = _get_local_model()
        return list(model.embed(text))[0].tolist()
```

### Fix approach

- **Batch embeddings:** OpenAI's API accepts a list of texts in a single call (up to 2048). Send chunks in batches of 100-500 instead of one at a time.
- **Async embedding calls:** Use the async OpenAI client for concurrent requests.

---

## 🔴 Critical Bottleneck #3: Entire PDF loaded into memory at once

**File:** `api.py` — Lines 103-106

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    content = await file.read()   # ← reads ENTIRE file into RAM
    tmp.write(content)
    tmp_path = tmp.name
```

### Why this is dangerous

A 1000-page PDF can be **50-500MB**. `file.read()` loads the **entire file** into memory before writing it to disk. If multiple users upload large PDFs simultaneously, this causes an **out-of-memory (OOM) crash**.

### Fix approach

- Use `shutil.copyfileobj(file.file, tmp)` to stream the file to disk in chunks instead of loading it all into memory.
- Or use `file.read(chunk_size)` in a loop.

---

## 🟡 Secondary Issue: Massive single ChromaDB upsert

**File:** `rag.py` — Lines 129-135

```python
if all_chunks:
    collection.upsert(
        documents=all_chunks,       # ~3,000 documents
        embeddings=all_embeddings,   # ~3,000 embedding vectors (1536 floats each)
        metadatas=all_metadatas,
        ids=all_ids
    )
```

### Why this matters

All ~3,000 chunks + their embeddings are accumulated in lists, then written in a **single atomic upsert**. This **doubles the memory footprint** at write time because ChromaDB needs to process the entire batch. For 3,000 embeddings of dimension 1536:

> 3,000 × 1,536 floats × 4 bytes = ~18MB just for embeddings  
> Plus document text, metadata, and ChromaDB internal overhead

### Fix approach

- Upsert in batches (e.g., 100-200 chunks at a time) to limit peak memory usage.

---

## Crash Chain Summary

```
User uploads 1000-page PDF
  │
  ├─ file.read() → loads entire PDF into RAM              [MEMORY SPIKE]
  │
  ├─ ingest_document() called synchronously                [EVENT LOOP BLOCKED]
  │   │
  │   ├─ DocumentParser.parse_pdf() reads all pages        [more memory]
  │   │
  │   ├─ for each page → chunk_text()                      [~3,000 chunks]
  │   │   └─ for each chunk → get_embedding()              [~3,000 sequential API calls]
  │   │       └─ ~300ms per call × 3,000 = ~15 MINUTES     [server frozen]
  │   │
  │   └─ collection.upsert(all 3,000 at once)              [memory doubles]
  │
  └─ ALL other endpoints unresponsive during this time     [SERVER "CRASHED"]
```

---

## Files Involved

| File | Lines | Issue |
|------|-------|-------|
| `api.py` | 104 | `file.read()` loads entire PDF into memory |
| `api.py` | 110 | Sync `ingest_document()` blocks async event loop |
| `rag.py` | 107-126 | Sequential embedding loop (3,000+ API calls) |
| `rag.py` | 43-56 | `get_embedding()` handles only single text |
| `rag.py` | 129-135 | Single massive ChromaDB upsert |
| `document_parser.py` | 42-52 | All pages parsed synchronously (minor) |

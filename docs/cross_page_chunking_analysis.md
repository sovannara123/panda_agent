# Cross-Page Chunking Failure — Lost Answers at Page Boundaries

**Date:** 2026-09-15  
**Scope:** Why `search_knowledge_base()` fails when the answer spans the end of one page and the beginning of the next

---

## The Bug

### Pages are chunked independently — no cross-page overlap

**File:** `rag.py` — Lines 107-112

```python
for page in pages:                   # ← each page processed INDEPENDENTLY
    page_text = page["text"]
    chunks = chunk_text(page_text)   # ← chunking happens WITHIN a single page only
```

**File:** `document_parser.py` — Lines 42-52

```python
for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = " ".join(text.split())
    if text.strip():
        pages.append({
            "page_number": page_num,
            "text": text              # ← each page is a separate, isolated text blob
        })
```

Pages are extracted individually and **never joined**. The `chunk_text()` overlap (50 words) only works **within** a single page's text. There is **zero overlap at page boundaries**.

---

## Concrete Example

A PDF where the answer to "What is the refund deadline?" spans pages 4-5:

> **Page 4 (last paragraph):** *"...customers may request a full refund within"*
>
> **Page 5 (first paragraph):** *"30 business days of the original purchase date, provided the item is unopened..."*

### What the chunker produces

```
Page 4 chunks:
  chunk0: "Chapter 3: Return Policy. Items purchased online..."
  chunk1: "...shipping costs are non-refundable. However customers may request a full refund within"
                                                                                        ↑ CUTS HERE

                                    ← NO OVERLAP ACROSS THIS BOUNDARY →

Page 5 chunks:
  chunk0: "30 business days of the original purchase date provided the item is unopened..."
           ↑ CONTINUES HERE
  chunk1: "...exceptions apply to clearance items..."
```

---

## Why Retrieval Fails

**File:** `rag.py` — Lines 195-206 (`search_knowledge_base`)

```python
query_embedding = get_embedding(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=k    # default k=3
)
```

User asks: **"What is the refund deadline?"**

| Chunk | Content | Semantic match? |
|-------|---------|-----------------|
| Page 4, last chunk | *"...request a full refund within"* | ⚠️ Has "refund" but no time period |
| Page 5, first chunk | *"30 business days...item is unopened"* | ⚠️ Has the number but lacks "refund" context |

**Neither chunk's embedding is a strong match** for the full question because:

1. Page 4's chunk has "refund" but no deadline value → weak match
2. Page 5's chunk has "30 days" but the word "refund" is missing → weak match
3. ChromaDB may return **unrelated chunks** that mention "deadline" or "refund" elsewhere
4. Even if both are returned, the LLM receives them as **separate, unconnected pieces** with no indication they are consecutive

---

## Visual: The Gap

```
Page 4: [...chunk0...][...chunk1 (overlap)...][...last chunk (ends abruptly)]
                                                                             ← GAP (zero overlap)
Page 5: [first chunk (starts abruptly)...][...chunk1 (overlap)...][...chunk2...]
```

Within-page overlap (50 words) works fine. Cross-page overlap is **nonexistent**.

---

## Root Cause

The chunking architecture treats each page as an **isolated document**. The `for page in pages` loop in `ingest_document()` never concatenates adjacent pages before chunking.

---

## Fix Approach

**Option A — Concatenate all pages before chunking:**

```python
full_text = " ".join(page["text"] for page in pages)
chunks = chunk_text(full_text, chunk_size=500, overlap=50)
```

This ensures the overlap window spans page boundaries naturally. Downside: you lose per-page metadata on each chunk.

**Option B — Add a cross-page boundary chunk:**

After chunking each page, create an extra "bridge" chunk that combines the last ~250 words of page N with the first ~250 words of page N+1. This preserves per-page chunking while also covering boundaries.

**Option C — Semantic chunking:**

Instead of splitting by word count, split by paragraphs or sentences using NLP sentence boundary detection. This respects natural content boundaries better than fixed word windows.

---

## Impact Summary

| Aspect | Severity |
|--------|----------|
| Answer quality | 🔴 **Critical** — complete answers split across pages are unfindable |
| Frequency | 🟡 **Common** — any content flowing across page breaks |
| User visibility | 🔴 **None** — user gets a wrong/incomplete answer, not an error |
| Affected function | `search_knowledge_base()` returns irrelevant or partial results |

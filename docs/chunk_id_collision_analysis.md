# ChromaDB Chunk ID Collision — Silent Data Loss Bug

**Date:** 2026-09-15  
**Scope:** What happens when 2 users upload a file with the same name (e.g. `policy.pdf`) — simultaneously or at any time

---

## The Bug

### Chunk ID generation has no uniqueness per upload

**File:** `rag.py` — Line 115

```python
chunk_id = f"{source}_page{page_number}_chunk{chunk_idx}"
```

The `source` value comes from the **filename** passed by the uploader:

```python
# rag.py line 97
source = source_name or metadata["filename"]

# api.py line 110
result = ingest_document(tmp_path, source_name=file.filename)  # e.g. "policy.pdf"
```

There is **no user ID, session ID, UUID, or timestamp** in the chunk ID. The ID is **entirely deterministic** based on filename + page number + chunk index.

---

## Collision Proof

When **any two users** upload a file named `policy.pdf`, both generate identical IDs:

| User A's chunk IDs | User B's chunk IDs |
|---|---|
| `policy.pdf_page1_chunk0` | `policy.pdf_page1_chunk0` |
| `policy.pdf_page1_chunk1` | `policy.pdf_page1_chunk1` |
| `policy.pdf_page2_chunk0` | `policy.pdf_page2_chunk0` |
| `policy.pdf_page3_chunk0` | `policy.pdf_page3_chunk0` |
| ... | ... |

**Every single ID collides.**

---

## What ChromaDB does with the collision

**File:** `rag.py` — Lines 130-135

```python
collection.upsert(          # ← upsert = insert OR UPDATE
    documents=all_chunks,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids              # ← identical IDs from both uploads
)
```

`upsert` means: if the ID already exists, **overwrite it silently**. No error. No warning.

### Result: **Last write wins**

1. User A uploads `policy.pdf` (company HR policy) → stored successfully
2. User B uploads `policy.pdf` (insurance policy) → **overwrites all of User A's chunks**
3. User A gets a `200 OK` — their data appears to be saved
4. When User A searches, they get results from **User B's document**
5. User A's document is **permanently lost** with zero indication

---

## This is NOT a race condition — it happens always

Even if User A uploads at 9:00 AM and User B uploads at 5:00 PM, the collision still occurs. Any two files with the same filename will overwrite each other, regardless of timing.

---

## Edge case: Different page counts

If User A's PDF has 10 pages and User B's has 5 pages:

- Pages 1-5: User B's content **overwrites** User A's
- Pages 6-10: User A's old chunks **survive as orphans**
- The knowledge base now contains a **Frankenstein mix** of both documents
- Search results return incoherent, mixed content from two unrelated PDFs

---

## Root Cause

The chunk ID formula lacks any upload-specific identifier:

```python
# Current — deterministic, collision-guaranteed for same filenames
chunk_id = f"{source}_page{page_number}_chunk{chunk_idx}"
```

## Fix Approach

Add a unique identifier per upload:

```python
import uuid

upload_id = str(uuid.uuid4())[:8]  # unique per upload session
chunk_id = f"{upload_id}_{source}_page{page_number}_chunk{chunk_idx}"
```

Or use a combination of user ID + timestamp + filename to ensure uniqueness while maintaining traceability.

---

## Impact Summary

| Aspect | Severity |
|--------|----------|
| Data loss | 🔴 **Critical** — silent, permanent |
| Error visibility | 🔴 **None** — both users get 200 OK |
| Trigger condition | 🔴 **Easy** — just same filename |
| Requires concurrency | ❌ No — happens even hours apart |
| Affects search results | 🔴 Yes — returns wrong document content |
